import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.transforms as T
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# -----------------------
# 1. 디바이스 설정
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# 2. 클래스 레이블 맵핑
# -----------------------
label_map = {0: "Anemia", 1: "Non-Anemia"}

# -----------------------
# 3. 모델 정의 및 가중치 로드
# -----------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for name, param in model.named_parameters():
        if not (name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )

    checkpoint_path = "best_fold3.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------
# 4. 전처리 정의
# -----------------------
IMG_SIZE = 224
val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# -----------------------
# 5. Bézier frame 좌표 계산 (크기 조절 가능)
# -----------------------
def get_conjunctiva_bezier_bbox(image_size):
    w, h = image_size
    
    # --- 이 부분을 조절하여 모바일 환경에 맞게 프레임 크기를 조정하였다. ---
    # frame_width_ratio: 프레임의 가로 폭 비율 (전체 너비 대비)
    # frame_height_ratio: 프레임의 세로 폭 비율 (전체 높이 대비)
    # center_x_ratio: 프레임 중앙의 가로 위치 (전체 너비 대비)
    # center_y_ratio: 프레임 중앙의 세로 위치 (전체 높이 대비)

    # 모바일 환경 고려: 프레임의 세로 폭을 더 줄이고, 살짝 위로 이동하여 눈에 맞춤
    frame_width_ratio = 0.15   
    frame_height_ratio = 0.05  

    center_x_ratio = 0.5
    center_y_ratio = 0.55 

    left_ratio = center_x_ratio - (frame_width_ratio / 2)
    right_ratio = center_x_ratio + (frame_width_ratio / 2)
    upper_ratio = center_y_ratio - (frame_height_ratio / 2)
    lower_ratio = center_y_ratio + (frame_height_ratio / 2)
    
    left = int(w * left_ratio)
    upper = int(h * upper_ratio)
    right = int(w * right_ratio)
    lower = int(h * lower_ratio)
    
    return left, upper, right, lower

# -----------------------
# 6. Bézier 곡선 점 계산 헬퍼 (결막 부분을 정확히 crop하기 위함)
# -----------------------
def cubic_bezier_points(p0, p1, p2, p3, n=200):
    t = np.linspace(0, 1, n)
    pts = []
    for ti in t:
        x = (1 - ti)**3 * p0[0] + 3 * (1 - ti)**2 * ti * p1[0] + 3 * (1 - ti) * ti**2 * p2[0] + ti**3 * p3[0]
        y = (1 - ti)**3 * p0[1] + 3 * (1 - ti)**2 * ti * p1[1] + 3 * (1 - ti) * ti**2 * p2[1] + ti**3 * p3[1]
        pts.append((int(x), int(y)))
    return pts

# -----------------------
# 7. Bézier 프레임 오버레이
# -----------------------
def draw_bezier_frame(cv_img: np.ndarray) -> np.ndarray:
    h, w, _ = cv_img.shape
    left, upper, right, lower = get_conjunctiva_bezier_bbox((w, h))
    
    frame_w = max(1, right - left) 
    frame_h = max(1, lower - upper) 

    # 윗 곡선 제어점
    p0 = (left, upper)
    p1 = (left + int(frame_w * 0.25), upper + int(frame_h * 0.4))
    p2 = (right - int(frame_w * 0.25), upper + int(frame_h * 0.4))
    p3 = (right, upper)
    top_curve = cubic_bezier_points(p0, p1, p2, p3, n=200)

    # 아랫 곡선 제어점
    p0_b = (right, upper)
    p1_b = (right + int(frame_w * 0.1), lower + int(frame_h * 0.05))
    p2_b = (left - int(frame_w * 0.1), lower + int(frame_h * 0.05))
    p3_b = (left, upper)
    bottom_curve = cubic_bezier_points(p0_b, p1_b, p2_b, p3_b, n=200)

    pts = np.array(top_curve + bottom_curve, dtype=np.int32)
    cv2.polylines(cv_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return cv_img

# -----------------------
# 8. Bézier 프레임 내부만 추출하고 외부를 흰색으로 마스킹
# -----------------------
def extract_and_mask_bezier_region(img_bgr: np.ndarray) -> Image.Image:
    # 이미지를 받아서 프레임 내부 영역만 추출하고, 외부를 흰색으로 마스킹합니다. (학습 모델의 특성 때문에)
    h, w, _ = img_bgr.shape
    left, upper, right, lower = get_conjunctiva_bezier_bbox((w, h))
    
    # 프레임의 폭과 높이가 0이 되지 않도록 최소값 설정
    frame_w = max(1, right - left)
    frame_h = max(1, lower - upper)

    # Bézier 곡선 좌표 계산
    p0 = (left, upper)
    p1 = (left + int(frame_w * 0.25), upper + int(frame_h * 0.4))
    p2 = (right - int(frame_w * 0.25), upper + int(frame_h * 0.4))
    p3 = (right, upper)
    top_curve = cubic_bezier_points(p0, p1, p2, p3, n=200)

    p0_b = (right, upper)
    p1_b = (right + int(frame_w * 0.1), lower + int(frame_h * 0.05))
    p2_b = (left - int(frame_w * 0.1), lower + int(frame_h * 0.05))
    p3_b = (left, upper)
    bottom_curve = cubic_bezier_points(p0_b, p1_b, p2_b, p3_b, n=200)

    polygon = np.array(top_curve + bottom_curve, dtype=np.int32)

    # 흰 배경 이미지 생성
    white_bg = np.ones_like(img_bgr) * 255

    # 마스크 생성 (Bézier 프레임 내부를 255로 채움)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # 마스크된 부분만 원본 이미지, 나머지는 흰색
    combined_img_bgr = np.where(mask[..., None] == 255, img_bgr, white_bg)
    
    # 마스킹된 이미지의 Bézier 프레임 영역의 바운딩 박스를 기준으로 크롭
    # 모델 입력 시 이미지 크기를 줄이기 위함
    cropped_final_bgr = combined_img_bgr[upper:lower, left:right]

    if cropped_final_bgr.shape[0] == 0 or cropped_final_bgr.shape[1] == 0:
        st.warning("경고: Bézier 프레임 영역이 너무 작거나 유효하지 않아 이미지를 크롭할 수 없습니다. 흰색 배경으로 대체합니다.")
        return Image.fromarray(np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255)

    # OpenCV BGR -> PIL RGB 변환
    return Image.fromarray(cv2.cvtColor(cropped_final_bgr, cv2.COLOR_BGR2RGB))

# -----------------------
# 9. 모델 예측 함수
# -----------------------
def predict_image(pil_img: Image.Image):
    x = val_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
        conf, pred = torch.max(probs, dim=0)
    return pred.item(), conf.item()

# -----------------------
# 10. 실시간 스트리밍 처리를 위한 Transformer
# -----------------------
class ConjunctivaProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_frame_bgr = None  # 마지막 원본 BGR 프레임을 저장할 변수

    def transform(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")  # BGR numpy
        img_bgr_processed = img_bgr 
        # Bézier 오버레이
        overlaid_img = draw_bezier_frame(img_bgr_processed.copy())
        self.last_frame_bgr = img_bgr.copy()  # 모델 예측을 위해 원본 프레임 저장
        return overlaid_img

# -----------------------
# 11. Streamlit 페이지 구성
# -----------------------
st.title("결막 사진으로 빈혈 예측 앱")

st.markdown(
    """
    **본 앱은 프레임 안의 결막 부분을 분석하여 빈혈 여부를 예측합니다.**
    
    **사용법:**
    1. 아래 'Start camera' 버튼을 눌러 카메라를 켜세요.
    2. 카메라 피드에 나타나는 **초록색 프레임 안에 결막 부분을 맞춰주세요.**
    3. '사진 찍기 & 예측' 버튼을 눌러 촬영하면, 프레임 안의 영역만 추출하여 예측합니다.
    """
)

# 11.1. webrtc_streamer 설정 (카메라 해상도 최대로 요청)
webrtc_ctx = webrtc_streamer(
    key="conjunctiva_capture_stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_processor_factory=ConjunctivaProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1920}, # Full HD
            "height": {"ideal": 1080},
            "frameRate": {"ideal": 30},
        },
        "audio": False,
    },
    async_processing=True,
)

# 11.2. ‘사진 찍기 & 예측’ 버튼
if webrtc_ctx.video_processor:
    if st.button("사진 찍기 & 예측"):
        # 마지막 원본 BGR 프레임 가져오기
        captured_bgr_frame = webrtc_ctx.video_processor.last_frame_bgr
        
        if captured_bgr_frame is None:
            st.warning("카메라 프레임이 아직 준비되지 않았습니다. 잠시 기다리거나 카메라를 다시 시작해주세요.")
        else:
            # 1) 베지어 프레임 영역 추출 및 마스킹 (배경 흰색)
            processed_pil_image = extract_and_mask_bezier_region(captured_bgr_frame)

            # 2) 모델 예측
            pred_label, confidence = predict_image(processed_pil_image)
            diagnosis = label_map[pred_label]

            # 3) 결과 표시
            st.subheader("모델 입력 이미지 (베지어 프레임 영역)")
            st.image(processed_pil_image, caption="베지어 프레임 안의 마스킹된 결막 영역", use_container_width=True)
            
            st.subheader("예측 결과")
            st.write(f"**진단 결과: {diagnosis}**")
            st.write(f"**확신도: {confidence:.4f}**")
            st.progress(float(confidence))

            if confidence < 0.65: # 확신도 임계값 조정
                st.info("모델의 확신도가 낮습니다. 조명, 카메라 위치/화질을 다시 확인하고, 베지어 프레임에 눈을 더 정확히 맞춰 재촬영하는 것이 좋습니다.")

else:
    st.info("카메라를 시작해주세요.")