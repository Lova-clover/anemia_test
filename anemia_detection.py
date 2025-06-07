import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
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
    # 1) 기본 ResNet18 불러오기
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for name, param in model.named_parameters():
        if not (name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False

    # 2) fc 레이어 재정의
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )

    # 3) 체크포인트 로드
    checkpoint_path = "best_fold3.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)

    # 4) DataParallel 저장된 경우 'module.' prefix 제거
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "", 1)
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    # 5) strict=False 옵션으로 로드 (키 불일치 무시)
    load_result = model.load_state_dict(state_dict, strict=False)
    st.write("⚙️ Load model result:", load_result)

    # 6) 디바이스 배치 및 eval 모드
    model = model.to(device)
    model.eval()
    return model

# ------------------------------------------------------
# 3.1. 모델을 전역으로 한 번만 로드해 두기
# ------------------------------------------------------
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
    frame_width_ratio = 0.15
    frame_height_ratio = 0.05
    center_x_ratio = 0.5
    center_y_ratio = 0.55

    left = int(w * (center_x_ratio - frame_width_ratio / 2))
    upper = int(h * (center_y_ratio - frame_height_ratio / 2))
    right = int(w * (center_x_ratio + frame_width_ratio / 2))
    lower = int(h * (center_y_ratio + frame_height_ratio / 2))
    return left, upper, right, lower

# -----------------------
# 6. Bézier 곡선 점 계산 헬퍼
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
    frame_w, frame_h = max(1, right-left), max(1, lower-upper)

    p0, p1, p2, p3 = (left, upper), (left+int(frame_w*0.25), upper+int(frame_h*0.4)), \
                     (right-int(frame_w*0.25), upper+int(frame_h*0.4)), (right, upper)
    top_curve = cubic_bezier_points(p0, p1, p2, p3, n=200)

    p0b, p1b, p2b, p3b = (right, upper), (right+int(frame_w*0.1), lower+int(frame_h*0.05)), \
                        (left-int(frame_w*0.1), lower+int(frame_h*0.05)), (left, upper)
    bottom_curve = cubic_bezier_points(p0b, p1b, p2b, p3b, n=200)

    pts = np.array(top_curve + bottom_curve, dtype=np.int32)
    cv2.polylines(cv_img, [pts], isClosed=True, color=(0,255,0), thickness=2)
    return cv_img

# -----------------------
# 8. Bézier 프레임 내부만 추출 및 마스킹
# -----------------------
def extract_and_mask_bezier_region(img_bgr: np.ndarray) -> Image.Image:
    h, w, _ = img_bgr.shape
    left, upper, right, lower = get_conjunctiva_bezier_bbox((w, h))
    frame_w, frame_h = max(1, right-left), max(1, lower-upper)

    top_curve = cubic_bezier_points((left, upper),
                                    (left+int(frame_w*0.25), upper+int(frame_h*0.4)),
                                    (right-int(frame_w*0.25), upper+int(frame_h*0.4)),
                                    (right, upper), n=200)
    bottom_curve = cubic_bezier_points((right, upper),
                                       (right+int(frame_w*0.1), lower+int(frame_h*0.05)),
                                       (left-int(frame_w*0.1), lower+int(frame_h*0.05)),
                                       (left, upper), n=200)
    polygon = np.array(top_curve + bottom_curve, dtype=np.int32)

    white_bg = np.ones_like(img_bgr)*255
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    combined = np.where(mask[...,None]==255, img_bgr, white_bg)
    crop = combined[upper:lower, left:right]

    if crop.size == 0:
        st.warning("경고: 프레임 영역이 유효하지 않습니다. 흰 배경으로 대체합니다.")
        return Image.fromarray(np.ones((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8)*255)

    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

# -----------------------
# 9. 모델 예측 함수
# -----------------------
def predict_image(pil_img: Image.Image):
    global model
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
        self.last_frame_bgr = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.last_frame_bgr = img.copy()
        return draw_bezier_frame(img)

# -----------------------
# 11. Streamlit 페이지 구성
# -----------------------
st.title("결막 사진으로 빈혈 예측 앱")
st.markdown("""
**본 앱은 프레임 안의 결막 부분을 분석하여 빈혈 여부를 예측합니다.**

1. 'Start camera' 버튼 클릭  
2. 초록색 프레임 안에 눈 결막을 맞춤  
3. '사진 찍기 & 예측' 버튼 클릭  
""")

webrtc_ctx = webrtc_streamer(
    key="capture",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
    video_processor_factory=ConjunctivaProcessor,
    media_stream_constraints={"video":{"width":{"ideal":1920},"height":{"ideal":1080},"frameRate":{"ideal":30}},"audio":False},
    async_processing=True,
)

if webrtc_ctx.video_processor:
    if st.button("사진 찍기 & 예측"):
        frame = webrtc_ctx.video_processor.last_frame_bgr
        if frame is None:
            st.warning("카메라 프레임 준비 중입니다...")
        else:
            pil = extract_and_mask_bezier_region(frame)
            label, conf = predict_image(pil)
            st.image(pil, caption="모델 입력 이미지", use_container_width=True)
            st.subheader(f"진단: {label_map[label]} (확신도 {conf:.4f})")
            st.progress(float(conf))
            if conf < 0.65:
                st.info("확신도가 낮습니다. 조명과 프레임 위치를 다시 확인해주세요.")
else:
    st.info("카메라를 시작해주세요.")
