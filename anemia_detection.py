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
# 3. ResNet18 모델 정의 및 가중치 로드
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
# 5. Haar Cascade 로드 (눈 영역 검출용)
# -----------------------
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# -----------------------
# 6. Bézier frame 좌표 계산 (휴대폰 기준으로 작게 크기를 조절하였다)
# -----------------------
def get_conjunctiva_bezier_bbox(image_size):
    # --- 이 부분을 조절하여 모바일 환경에 맞게 프레임 크기를 조정하였다. ---
    # frame_width_ratio: 프레임의 가로 폭 비율 (전체 너비 대비)
    # frame_height_ratio: 프레임의 세로 폭 비율 (전체 높이 대비)
    # center_x_ratio: 프레임 중앙의 가로 위치 (전체 너비 대비)
    # center_y_ratio: 프레임 중앙의 세로 위치 (전체 높이 대비)
    
    w, h = image_size
    
    frame_width_ratio = 0.10   
    frame_height_ratio = 0.03  
    
    center_x_ratio = 0.5
    center_y_ratio = 0.55

    left = int(w * (center_x_ratio - frame_width_ratio / 2))
    right = int(w * (center_x_ratio + frame_width_ratio / 2))
    upper = int(h * (center_y_ratio - frame_height_ratio / 2))
    lower = int(h * (center_y_ratio + frame_height_ratio / 2))

    return left, upper, right, lower

# -----------------------
# 7. Bézier 곡선 점 계산 헬퍼 (결막 부분 정확히 crop하기 위함)
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
# 8. Bézier 프레임 오버레이
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
# 9. Bézier 프레임 내부만 추출하고 외부를 흰색으로 마스킹
# -----------------------
def extract_and_mask_bezier_region(img_bgr: np.ndarray) -> np.ndarray:
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
    combined = np.where(mask[..., None] == 255, img_bgr, white_bg)

    # bounding box 기준으로 크롭
    cropped = combined[upper:lower, left:right]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        # 너무 작으면 흰색 반환
        return np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

    return cropped

# -----------------------
# 10. 눈 영역(ROI) 검출 후 홍채(iris) 검출
# -----------------------
def detect_eye_region(gray_img):
    """
    Haar Cascade로 눈 패치(ROI)를 검출.
    성공 시 (x, y, w, h) 반환, 없으면 None.
    """
    eyes = eye_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
    if len(eyes) == 0:
        return None
    # 가장 넓은 눈 영역 선택
    eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)
    ex, ey, ew, eh = eyes[0]
    return (ex, ey, ew, eh)

def detect_iris_circle(gray_img, eye_rect=None):
    """
    HoughCircles로 홍채(iris) 검출.
    eye_rect가 주어지면 그 영역 내부에서만 탐색.
    성공 시 iris_mask(원 내부=255), (x, y, r) 반환.
    실패 시 None, None 반환.
    """
    # ROI 설정: eye_rect가 있으면 그 부분만 사용
    if eye_rect is not None:
        ex, ey, ew, eh = eye_rect
        roi_gray = gray_img[ey:ey+eh, ex:ex+ew]
        offset = (ex, ey)
    else:
        roi_gray = gray_img
        offset = (0, 0)

    # 1) CLAHE 적용하여 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_clahe = clahe.apply(roi_gray)

    # 2) GaussianBlur로 노이즈 제거 (더 부드러운 경계 확보)
    blurred = cv2.GaussianBlur(roi_clahe, (9, 9), 2)

    rows = roi_gray.shape[0]

    # 3) HoughCircles 파라미터 조정
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,                            # 누격 비율: 1.2로 약간 높여 작은 원도 감지
        minDist=rows / 4,                 # 같은 원이 여러 번 감지되지 않도록 거리 제한
        param1=50,                        # Canny 에지의 upper threshold
        param2=20,                        # accumulator threshold: 낮출수록 더 많은 원 감지
        minRadius=int(rows * 0.10),       # 최소 동공 크기 (ROI 높이의 10%)
        maxRadius=int(rows * 0.25)        # 최대 동공 크기 (ROI 높이의 25%)
    )
    if circles is None:
        return None, None

    circles = np.round(circles[0, :]).astype(int)
    # 동공(검은원)은 반지름이 작은 쪽에 가깝기 때문에 반지름 작은 순서로 정렬
    circles = sorted(circles, key=lambda c: c[2])
    cx, cy, r = circles[0]

    # 4) ROI 좌표를 전체 이미지 좌표로 보정
    cx_full = cx + offset[0]
    cy_full = cy + offset[1]

    # 5) 마스크 생성
    iris_mask = np.zeros_like(gray_img, dtype=np.uint8)
    cv2.circle(iris_mask, (cx_full, cy_full), r, 255, thickness=-1)

    return iris_mask, (cx_full, cy_full, r)

def generate_sclera_mask_by_rule(shape, iris_center, iris_radius):
    """
    홍채 중심에서 1.25*r 아래, 반지름 0.35*r 원을 공막(sclera) 영역으로 생성.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if iris_center is None or iris_radius is None:
        return mask, (0, 0, 0)
    x, y = iris_center
    sclera_cx = int(round(x))
    sclera_cy = int(round(y + 1.25 * iris_radius))
    sclera_r = int(round(0.35 * iris_radius))
    cv2.circle(mask, (sclera_cx, sclera_cy), sclera_r, 255, thickness=-1)
    return mask, (sclera_cx, sclera_cy, sclera_r)

# -----------------------
# 11. 공막 밝기 계산 함수
# -----------------------
def compute_sclera_brightness(gray_img, sclera_mask):
    """
    공막(sclera) 마스크 영역 평균 그레이스케일 밝기 반환.
    픽셀이 없으면 1.0 리턴.
    """
    pixels = gray_img[sclera_mask == 255]
    return float(np.mean(pixels)) if pixels.size > 0 else 1.0

# -----------------------
# 12. 실시간 스트리밍 처리를 위한 Transformer
# -----------------------
class ConjunctivaProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_frame_bgr = None

    def transform(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        overlaid = draw_bezier_frame(img_bgr.copy())
        self.last_frame_bgr = img_bgr.copy()
        return overlaid

# -----------------------
# 13. Streamlit 페이지 구성
# -----------------------
st.title("📸 결막 사진으로 빈혈 예측 앱")

st.markdown(
    """
    **이 앱은 Bézier 프레임 안의 결막을 잘라낸 뒤, 눈으로 홍채를 찾고  
    공막 밝기를 계산하여 밝기 정규화 후 ResNet18 모델로 빈혈 여부를 예측합니다.**

    **사용법:**
    1. 아래 'Start camera' 버튼을 눌러 카메라를 켜세요.
    2. 카메라 피드에 나타나는 **초록색 프레임 안에 결막 부분을 맞춰주세요.**
    3. '사진 찍기 & 예측' 버튼을 눌러 촬영하면, 프레임 안의 영역만 추출하여 예측합니다.
    """
)

# 13.1. webrtc_streamer 설정 (카메라 해상도 최대로 요청)
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

# -----------------------
# 14. “사진 찍기 & 예측” 버튼
# -----------------------
if webrtc_ctx.video_processor:
    if st.button("사진 찍기 & 예측"):
        captured_bgr = webrtc_ctx.video_processor.last_frame_bgr
        if captured_bgr is None:
            st.warning("카메라 프레임이 아직 준비되지 않았습니다. 잠시 기다려주세요.")
        else:
            # (1) Bézier 프레임 영역만 추출(마스킹+크롭) → BGR numpy
            cropped_bgr = extract_and_mask_bezier_region(captured_bgr)

            # (2) 눈 영역(ROI) 검출 → 홍채(Hough Circle) 검출
            gray_full = cv2.cvtColor(captured_bgr, cv2.COLOR_BGR2GRAY)
            eye_rect = detect_eye_region(gray_full)

            iris_mask, iris_info = detect_iris_circle(gray_full, eye_rect)
            # iris_info가 None이면 전체 이미지에서 재시도
            if iris_info is None:
                iris_mask, iris_info = detect_iris_circle(gray_full, None)

            # (3) 공막 마스크 생성 → 공막 밝기 계산
            sclera_mask, sclera_info = generate_sclera_mask_by_rule(
                gray_full.shape,
                iris_info[:2] if iris_info else None,
                iris_info[2] if iris_info else None
            )
            sclera_brightness = compute_sclera_brightness(gray_full, sclera_mask)

            # (4) 밝기 정규화: cropped_bgr을 공막 평균 밝기 기준으로 스케일링
            if iris_info is not None and sclera_brightness > 0:
                target_brightness = 128.0
                alpha = target_brightness / (sclera_brightness + 1e-6)
                cropped_float = cropped_bgr.astype(np.float32) * alpha
                cropped_float = np.clip(cropped_float, 0, 255).astype(np.uint8)
            else:
                cropped_float = cropped_bgr.copy()

            # (5) RGB 순서로 변환하여 PIL.Image 생성
            cropped_rgb = cv2.cvtColor(cropped_float, cv2.COLOR_BGR2RGB)
            pil_input = Image.fromarray(cropped_rgb)

            # (6) ResNet18 예측
            input_tensor = val_transform(pil_input).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_label = int(probs.argmax())
                confidence = float(probs.max().item())
            diagnosis = label_map[pred_label]

            # (7) 시각화를 위해 윤곽선 그리기
            vis = captured_bgr.copy()
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

            # 홍채 원 (빨간색)
            if iris_info:
                ix, iy, ir = iris_info
                cv2.circle(vis_rgb, (ix, iy), ir, (255, 0, 0), 2)

            # 공막 원 (노란색)
            scx, scy, scr = sclera_info
            if (scx, scy, scr) != (0, 0, 0):
                cv2.circle(vis_rgb, (scx, scy), scr, (255, 255, 0), 2)

            # Bézier 결막 윤곽 (녹색)
            h, w, _ = captured_bgr.shape
            left, upper, right, lower = get_conjunctiva_bezier_bbox((w, h))
            polygon_pts = np.array(
                cubic_bezier_points((left, upper),
                                   (left + int((right-left)*0.25), upper + int((lower-upper)*0.4)),
                                   (right - int((right-left)*0.25), upper + int((lower-upper)*0.4)),
                                   (right, upper), n=200)
                + cubic_bezier_points((right, upper),
                                      (right + int((right-left)*0.1), lower + int((lower-upper)*0.05)),
                                      (left - int((right-left)*0.1), lower + int((lower-upper)*0.05)),
                                      (left, upper), n=200),
                dtype=np.int32
            )
            cv2.polylines(vis_rgb, [polygon_pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # (8) 결과 출력
            st.subheader("촬영 직후: 홍채 / 공막 / 결막 ROI 시각화")
            st.image(vis_rgb, use_container_width=True)

            st.subheader("모델 입력용 결막 이미지 (밝기 보정 후)")
            st.image(pil_input, use_container_width=True)

            st.subheader("예측 결과")
            st.write(f"- **진단:** {diagnosis}")
            st.write(f"- **신뢰도:** {confidence:.4f}")
            st.progress(confidence)

            if confidence < 0.65:
                st.info("모델 확신도가 낮습니다. 조명·화질을 개선하고 재촬영해 보세요.")
else:
    st.info("카메라를 켜려면 위 ‘Start camera’ 버튼을 눌러주세요.")
