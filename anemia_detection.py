import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

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
    checkpoint_path = "best_fold1.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------
# 4. 전처리 정의
# -----------------------
IMG_SIZE = 224
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------
# 5. Bézier 프레임 좌표 계산 (크기를 줄임)
# -----------------------
def get_conjunctiva_bezier_bbox(image_size):
    w, h = image_size
    left = int(w * 0.35)    # 35% 위치로 옮김
    upper = int(h * 0.50)   # 50% 위치로 옮김
    right = int(w * 0.65)   # 65% 위치로 옮김
    lower = int(h * 0.75)   # 75% 위치로 옮김
    return left, upper, right, lower

# -----------------------
# 6. Bézier 곡선 점 계산 헬퍼
# -----------------------
def cubic_bezier_points(p0, p1, p2, p3, n=200):
    """
    제어점 p0, p1, p2, p3에 대해 n개의 점을 계산하여 반환합니다.
    p0~p3는 (x, y) 튜플.
    """
    t = np.linspace(0, 1, n)
    points = []
    for ti in t:
        x = (1-ti)**3 * p0[0] + 3*(1-ti)**2 * ti * p1[0] + 3*(1-ti)*ti**2 * p2[0] + ti**3 * p3[0]
        y = (1-ti)**3 * p0[1] + 3*(1-ti)**2 * ti * p1[1] + 3*(1-ti)*ti**2 * p2[1] + ti**3 * p3[1]
        points.append((x, y))
    return points

# -----------------------
# 7. Bézier 프레임 오버레이 함수
# -----------------------
def draw_bezier_frame(cv_img):
    """
    OpenCV BGR 이미지에 Bézier 프레임을 녹색 선으로 그려서 반환합니다.
    (크기가 줄어든 새로운 bounding box에 맞춰 그립니다)
    """
    h, w, _ = cv_img.shape
    left, upper, right, lower = get_conjunctiva_bezier_bbox((w, h))
    frame_w = right - left
    frame_h = lower - upper

    # 윗곡선 제어점
    p0 = (left, upper)
    p1 = (left + frame_w * 0.25, upper + frame_h * 0.4)
    p2 = (right - frame_w * 0.25, upper + frame_h * 0.4)
    p3 = (right, upper)
    top_curve = cubic_bezier_points(p0, p1, p2, p3, n=200)

    # 아랫곡선 제어점
    p0_b = (right, upper)
    p1_b = (right + frame_w * 0.1, lower + frame_h * 0.05)
    p2_b = (left - frame_w * 0.1, lower + frame_h * 0.05)
    p3_b = (left, upper)
    bottom_curve = cubic_bezier_points(p0_b, p1_b, p2_b, p3_b, n=200)

    pts = np.array(top_curve + bottom_curve, dtype=np.int32)
    cv2.polylines(cv_img, [pts], isClosed=True, color=(0,255,0), thickness=2)
    return cv_img

# -----------------------
# 8. Bézier 내부만 크롭 후 예측 (외부 흰색)
# -----------------------
def crop_conjunctiva_bezier(image: Image.Image) -> Image.Image:
    """
    PIL 이미지에서 Bézier 프레임 내부를 잘라서 반환합니다.
    외부 영역은 흰색으로 채워집니다.
    (새 bounding box 비율을 그대로 사용)
    """
    np_img = np.array(image)
    h, w, _ = np_img.shape
    left, upper, right, lower = get_conjunctiva_bezier_bbox((w, h))
    frame_w = right - left
    frame_h = lower - upper

    # Bézier 경로 계산
    p0 = (left, upper)
    p1 = (left + frame_w * 0.25, upper + frame_h * 0.4)
    p2 = (right - frame_w * 0.25, upper + frame_h * 0.4)
    p3 = (right, upper)
    top_curve = cubic_bezier_points(p0, p1, p2, p3, n=200)

    p0_b = (right, upper)
    p1_b = (right + frame_w * 0.1, lower + frame_h * 0.05)
    p2_b = (left - frame_w * 0.1, lower + frame_h * 0.05)
    p3_b = (left, upper)
    bottom_curve = cubic_bezier_points(p0_b, p1_b, p2_b, p3_b, n=200)

    polygon = top_curve + bottom_curve

    # 전체 이미지 배경을 흰색으로 설정
    white_bg = np.ones_like(np_img) * 255

    # Bézier 내부만 원본, 외부 흰색
    mask = np.zeros((h, w), dtype=np.uint8)
    polygon_np = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [polygon_np], 255)
    inside = cv2.bitwise_and(np_img, np_img, mask=mask)
    combined = np.where(mask[..., None] == 255, inside, white_bg)

    # bounding box 영역만 크롭
    cropped_np = combined[upper:lower, left:right]
    return Image.fromarray(cropped_np)

def predict_image(image: Image.Image):
    """
    Bézier 내부만 크롭하고 모델 예측을 수행합니다.
    """
    try:
        cropped = crop_conjunctiva_bezier(image)
        if cropped.size[0] == 0 or cropped.size[1] == 0:
            raise ValueError("Cropped size zero")
    except Exception as e:
        cropped = image
        st.warning(f"크롭 실패, 전체 이미지로 예측합니다: {e}")

    x = val_transform(cropped).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
        conf, pred = torch.max(probs, dim=0)
    return pred.item(), conf.item(), cropped

# -----------------------
# 9. 세션 상태 초기화
# -----------------------
if "run" not in st.session_state:
    st.session_state.run = False
if "captured_frame" not in st.session_state:
    st.session_state.captured_frame = None

st.title("빈혈 여부 예측 앱")

# -----------------------
# 10. UI 레이아웃: 버튼 및 카메라/결과 표시 영역
# -----------------------
col1, col2 = st.columns([2, 1])
with col1:
    # 카메라 스트리밍 프레임 디스플레이
    FRAME_WINDOW = st.empty()
with col2:
    # 버튼 영역
    start_btn = st.button("카메라 시작")
    capture_btn = st.button("촬영 및 예측")

# -----------------------
# 11. 카메라 스트리밍 및 Bézier 오버레이
# -----------------------
cap = cv2.VideoCapture(0)

if start_btn:
    st.session_state.run = True
if capture_btn:
    st.session_state.run = False

while st.session_state.run:
    ret, frame = cap.read()
    if not ret:
        st.warning("카메라를 열 수 없습니다.")
        st.session_state.run = False
        break

    # Bézier 오버레이
    overlay_frame = draw_bezier_frame(frame.copy())

    # 마지막 프레임을 저장 (PIL 포맷)
    img_rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
    st.session_state.captured_frame = Image.fromarray(img_rgb)

    # 화면에 출력
    FRAME_WINDOW.image(img_rgb, channels="RGB")

    # 약간의 딜레이
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# -----------------------
# 12. 촬영 및 예측
# -----------------------
if st.session_state.captured_frame is not None and not st.session_state.run:
    col1, col2 = st.columns([2, 1])
    with col1:
        image = st.session_state.captured_frame
        st.image(image, caption="촬영된 사진 (프레임 오버레이 포함)", use_container_width=True)
    with col2:
        pred, conf, cropped_img = predict_image(image)
        st.image(cropped_img, caption="크롭된 결막 부분", use_container_width=True)
        st.write(f"**예측 결과:** {label_map[pred]}")
        st.write(f"**신뢰도:** {conf:.4f}")
