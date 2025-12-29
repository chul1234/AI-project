import cv2
import dlib
import numpy as np
from math import atan2, degrees

# 얼굴 디텍터와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 웹캠에서 비디오 캡처
cap = cv2.VideoCapture(0)

# 필터 이미지 로드 함수
def load_filter_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Image file {filepath} not found.")
        exit(1)
    return image

# 필터 이미지 로드
filters = {
    1: {
        'left_ear': load_filter_image('./samples/left_ear_1.png'),
        'right_ear': load_filter_image('./samples/right_ear_1.png'),
        'nose': load_filter_image('./samples/nose_1.png')
    },
    2: {
        'left_ear': load_filter_image('./samples/left_ear_2.png'),
        'right_ear': load_filter_image('./samples/right_ear_2.png'),
        'nose': load_filter_image('./samples/nose_2.png')
    }
}

def ensure_alpha_channel(image):
    if image.shape[2] == 3:
        b, g, r = cv2.split(image)
        a = np.ones_like(b) * 255
        image = cv2.merge([b, g, r, a])
    else:
        b, g, r, a = cv2.split(image)
        black_mask = (b == 0) & (g == 0) & (r == 0)
        a[black_mask] = 0
        image = cv2.merge([b, g, r, a])
    return image

# 필터 이미지에 알파 채널 추가 및 수정
for filter_set in filters.values():
    filter_set['left_ear'] = ensure_alpha_channel(filter_set['left_ear'])
    filter_set['right_ear'] = ensure_alpha_channel(filter_set['right_ear'])
    filter_set['nose'] = ensure_alpha_channel(filter_set['nose'])

# 코 이미지 색상 반전 함수
def invert_colors(image):
    if image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        b = cv2.bitwise_not(b)
        g = cv2.bitwise_not(g)
        r = cv2.bitwise_not(r)
        image = cv2.merge([b, g, r, a])
    else:
        image = cv2.bitwise_not(image)
    return image

# 코 이미지 색상 반전
for filter_set in filters.values():
    filter_set['nose'] = invert_colors(filter_set['nose'])

# 각도 계산 함수
def angle_value(p1, p2):
    xdiff = p2[0] - p1[0]
    ydiff = p2[1] - p1[1]
    return degrees(atan2(ydiff, xdiff))

# 투명 이미지 오버레이 함수
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None, angle=0):
    bg_img = background_img.copy()
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    if angle != 0:
        (h, w) = img_to_overlay_t.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_to_overlay_t = cv2.warpAffine(img_to_overlay_t, M, (w, h), flags=cv2.INTER_LINEAR)

    if img_to_overlay_t.shape[2] == 4:
        b, g, r, a = cv2.split(img_to_overlay_t)
        mask = cv2.medianBlur(a, 5)
    else:
        b, g, r = cv2.split(img_to_overlay_t)
        mask = np.ones_like(b, dtype=np.uint8) * 255

    h, w, _ = img_to_overlay_t.shape
    roi_x1 = max(x - w // 2, 0)
    roi_y1 = max(y - h // 2, 0)
    roi_x2 = min(x + w // 2, bg_img.shape[1])
    roi_y2 = min(y + h // 2, bg_img.shape[0])

    roi = bg_img[roi_y1:roi_y2, roi_x1:roi_x2]

    overlay_h, overlay_w = roi.shape[:2]
    if overlay_h == 0 or overlay_w == 0:
        return bg_img

    img_to_overlay_t = img_to_overlay_t[:overlay_h, :overlay_w]
    mask = mask[:overlay_h, :overlay_w]

    if img_to_overlay_t.shape[:2] != roi.shape[:2]:
        img_to_overlay_t = cv2.resize(img_to_overlay_t, (roi.shape[1], roi.shape[0]))
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.add(img1_bg, img2_fg)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img

# 사용자 입력 받기
filter_choice = int(input("Enter 1, 2, or 3: "))

# 스케일러 설정
scaler = 1.0

# 메인 루프
while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    ori = img.copy()

    faces = detector(img, 1)

    if len(faces) == 0:
        print('no faces!')
        continue

    for idx, face in enumerate(faces):
        dlib_shape = predictor(img, face)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        for s in shape_2d:
            cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # 코 끝 점 (30번 점)
        nose_center = (int(shape_2d[30][0]), int(shape_2d[30][1]))

        # 회전 각도 계산
        angle = -angle_value(shape_2d[5], shape_2d[11])

        # 얼굴 크기 계산
        min_coords = np.min(shape_2d, axis=0)
        max_coords = np.max(shape_2d, axis=0)
        face_size = max(max_coords - min_coords)
        mean_face_size = int(face_size * 1.2)

        # 귀 필터 크기 조정 (얼굴 크기에 맞추기)
        if filter_choice == 2:
            ear_size = (int(mean_face_size * 0.5), int(mean_face_size * 0.5))
            nose_size = (int(mean_face_size * 0.5), int(mean_face_size * 0.5))
        else:
            ear_size = (int(mean_face_size * 0.4), int(mean_face_size * 0.4))
            nose_size = (int(mean_face_size * 1.5), int(mean_face_size * 1.0))

        # 필터 선택
        if filter_choice == 1:
            selected_filter = filters[1]
        elif filter_choice == 2:
            selected_filter = filters[2]
        elif filter_choice == 3:
            selected_filter = filters[1] if idx % 2 == 0 else filters[2]

        # 왼쪽 귀 오버레이 (왼쪽 눈 위쪽)
        left_eye_top = (int(shape_2d[1][0]), int(shape_2d[1][1] - ear_size[1]))
        ori = overlay_transparent(ori, selected_filter['left_ear'], left_eye_top[0], left_eye_top[1], overlay_size=ear_size)

        # 오른쪽 귀 오버레이 (오른쪽 눈 위쪽)
        right_eye_top = (int(shape_2d[15][0]), int(shape_2d[15][1] - ear_size[1]))
        ori = overlay_transparent(ori, selected_filter['right_ear'], right_eye_top[0], right_eye_top[1], overlay_size=ear_size)

        # 코 오버레이 
        ori = overlay_transparent(ori, selected_filter['nose'], nose_center[0], nose_center[1], overlay_size=nose_size, angle=angle)

    cv2.imshow('original', img)
    cv2.imshow('result', ori)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
