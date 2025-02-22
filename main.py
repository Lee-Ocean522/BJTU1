import cv2
import numpy as np


def are_rectangles_similar(rect1, rect2):
    angle_diff = abs(rect1[2] - rect2[2])
    aspect_ratio1 = rect1[1][0] / rect1[1][1] if rect1[1][1] != 0 else 0
    aspect_ratio2 = rect2[1][0] / rect2[1][1] if rect2[1][1] != 0 else 0
    aspect_ratio_diff = abs(aspect_ratio1 - aspect_ratio2)
    area1 = rect1[1][0] * rect1[1][1]
    area2 = rect2[1][0] * rect2[1][1]
    area_diff = abs(area1 - area2)
    return angle_diff < 1 and aspect_ratio_diff < 0.1 and area_diff < 10


def match_rectangles(contours):
    matched_rectangles = []
    for i, cnt1 in enumerate(contours):
        rect1 = cv2.minAreaRect(cnt1)
        box1 = cv2.boxPoints(rect1)
        box1 = np.int32(box1)
        for j, cnt2 in enumerate(contours):
            if i != j and are_rectangles_similar(rect1, cv2.minAreaRect(cnt2)):
                rect2 = cv2.minAreaRect(cnt2)
                box2 = cv2.boxPoints(rect2)
                box2 = np.int32(box2)
                matched_rectangles.append((box1, box2))
    return matched_rectangles

image = cv2.imread("D:/3.png")
processed_img = image.copy()

img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
img_h, img_s, img_v = cv2.split(img_hsv)

mask_h = cv2.inRange(img_h, 0, 180)
mask_s = cv2.inRange(img_s, 80, 200)
mask_v = cv2.inRange(img_v, 210, 255)
mask = cv2.bitwise_and(cv2.bitwise_and(mask_h, mask_s), mask_v)

img_out = cv2.bitwise_and(image, image, mask=mask)

edges = cv2.Canny(img_out, 50, 150, apertureSize=3)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_width, max_width = 1, 5
min_height, max_height = 2, 4

filtered_contours = []
for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    width, height = rect[1]
    if (min_width <= width <= max_width) and (min_height <= height <= max_height):
        filtered_contours.append(cnt)

for cnt in filtered_contours:
    mask_cnt = np.zeros(img_h.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_cnt, [cnt], 0, 255, -1)

    mean_h = cv2.mean(img_h, mask=mask_cnt)[0]

    if 90 <= mean_h <= 130:  #blue
        color = (255, 0, 0)
        color_name = "Blue"
    else:  #red
        color = (0, 0, 255)
        color_name = "Red"

    rect = cv2.minAreaRect(cnt)
    center = (int(rect[0][0]), int(rect[0][1]))
    left = tuple(cnt[cnt[:, :, 0].argmin()][0])
    right = tuple(cnt[cnt[:, :, 0].argmax()][0])

    cv2.circle(processed_img, center, 5, color, -1)  # 中心点
    cv2.circle(processed_img, left, 3, color, -1)  # 左端点
    cv2.circle(processed_img, right, 3, color, -1)  # 右端点

    cv2.putText(processed_img, color_name,
                (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

matched_pairs = match_rectangles(filtered_contours)

for box1, box2 in matched_pairs:
    for i in range(4):
        cv2.line(processed_img,
                 tuple(box1[i]),
                 tuple(box2[i]),
                 (0, 255, 0), 2)

cv2.imshow('Detection Result', processed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()