import os
from copy import copy
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms
model = torch.hub.load('.', 'custom', path_or_model ='runs/train/exp3/weights/best.pt', source='local', force_reload=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def euler_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def quaternion_to_euler(q0, q1, q2, q3):
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q0 * q2 - q3 * q1)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def angle_between(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

previous_angle = 0
def detect_color(hsv_frame):#没用
    color = '0'
    # 创建一个白色区域的掩模
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 255, 255])
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)

    # 红色的HSV范围
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 蓝色的HSV范围
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # 获得红色和蓝色掩模
    mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # 计算每个掩膜的亮度
    brightness_red = np.sum(mask_red)
    brightness_blue = np.sum(mask_blue)

    # 选择亮度更高的颜色
    if brightness_red > brightness_blue:
        mask = mask_red
        color = 'red'
    else:
        mask = mask_blue
        color = 'blue'



    # 对选定的掩膜应用轮廓检测
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []  # 用于存储轮廓的中心点
    sharp_points = []  # 用于存储轮廓的锐角点

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 5:  # 只处理五边形
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))

                min_angle = 180  # 初始化最小角度
                best_point = None  # 初始化最佳点

                for i in range(len(approx)):
                    p1 = approx[i][0]
                    p2 = approx[(i + 1) % len(approx)][0]
                    p3 = approx[(i + 2) % len(approx)][0]

                    angle = angle_between(p1, p2, p3)

                    if angle < min_angle and angle < 80:
                        min_angle = angle
                        best_point = p2

                if best_point is not None:
                    sharp_points.append(best_point)
    return contours, centers, sharp_points, color

# 假设已知的四元数
q0, q1, q2, q3 = 0.5, 0.5, 0, 0  # 这里需要填入实际的四元数值

# 转换四元数到欧拉角
roll, pitch, yaw = quaternion_to_euler(q0, q1, q2, q3)

# 计算旋转矩阵
R = euler_to_rotation_matrix(roll, pitch, yaw)
input_video_path = 'some_path'
cap = cv2.VideoCapture(input_video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

output_video_path = f'some_path'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLOv5 model for detection
    results = model(frame)
    detected_frame = results.render()[0]

    # Get YOLO detected bounding boxes
    detected_boxes = results.pred[0][:, :4].cpu().numpy()

    distances_between_sharp_points = []

    # Process detected boxes
    for box in detected_boxes:
        x1, y1, x2, y2 = map(int, box)
        roi = detected_frame[y1:y2, x1:x2]

        # Detect color and get contours, centers and sharp points
        contours, centers, sharp_points, _ = detect_color(roi)

        # Calculate distances between sharp points and store
        for i in range(len(sharp_points)):
            for j in range(i+1, len(sharp_points)):
                distance = np.linalg.norm(np.array(sharp_points[i])-np.array(sharp_points[j]))
                distances_between_sharp_points.append(distance)

        # Draw detected centers in YOLO bounding boxes
        for center in centers:
            cv2.circle(detected_frame, (x1 + center[0], y1 + center[1]), 3, (0, 255, 0), -1)

    # Compute average distance between sharp points
    if distances_between_sharp_points:
        avg_distance = np.mean(distances_between_sharp_points)
        scale_factor = 1.3 / avg_distance  # Pixel scale

        # Only calculate if 'centers' list is not empty
        if centers:
            cX, cY = centers[0]

            target_pixel_width = avg_distance

            # Calculate real-world distance based on scale factor
            global_coordinates_pixel = np.dot(R, np.array([cX, cY, 1]))
            global_coordinates_real = global_coordinates_pixel * scale_factor

            # Print global coordinates
            print("Global Coordinates (in m):", global_coordinates_real)

    out.write(detected_frame)
    cv2.imshow('Processed Frame', detected_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()