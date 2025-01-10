import numpy as np

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

def compute_real_world_coords(drone_gps_position, roll, pitch, yaw, pixel_coords, pixel_scale, image_width, image_height):

    # A点在相机坐标系中的坐标
    camera_x = (pixel_coords[0] - image_width / 2) * pixel_scale
    camera_y = (pixel_coords[1] - image_height / 2) * pixel_scale
    camera_z = 0  # 假设垂直拍摄，深度方向为0

    camera_coords = np.array([camera_x, camera_y, camera_z])

    # 计算旋转矩阵
    R = euler_to_rotation_matrix(roll, pitch, yaw)

    # 将相机坐标转换为世界坐标
    world_coords = R.dot(camera_coords) + drone_gps_position

    return world_coords

# 调用示例
drone_gps_position = np.array([lat, lon, alt])  # 无人机的GPS位置
roll, pitch, yaw = roll_angle, pitch_angle, yaw_angle  # 无人机的俯仰，翻转及横滚角度
pixel_coords = np.array([u, v])  # A点在照片上的像素坐标
pixel_scale = 0.01  # 每个像素对应的实际距离（米/像素）
image_width, image_height = 1920, 1080 # 图像的宽度和高度（像素）

real_world_coords = compute_real_world_coords(drone_gps_position, roll, pitch, yaw, pixel_coords, pixel_scale, image_width, image_height)

print(f"A点在现实中的坐标: ({real_world_coords[0]}, {real_world_coords[1]}, {real_world_coords[2]})")
