import cv2
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
from time import time


def sort_by_number(filename):
    return int(''.join(filter(str.isdigit, filename)))


# 点云数据
pointcloud_path = '../dataset/cp/pcds'
pointcloud = sorted(os.listdir(pointcloud_path), key=sort_by_number)
# print(pointcloud)
# 相机内参
fx = 332.232689  # x 轴方向的焦距
fy = 332.644823  # y 轴方向的焦距
cx = 333.058485  # x 轴方向的光心坐标
cy = 240.998586  # y 轴方向的光心坐标
big_num = 1e10
# 构建相机内参矩阵
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# 构建投影矩阵（相机内参 + 雷达到相机的转换矩阵）
# 这里假设雷达和相机之间的转换矩阵为单位矩阵
P = np.array([[-0.0264814146660713, -0.999649386885150, 0.00173906523035761, 0.0659530032036776],
              [0.0121395165905959, -0.00206233611926795, -0.999925722855674, 0.108374858163385],
              [0.999576628994339, -0.0264570578759927, 0.0121885681396033, 0.0699178702997177]])

if __name__ == "__main__":
    for i, pcd in tqdm(enumerate(pointcloud)):
        start_time = time()
        pcd = o3d.io.read_point_cloud(os.path.join(pointcloud_path, pcd))
        points = np.asarray(pcd.points)
        R = P[:, :3]
        T = P[:, 3]
        # print(points.shape[0])
        # 将点云投影到深度图
        count=0
        count1 = 0
        depth_image = np.zeros((480, 640), dtype=np.float32)  # 初始化深度图
        for point in points:
            # 将点云转换为齐次坐标
            point_homogeneous = np.array([point[0], point[1], point[2], 1])
            # 计算点云在相机坐标系下的投影
            projected_point = np.dot(P, point_homogeneous)
            projected_point = np.dot(K, projected_point)
            # print(projected_point)
            # 归一化得到像素坐标
            pixel_coords = projected_point[:2] / projected_point[2]
            # 将像素坐标转换为整数
            u, v = int(pixel_coords[0]), int(pixel_coords[1])
            # 如果像素坐标在图像范围内，则更新深度图
            if 0 <= u < 640 and 0 <= v < 480:
                if depth_image[v, u]:
                    count+=1
                    depth_image[v, u] = min(depth_image[v, u], projected_point[2])
                    # depth_image[v, u] = projected_point[2]
                # print(v, u, depth_image[v, u])
                else:
                    count1+=1
                    depth_image[v, u] = projected_point[2]
        # print("count:",count)
        # print("count1:",count1)
        cv2.imwrite("../dataset/cp/depths_ambiguity/depth_{}.png".format(i), depth_image)
        end_time = time()
        # print('time_{}'.format(i), end_time - start_time)
# # 可视化深度图
# cv2.imshow("Depth Image", depth_image / np.max(depth_image))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
