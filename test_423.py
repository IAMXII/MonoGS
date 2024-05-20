from PIL import Image

def read_depth_image(depth_image_path):
    # 打开深度图
    depth_image = Image.open(depth_image_path)

    # 获取深度图的像素数据
    depth_data = list(depth_image.getdata())
    print(max(depth_data))
    # 将深度数据写入文本文件
    with open("depth_values.txt", "w") as f:
        # 逐行写入深度值
        for depth_value in depth_data:
            f.write(str(depth_value) + "\n")

# 深度图路径
depth_image_path = "./datasets/tum/rgbd_dataset_freiburg3_long_office_household/depth/1341847980.723020.png"

# 读取深度图并写入文本文件
read_depth_image(depth_image_path)