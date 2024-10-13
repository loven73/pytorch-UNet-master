import numpy as np
import csv
from PIL import Image

# 路径设置
image_path = 'my_dataset/image/image1.png'  # 指向B超图像
coords_file_path = 'my_dataset/4351-31-横起1.csv'  # 坐标文件路径
mask_image_path = 'my_dataset/masks/image1.png'  # 保存掩码图像的路径

# 读取B超图像
image = Image.open(image_path)
width, height = image.size

# 打开CSV文件并读取坐标
tumor_coords = []
with open(coords_file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        try:
            # 读取 X, Y 坐标
            x = int(row[0])
            y = int(row[1])
            # 选择性需要 R, G, B 值
            # r = int(row[2])
            # g = int(row[3])
            # b = int(row[4])
            tumor_coords.append((x, y))
        except (ValueError, IndexError) as e:
            print(f"无效的坐标行: {row} - 错误: {e}")  # 输出无效行的警告

# 创建掩码图像，初始化为全黑（0）
mask = Image.new('L', (width, height), 0)

# 设置肿瘤区域
for x, y in tumor_coords:
    if 0 <= x < width and 0 <= y < height:  # 确保坐标在图像范围内
        mask.putpixel((x, y), 255)  # 将对应像素设置为白色（255）

# 保存掩码图像
mask.save(mask_image_path)
