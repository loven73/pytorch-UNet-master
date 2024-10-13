from PIL import Image

# 打开两张图片
image = Image.open('my_dataset/mask1.png')

# 裁剪成相同的大小
# 如果需要裁剪的尺寸不同于原图尺寸，可以定义裁剪框
# 这里假设要裁剪为左上角的100x100区域
cropped_image = image.crop((81, 95, 81+256, 95+256))

# 保存结果
cropped_image.save('docs/result/combined_image2.png')
