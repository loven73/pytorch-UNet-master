from PIL import Image
import os

if __name__ == '__main__':
    bash_path = './dataset_new'
    folder = os.listdir(bash_path)
    for i in folder:
        file_list = os.listdir(bash_path+'/'+i)
        for j in file_list:
            # 假设你的图像路径为image_path
            image_path = bash_path+"/"+i+"/"+j # 替换为你的图像路径
            output_path =  f"./dataset_new_new/{i}/{j}" # 替换为你想保存的路径和文件名
            # 打开图像并转换为 RGB 格式
            img = Image.open(image_path).convert("RGB")

            # 保存转换后的图像
            img.save(output_path)

            print(f"图像已保存为: {output_path}")
    print("全部完成")
