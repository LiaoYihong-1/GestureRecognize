import os
import shutil

# 指定包含子目录的主目录
main_directory = './xiang'

# 遍历主目录下的每个子目录
for subdir in os.listdir(main_directory):
    subdir_path = os.path.join(main_directory, subdir)

    # 检查子目录是否是目录而不是文件
    if os.path.isdir(subdir_path):
        # 获取子目录下的所有图片文件名
        images = [file for file in os.listdir(subdir_path) if file.endswith('.jpg')]  # 假设图片都以.jpg结尾

        # 对图片文件名进行排序
        images.sort(key=lambda x: int(x.split('.')[0]))  # 根据文件名中的数字进行排序

        # 保留每个子目录中的前20张图片，删除其余的图片
        size = len(images)
        group = int(size/100)
        for j in range(group):
            for i in range(20, 100):
                image_path = os.path.join(subdir_path, images[i+100*j])
                os.remove(image_path)  # 删除不保留的图片