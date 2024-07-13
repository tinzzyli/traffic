import os
from PIL import Image

# 定义图片目录和目标大小
data_folder = 'data_after_attacking'
target_size = (1088, 608)

# 获取文件夹中所有文件，按文件名排序
files = sorted(os.listdir(data_folder))

# 计数器，记录已处理的图片数量
count = 0

# 遍历文件
for file in files:
    # 检查文件名是否符合格式，并且计数器小于400
    if file.endswith('.jpg') and file[:-4].isdigit() and int(file[:-4]) <= 400:
        file_path = os.path.join(data_folder, file)
        
        # 打开和修改图片大小
        with Image.open(file_path) as img:
            # resized_img = img.resize(target_size)
            # resized_img.save(file_path)
            print(f"{file_path}: shape = {img.size}")

        count += 1

        # 如果处理了400张图片，停止遍历
        if count >= 400:
            break

print(f"共处理了 {count} 张图片。")
