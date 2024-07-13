import cv2
import os

# 定义图片目录和视频输出路径
# data_folder = 'data'
# output_file = '0.mp4'
data_folder = 'data_after_attacking'
output_file = '3.mp4'
frame_size = (1088, 608)
fps = 30  # 每秒帧数

# 获取文件夹中所有文件，按文件名排序
files = sorted(os.listdir(data_folder))

# 过滤符合条件的文件，并限制数量为400
images = [file for file in files if file.endswith('.jpg') and file[:-4].isdigit() and int(file[:-4]) <= 400]
images = images[:400]

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

# 遍历图片并写入视频
for image in images:
    img_path = os.path.join(data_folder, image)
    img = cv2.imread(img_path)
    if img is not None:
        # resized_img = cv2.resize(img, frame_size)
        video_writer.write(img)

# 释放视频写入对象
video_writer.release()
print("视频已保存到", output_file)
