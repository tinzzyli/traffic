import cv2
import os

input_image_dir = "/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/input_image/before_attacking"
output_image_dir = "/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/output_image/before_attacking"

# 确保输出目录存在，如不存在则创建
os.makedirs(output_image_dir, exist_ok=True)

# 获取输入目录中所有图片文件名并排序
image_files = sorted([f for f in os.listdir(input_image_dir) if f.endswith('.jpg')])

# 遍历所有图片并处理
for image_name in image_files:
    image_path = os.path.join(input_image_dir, image_name)

    # 读取图片
    img = cv2.imread(image_path)

    if img is not None:
        # 显示关于图片的信息
        print(f"图片 {image_name}：宽度={img.shape[1]}, 高度={img.shape[0]}")

        # 在输出目录中保存图片
        output_path = os.path.join(output_image_dir, image_name)
        cv2.imwrite(output_path, img)
    else:
        print(f"错误：无法读取 {image_name}")

print("图片处理和保存完成。")
