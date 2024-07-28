# https://huggingface.co/hustvl/yolos-tiny

from transformers import YolosImageProcessor, YolosForObjectDetection
from torchvision import transforms
from PIL import Image
import torch
import cv2

import time
from pprint import pprint
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_device = "cpu"

# model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
# image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

model_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/yolo-tiny/yolos-tiny_model.pth"
image_processor_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/yolo-tiny/yolos-tiny_image_processor.pth"

# torch.save(model, model_path)
# torch.save(image_processor, image_processor_path)

model = torch.load(model_path, map_location=device)
image_processor = torch.load(image_processor_path, map_location=device)

channels = 3
height = 720
width = 1080

frame_id = 2
image_path = f"data_after_attacking/{frame_id:06d}.jpg"
# image_path = f"data/{frame_id:06d}.jpg" # 4 objects
image = Image.open(image_path)

# image_cv2 = cv2.imread(image_path)
# image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

# (height, width, channels) -> (channels, height, width)
# image_array = np.array(image.resize((width, height))).transpose(2, 0, 1)
# image_array = np.array(image, dtype=np.float32).transpose(2, 0, 1)
# image_array = np.array(image.resize((width, height)), dtype=np.float32).transpose(2, 0, 1) # !!! dtype 不对
image_array = np.array(image).transpose(2, 0, 1) # !!! dtype 不对

print(f"image_array = {image_array}")

# # bytes from array
# image_bytes = image_array.tobytes()
# # print(image_bytes)
# # array from bytes
# image_array = np.frombuffer(image_bytes, dtype=np.uint8)
# # 将一维数组重塑为多维数组
# image_array = image_array.reshape((channels, width, height))
# print(image_array)
print(f"image_array.shape = {image_array.shape}")



# RuntimeError: Input type (unsigned char) and bias type (float) should be the same
# image_array = image_array.astype(np.float32) # / 255.0
# print(f"image_array.shape = {image_array.shape}")


# 定义转换
transform = transforms.Compose([
    transforms.Resize((height, width)),  # 调整大小为所需的尺寸
    transforms.ToTensor()  # 转换为张量
    # 正则化 [-2, 2]
])

def calculate(batch_sizes, test_num_per_bs):
    inference_times = {
        batch_size: [] for batch_size in batch_sizes
    }

    inference_times_avg = {
        batch_size: 0 for batch_size in batch_sizes
    }
    
    for batch_size in batch_sizes:
        # Method 1:
        images = [image_array for _ in range(batch_size)]
        # inputs = image_processor(images=images, return_tensors="pt").to(device)
        inputs = {
            'pixel_values': (torch.from_numpy(image_array).unsqueeze(0).float() / 255.0).to(device)
        }
        # print(inputs) # 包含正则化 [-2, 2]
        
        # # Method 2:
        # inputs = torch.randn(batch_size, 3, 224, 224).to(device)  # Example input tensor
        # # print(f"inputs.shape = {inputs.shape}")

        # # Method 3 (TODO: bytes):
        # image_tensor = torch.from_numpy(image_array)
        # print(f"image_tensor.shape = {image_tensor.shape}")
        
        # tensor_list = []
        # for _ in range(batch_size):
        #     tensor_list.append(image_tensor)
        # # 将张量列表堆叠为一个张量
        # inputs = torch.stack(tensor_list, dim=0).to(device)
        # # print(f"inputs.shape = {inputs.shape}")
        
        # 多次推理取平均
        for test_num in range(test_num_per_bs):
            start_time = time.time()
            
            # inputs = image_processor(images=images, return_tensors="pt").to(device) # 可以自己写预处理

            with torch.no_grad():
                outputs = model(inputs['pixel_values']) # ['pixel_values'])

            # # model predicts bounding boxes and corresponding COCO classes
            # logits = outputs.logits
            # bboxes = outputs.pred_boxes

            # print results
            # target_sizes = torch.tensor([image.size[::-1] for image in images])
            target_sizes = torch.tensor([image.size[::-1] for _ in range(batch_size)])
            print(f"image.size = {image.size}, target_sizes = {target_sizes}")
            # target_sizes = torch.tensor([(height, width) for _ in range(batch_size)])

            results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)
            
            for i, result in enumerate(results):
                for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    print(
                        f"Detected {model.config.id2label[label.item()]} with confidence "
                        f"{round(score.item(), 3)} at location {box}"
                    )
                print(f"Result {i}: {len(result['labels'])} objects detected")
            
            end_time = time.time()
            
            inference_times[batch_size].append(end_time - start_time)
        
        inference_times_avg[batch_size] = round(sum(inference_times[batch_size]) / len(inference_times[batch_size]), 4)
        print(f"batch size = {batch_size}, inference time = {inference_times_avg[batch_size]}")
    
    pprint(inference_times_avg)
        

print(f"\n=================== CALCULATE ===================")
calculate([1], 1)



# batch_sizes = [batch_size for batch_size in range(1, 33)]

# test_num_per_bs = 100

# warm up
# print(f"=================== WARM UP ===================")
# calculate([bs for bs in range(1, 11)], 10)


# print(f"\n=================== CALCULATE ===================")
# calculate(batch_sizes, test_num_per_bs)

