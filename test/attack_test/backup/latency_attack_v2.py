# https://huggingface.co/hustvl/yolos-tiny

from transformers import YolosImageProcessor, YolosForObjectDetection
from torchvision import transforms
from PIL import Image
import torch

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

# 打印模型结构
print(model)

# 打印模型损失函数
print(model.loss)

channels = 3
height = 720
width = 1080

image_path = f"../image/640_480.jpg"
image = Image.open(image_path)
# (height, width, channels) -> (channels, height, width)
# image_array = np.array(image.resize((width, height)), dtype=np.float32).transpose(2, 0, 1) # !!! dtype 不对
image_array = np.array(image.resize((width, height)), dtype=np.float32).transpose(2, 0, 1) # !!! dtype 不对

# print(f"image_array = {image_array}")

print(f"image_array.shape = {image_array.shape}")

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
        inputs = image_processor(images=images, return_tensors="pt").to(device)
        
        # 多次推理取平均
        for test_num in range(test_num_per_bs + 10):
            start_time = time.time()

            # inputs = image_processor(images=images, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(inputs['pixel_values']) # ['pixel_values'])

            # # model predicts bounding boxes and corresponding COCO classes
            # logits = outputs.logits
            # bboxes = outputs.pred_boxes

            # print results
            # target_sizes = torch.tensor([image.size[::-1] for image in images])
            target_sizes = torch.tensor([image.size[::-1] for _ in range(batch_size)])
            # target_sizes = torch.tensor([(height, width) for _ in range(batch_size)])

            results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)
            
            # for i, result in enumerate(results):
            #     print(f"Result {i}:")
            #     for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            #         box = [round(i, 2) for i in box.tolist()]
            #         print(
            #             f"Detected {model.config.id2label[label.item()]} with confidence "
            #             f"{round(score.item(), 3)} at location {box}"
            #         )
            
            end_time = time.time()
            
            if test_num < 10:
                continue
            inference_times[batch_size].append(end_time - start_time)
        
        inference_times_avg[batch_size] = round(sum(inference_times[batch_size]) / len(inference_times[batch_size]), 4)
        print(f"batch size = {batch_size}, inference time = {inference_times_avg[batch_size]}")
    
    pprint(inference_times_avg)
        

print(f"\n=================== CALCULATE ===================")
calculate([1], 100)

# batch_sizes = [batch_size for batch_size in range(1, 33)]

# test_num_per_bs = 100

# # warm up
# print(f"=================== WARM UP ===================")
# calculate([bs for bs in range(1, 11)], 10)


# print(f"\n=================== CALCULATE ===================")
# calculate(batch_sizes, test_num_per_bs)

