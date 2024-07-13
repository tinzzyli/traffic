from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators.stra_attack import MOTEvaluator

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

import argparse
import os
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path

import time
from PIL import Image
import numpy as np
from pprint import pprint

# model_path = "bytetrack_s_mot17.pth.tar"

exp = get_exp(exp_file=None, exp_name="yolox-s")

confthre=exp.test_conf
nmsthre=exp.nmsthre
num_classes=exp.num_classes

print(f"confthre = {confthre}"
      f"\nnmsthre = {nmsthre}"
      f"\nnum_classes = {num_classes}")

# model = exp.get_model()
# model = model.cuda()
# model = fuse_model(model)

# checkpoint = torch.load(model_path, map_location="cuda")
# model.load_state_dict(checkpoint["model"])
# model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "bytetrack_s_mot17.pth"
model = torch.load(model_path, map_location=device)
model = model.eval()
decoder = model.head.decode_outputs

image_processor_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/yolo-tiny/yolos-tiny_image_processor.pth"
image_processor = torch.load(image_processor_path, map_location=device)

channels = 3
height = 720
width = 1080

# image_path = f"../image/640_480.jpg"
# image_path = f"attack_data/000001.jpg"
image_path = f"original_data/000001.jpg"
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
        # # Method 1:
        images = [image_array for _ in range(batch_size)]
        inputs = image_processor(images=images, return_tensors="pt").to(device)
        
        # 多次推理取平均
        for test_num in range(test_num_per_bs):
            start_time = time.time()

            # inputs = image_processor(images=images, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(inputs['pixel_values']) # ['pixel_values'])

            # # model predicts bounding boxes and corresponding COCO classes
            # logits = outputs.logits
            # bboxes = outputs.pred_boxes

            # print results
            # target_sizes = torch.tensor([image.size[::-1] for _ in range(batch_size)])
            # print(outputs)
            
            if decoder is not None:
                outputs = decoder(outputs, dtype=outputs.type())
            count = (outputs[:,:,5]* outputs[:,:,4] > 0.3).sum()
            outputs = postprocess(outputs, num_classes, confthre, nmsthre)[0]
            
            print(count)
            if outputs is not None:
                print(outputs)

            # results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)
            
            # for i, result in enumerate(results):
            #     print(f"Result {i}:")
            #     for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            #         box = [round(i, 2) for i in box.tolist()]
            #         print(
            #             f"Detected {model.config.id2label[label.item()]} with confidence "
            #             f"{round(score.item(), 3)} at location {box}"
            #         )
            
            end_time = time.time()
        
            inference_times[batch_size].append(end_time - start_time)
        
        inference_times_avg[batch_size] = round(sum(inference_times[batch_size]) / len(inference_times[batch_size]), 4)
        print(f"batch size = {batch_size}, inference time = {inference_times_avg[batch_size]}")
    
    pprint(inference_times_avg)
        

print(f"\n=================== CALCULATE ===================")
calculate([1], 1)

# batch_sizes = [batch_size for batch_size in range(1, 33)]

# test_num_per_bs = 100

# # warm up
# print(f"=================== WARM UP ===================")
# calculate([bs for bs in range(1, 11)], 10)


# print(f"\n=================== CALCULATE ===================")
# calculate(batch_sizes, test_num_per_bs)





