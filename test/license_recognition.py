import os
import torch
import easyocr

import numpy as np

from PIL import Image
from pprint import pprint

image_path = "image/640_339.jpg"

image = Image.open(image_path)

image_array = np.array(image)

device = torch.device("cuda:2")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用第 0 号 GPU

model_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/easyocr/easyocr_model.pth"

# reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
# reader.device = device
# torch.save(reader, model_path)

model = torch.load(model_path) # , map_location=device)
model.device = device
# model = model.to(device)

with torch.no_grad():
    result = model.readtext(image_array)

pprint(result)
print(f"len(result) = {len(result)}")

# 寻找置信度最高的结果
best_result = None
best_confidence = 0
for r in result:
    if r[2] > best_confidence:
        best_result = r
        best_confidence = r[2]
        
print(f"best_result = {best_result}")
print(f"best_confidence = {best_confidence}")
