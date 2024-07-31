import torch
from thop import profile
# from archs.ViT_model import get_vit, ViT_Aes
from torchvision.models import resnet50
import time
import numpy as np

import inspect

import dlib
import pickle
import face_recognition
import face_recognition_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# face_recognition
model_path = "/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/face_recognition/face_encodings.pkl"

# face_recognition.face_encodings 所在文件
print(inspect.getfile(face_recognition.face_encodings))
model = dlib.get_frontal_face_detector()
print(model)

import dlib
from thop import profile
import torch

# # # 加载 dlib 的人脸识别模型
# # dlib_face_recognition_model = dlib.face_recognition_model_v1(
# #     dlib.face_recognition_model_location()
# # )

# cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
# cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)
# print(cnn_face_detection_model)
# print(cnn_face_detector)

# # 定义一个包装器类来兼容 thop 的输入
# class DlibModelWrapper(torch.nn.Module):
#     def __init__(self, dlib_model):
#         super(DlibModelWrapper, self).__init__()
#         self.dlib_model = dlib_model

#     def forward(self, x):
#         # dlib 模型在输入时需要额外处理，这里假设 x 为预处理后的图像
#         return self.dlib_model.compute_face_descriptor(x)
#         # return face_recognition.face_encodings(x)

# # 包装 dlib 模型
# wrapped_model = DlibModelWrapper(cnn_face_detector)

# # 定义一个假数据输入，假设 dlib 模型接受 150x150 灰度图像
# input_tensor = torch.randn(1, 1, 150, 150)

model = resnet50().to(device)

# 使用 thop 计算 FLOPs 和参数数
input1 = torch.randn(1, 3, 100, 100).to(device)
flops, params = profile(model, inputs=(input1, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')


# # easyocr
# model_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/easyocr/easyocr_model.pth"

# model = torch.load(model_path, map_location=device)

# # model.readtext 所在文件
# print(inspect.getfile(model.readtext))

# detector = model.detector
# recognizer = model.recognizer

# # detector.device = device

# # print(recognizer.module)

# # input1 = torch.randn(4, 3, 224, 224).to(device)
# input1 = torch.randn(1, 3, 100, 100).to(device)
# flops, params = profile(detector.module, inputs=(input1, ))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')

# # input2 = torch.randn(4, 3, 480, 640).to(device)
# # flops, params = profile(recognizer.module, inputs=(input2, )) # wrong method
# # print('FLOPs = ' + str(flops/1000**3) + 'G')
# # print('Params = ' + str(params/1000**2) + 'M')



# # yolo-tiny
# model_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/yolo-tiny/yolos-tiny_model.pth"
# image_processor_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/yolo-tiny/yolos-tiny_image_processor.pth"
# model = torch.load(model_path, map_location=device)
# input1 = torch.randn(1, 3, 100, 100).to(device)
# flops, params = profile(model, inputs=(input1, ))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')

# # image
# from PIL import Image
# import numpy as np
# image_path = f"image/640_480.jpg"
# image = Image.open(image_path)
# # (height, width, channels) -> (channels, height, width)
# # image_array = np.array(image.resize((width, height))).transpose(2, 0, 1)
# # image_array = np.array(image, dtype=np.float32).transpose(2, 0, 1)
# image_array = np.array(image, dtype=np.float32).transpose(2, 0, 1) # !!! dtype 不对

# input3 = torch.from_numpy(image_array).unsqueeze(0).to(device)

# flops, params = profile(model, inputs=(input3, ))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')


