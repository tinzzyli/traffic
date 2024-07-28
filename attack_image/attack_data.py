from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import cv2
learning_rate = 0.02 #0.07
epochs = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/yolo-tiny/yolos-tiny_model.pth"
image_processor_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/yolo-tiny/yolos-tiny_image_processor.pth"
yolo_model = torch.load(yolo_model_path, map_location=device)
yolo_image_processor = torch.load(image_processor_path, map_location=device)

save_dir = f"data_after_attacking/"

from PIL import Image
import logging

def create_logger(module, filename, level):
    # Create a formatter for the logger, setting the format of the log with time, level, and message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a logger named 'logger_{module}'
    logger = logging.getLogger(f'logger_{module}')
    logger.setLevel(level)     # Set the log level for the logger
    
    # Create a file handler, setting the file to write the logs, level, and formatter
    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(level)         # Set the log level for the file handler
    fh.setFormatter(formatter) # Set the formatter for the file handler
    
    # Add the file handler to the logger
    logger.addHandler(fh)
    
    return logger

logger = create_logger("attack_data", "attack_data.log", logging.INFO)

def generate_mask(outputs,result, x_shape, y_shape):

    mask_x = 4
    mask_y = 2
    # mask = torch.ones(mask_y,mask_x)  # 初始mask为3*3
    mask = torch.ones(y_shape,x_shape)
    # print(outputs.shape) # lifang535 remove
    # outputs = outputs.unsqueeze(0)
    # outputs = postprocess(outputs, num_classes=1, conf_thre=0.1, nms_thre=0.4)[0]
    
    boxes = result["boxes"] # lifang535
    
    # print(f"========== outputs ==========") # lifang535
    # print(outputs)
    
    # pred = non_max_suppression(
    #                 outputs[0], conf_thres, iou_thres, classes, agnostic_nms)
    x_len = int(x_shape / mask_x)
    y_len = int(y_shape / mask_y)
    if boxes is not None:
        for i in range(len(boxes)):
            detection = boxes[i]
            center_x, center_y = (detection[0]+detection[2])/2, (detection[1]+detection[3])/2
            # 根据检测框的中心点位置，判断它在哪个区域
            region_x = int(center_x / x_len)
            region_y = int(center_y / y_len)
            
            # print(f"========== center_x ==========") # lifang535
            # print(center_x)
            # print(f"========== center_y ==========") # lifang535
            # print(center_y)
            # time.sleep(20)
            
            mask[region_y*y_len:(region_y+1)*y_len, region_x*x_len:(region_x+1)*y_len] -= 0.05
    
    
    return mask

def run_attack(outputs,result,bx, strategy, max_tracker_num, mask):

    per_num_b = (25*45)/max_tracker_num
    per_num_m = (50*90)/max_tracker_num
    per_num_s = (100*180)/max_tracker_num

    # scores = outputs[:,5] * outputs[:,4] # lifang535 remove
    # print(f"========== scores: {scores.size()} ==========") # lifang535
    # print(scores)
    # # time.sleep(20)
    
    scores = result["scores"] # lifang535
    # print(f"========== scores: {scores.size()} ==========") # lifang535
    # print(scores)
    # time.sleep(20)
    
    # 修改 scores 的 grad_fn 为 grad_fn=<SelectBackward0>
    
    # print(f"========== scores: {scores.size()} ==========") # lifang535
    # print(scores)
    # time.sleep(20)

    loss2 = 40*torch.norm(bx, p=2)
    targets = torch.ones_like(scores)
    loss3 = F.mse_loss(scores, targets, reduction='sum')
    loss = loss3#+loss2
    
    # print(f"========== loss ==========") # lifang535
    # print(loss)
    
    # print(f"========== 1 bx.grad ==========") # lifang535
    # print(bx.grad)
    
    loss.requires_grad_(True)
    loss.backward(retain_graph=True)
    
    # adam_opt.step()
    
    # print(f"========== 2 bx.grad ==========") # lifang535
    # print(bx.grad)
    
    # 为什么 loss.backward 之后的 bx.grad 是 None？
    
    # time.sleep(20)
    
    bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    bx.data = -3.5 * mask * bx.grad+ bx.data
    count = (scores > 0.9).sum()
    print('loss',loss.item(),'loss_2',loss2.item(),'loss_3',loss3.item(),'count:',count.item())
    return bx

def attack(
    frame_list,
    half=False,
):
    """
    COCO average precision (AP) Evaluation. Iterate inference on the test dataset
    and the results are evaluated by COCO API.

    NOTE: This function will change training mode to False, please save states if needed.

    Args:
        model : model to evaluate.

    Returns:
        ap50_95 (float) : COCO AP of IoU=50:95
        ap50 (float) : COCO AP of IoU=50
        summary (sr): summary info of evaluation.
    """
    # TODO half to amp_test
    tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        
    frame_id = 0
    total_l1 = 0
    total_l2 = 0
    strategy = 0
    max_tracker_num = int(15)
    rgb_means=torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).to(device)
    std=torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).to(device)
    
    def process_img(imgs):
        nonlocal frame_id, total_l1, total_l2, strategy, max_tracker_num, rgb_means, std

        frame_id += 1
        bx = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
        bx = bx.astype(np.float32)
        bx = torch.from_numpy(bx).to(device).unsqueeze(0)
        bx = bx.data.requires_grad_(True)
        # imgs = imgs.type(tensor_type)
        imgs = imgs.to(device)
        
        for iter in tqdm(range(epochs)):
            # print(f"========== iter: {iter} ==========") # lifang535
            
            added_imgs = imgs+bx
            
            # print(f"========== imgs.shape: {imgs.shape} ==========") # lifang535
            # print(f"========== bx.shape: {bx.shape} ==========") # lifang535
            # print(f"========== rgb_means.shape: {rgb_means.shape} ==========") # lifang535
            # print(f"========== added_imgs.shape: {added_imgs.shape} ==========") # lifang535
            # print(f"========== std.shape: {std.shape} ==========") # lifang535
            # time.sleep(100)
            
            l2_norm = torch.sqrt(torch.mean(bx ** 2))
            l1_norm = torch.norm(bx, p=1)/(bx.shape[3]*bx.shape[2])
            # added_imgs.clamp_(min=0, max=1)
            # input_imgs = (added_imgs - rgb_means)/std
            
            # if half:
            #     input_imgs = input_imgs.half()
            
            outputs = None
            
            # print(f"========== added_imgs.shape: {added_imgs.shape} ==========")
            # print(added_imgs)
            # input_imgs = torch.clamp(added_imgs*255,0,255).to(device)
            # print(f"========== added_imgs.shape: {added_imgs.shape} ==========")
            # print(added_imgs)
            # time.sleep(100)
            # input_imgs = yolo_image_processor(images=added_imgs, return_tensors="pt").to(device)
            # with torch.no_grad():
            yolo_outputs = yolo_model(added_imgs)

            target_sizes = [imgs.shape[2:] for _ in range(1)]
            process_yolo_outputs = yolo_image_processor.post_process_object_detection(yolo_outputs, threshold=0.0, target_sizes=target_sizes)
            result = process_yolo_outputs[0]
            
            if iter == 0:
                mask = generate_mask(outputs,result,added_imgs.shape[3],added_imgs.shape[2]).to(device) # lifng535: 掩码只改变一次
            bx = run_attack(outputs,result,bx, strategy, max_tracker_num, mask)

        if strategy == max_tracker_num-1:
            strategy = 0
        else:
            strategy += 1
        print(added_imgs.shape)
        added_blob = torch.clamp(added_imgs*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        added_blob = added_blob[..., ::-1]
        # print(f"========== added_blob.shape: {added_blob.shape} ==========")
        # print(added_blob)
        print(f"max: {added_blob.max()}, min: {added_blob.min()}")
        added_blob = np.round(added_blob).astype(np.uint8)
        # print(f"========== added_blob.shape: {added_blob.shape} ==========")
        # print(added_blob)
        
        # added_blob = (added_imgs[0] * 255).permute(1, 2, 0).detach().cpu().numpy()
        # print(f"========== added_blob.shape: {added_blob.shape} ==========")
        # time.sleep(100)
        
        # # print(f"========== added_imgs: {added_imgs.shape} ==========")
        # # print(added_imgs)
        # added_imgs_np = (added_imgs * 255).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # # added_imgs_np = (added_imgs).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # print(f"========== added_imgs_np.shape: {added_imgs_np.shape} ==========")
        # print(added_imgs_np)
        # added_imgs_np = cv2.cvtColor(added_imgs_np, cv2.COLOR_RGB2BGR)
        # added_blob = added_imgs_np # .astype(np.uint8)
        # print(f"========== added_blob.shape: {added_blob.shape} ==========")
        # print(added_blob)
        
        # cv2.imwrite(f"{save_dir}/{0:06d}.jpg", added_imgs_np)
        # from PIL import Image
        # test_img = Image.open(f"{save_dir}/{0:06d}.jpg")
        # test_img = np.array(test_img).transpose(2, 0, 1)
        # test_img = yolo_image_processor(images=test_img, return_tensors="pt").to(device)
        
        # yolo_outputs = yolo_model(**test_img)
        # process_yolo_outputs = yolo_image_processor.post_process_object_detection(yolo_outputs, threshold=0.9, target_sizes=target_sizes)
        # result = process_yolo_outputs[0]
        # for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        #     box = [round(i, 2) for i in box.tolist()]
        #     print(
        #         f"Detected {yolo_model.config.id2label[label.item()]} with confidence "
        #         f"{round(score.item(), 3)} at location {box}"
        #     )
        # time.sleep(100)
        
        # added_blob = (added_imgs * 255).to(torch.uint8)[0].permute(1, 2, 0).detach().cpu().numpy()
        
        
        # temp_imgs = (added_imgs * 255).to(torch.uint8)
        
        # added_imgs_int = temp_imgs.float() / 255.0 # TODO：需要转化成 float 再 / 255.0
        # yolo_outputs_int = yolo_model(added_imgs_int)
        # process_yolo_outputs_int = yolo_image_processor.post_process_object_detection(yolo_outputs_int, threshold=0.9, target_sizes=target_sizes)
        # result_int = process_yolo_outputs_int[0]
        # print(f"len(result_int['labels']): {len(result_int['labels'])}")

        # temp_inputs = yolo_image_processor(images=temp_imgs, return_tensors="pt").to(device)
        # yolo_outputs = yolo_model(**temp_inputs)
        # process_yolo_outputs = yolo_image_processor.post_process_object_detection(yolo_outputs, threshold=0.9, target_sizes=target_sizes)
        # result = process_yolo_outputs[0]
        # print(f"len(result['labels']): {len(result['labels'])}")
        
        # temp_imgs = temp_imgs.detach().cpu().numpy()
        # temp_imgs = temp_imgs[0]
        # added_blob = temp_imgs
        
        # # temp_imgs = cv2.cvtColor(temp_imgs, cv2.COLOR_RGB2BGR)
        # temp_inputs = yolo_image_processor(images=temp_imgs, return_tensors="pt").to(device)
        # yolo_outputs = yolo_model(**temp_inputs)
        # process_yolo_outputs = yolo_image_processor.post_process_object_detection(yolo_outputs, threshold=0.9, target_sizes=target_sizes)
        # result = process_yolo_outputs[0]
        
        # print(f"========== temp_imgs: {temp_imgs.shape} ==========")
        # print(temp_imgs)
        
        # print(f"len(result['labels']): {len(result['labels'])}")
        # time.sleep(100)
        
        # frame_id = 0 # TODO
        save_path = f"{save_dir}/{frame_id:06d}.jpg"
        print(f"========== save_path: {save_path} ==========")
        cv2.imwrite(save_path, added_blob)
        
        # Image.fromarray(added_blob).save(save_path)
        
        test_img = Image.open(save_path)
        test_img = np.array(test_img).transpose(2, 0, 1)
        test_img = (torch.from_numpy(test_img).unsqueeze(0).float() / 255.0).to(device)
        # test_img = yolo_image_processor(images=test_img, return_tensors="pt").to(device)
        yolo_outputs_read = yolo_model(test_img)
        
        # test_img = cv2.imread(save_path)
        # test_img = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        # test_img = torch.tensor(test_img).permute(2, 0, 1).float() / 255.0
        # test_img = test_img.unsqueeze(0).to(device)
        # print(f"========== test_img: {test_img.shape} ==========")
        # yolo_outputs_read = yolo_model(test_img)
        
        process_yolo_outputs_read = yolo_image_processor.post_process_object_detection(yolo_outputs_read, threshold=0.9, target_sizes=target_sizes)
        result_read = process_yolo_outputs_read[0]
        
        yolo_outputs_original = yolo_model(imgs)
        process_yolo_outputs_original = yolo_image_processor.post_process_object_detection(yolo_outputs_original, threshold=0.9, target_sizes=target_sizes)
        result_original = process_yolo_outputs_original[0]
        
        yolo_outputs_float = yolo_model(added_imgs)
        process_yolo_outputs_float = yolo_image_processor.post_process_object_detection(yolo_outputs_float, threshold=0.9, target_sizes=target_sizes)
        result_float = process_yolo_outputs_float[0]
        
        added_imgs_int = torch.round((added_imgs * 255)).to(torch.uint8) # .float() / 255.0
        # print(f"========== added_imgs_int.shape: {added_imgs_int.shape} ==========")
        # print(added_imgs_int)
        added_imgs_int = (added_imgs_int.float() / 255.0)
        yolo_outputs_int = yolo_model(added_imgs_int)
        process_yolo_outputs_int = yolo_image_processor.post_process_object_detection(yolo_outputs_int, threshold=0.9, target_sizes=target_sizes)
        result_int = process_yolo_outputs_int[0]
        
        print(f"[Frame {frame_id}] The number of detected objects: {len(result_original['labels'])} → {len(result_read['labels'])} (if float: {len(result_float['labels'])}, if int: {len(result_int['labels'])})")
        logger.info(f"[Frame {frame_id}] The number of detected objects: {len(result_original['labels'])} → {len(result_read['labels'])} (if float: {len(result_float['labels'])}, if int: {len(result_int['labels'])})")
        # for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        #     box = [round(i, 2) for i in box.tolist()]
        #     print(
        #         f"Detected {yolo_model.config.id2label[label.item()]} with confidence "
        #         f"{round(score.item(), 3)} at location {box}"
        #     )
        
        print(l1_norm.item(),l2_norm.item())
        total_l1 += l1_norm
        total_l2 += l2_norm
        mean_l1 = total_l1/frame_id
        mean_l2 = total_l2/frame_id
        print(mean_l1.item(),mean_l2.item())
        
        del bx
        del outputs
        del imgs
        
        return mean_l1, mean_l2
        
    for frame_id, frame in enumerate(frame_list):
        mean_l1, mean_l2 = process_img(frame)

    return mean_l1, mean_l2


if __name__ == "__main__":
    frame_list = []
    
    input_video_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/attack_image/0.mp4"
    cap = cv2.VideoCapture(input_video_path)
    
    video_fps = int(cap.get(5))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video_fps = {video_fps}, total_frames = {total_frames}")
    
    while True:
        ret, frame = cap.read() # frame: (height, width, channel), BGR
        if not ret:
            break
        # frame.resize((224, 224, 3))
        frame_array = np.array(frame) # , dtype=np.float32) # .transpose(2, 0, 1)
        # print(f"frame_array.shape = {frame_array.shape}")
        
        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        
        frame_tensor = torch.tensor(frame_array).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)
        
        # print(f"frame_tensor.shape = {frame_tensor.shape}")
        
        frame_list.append(frame_tensor)
        
    cap.release()
    
    attack(frame_list=frame_list)
    

