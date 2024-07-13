import os
import cv2
import time
import torch

from PIL import Image
from queue import Empty
from threading import Thread
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor

from configs import config
from request import Request

class ObjectDetection(Process):
    def __init__(self, 
                 config: dict,
                 frame_queue: Queue,
                 car_queue: Queue,
                 person_queue: Queue,):
        super().__init__()

        # from video_to_frame module
        self.frame_queue = frame_queue
        # to license_recognition module
        self.car_queue = car_queue
        # to person_recognition module
        self.person_queue = person_queue
        
        self.frame_size = config['frame_size']
        
        self.device = torch.device(config['object_detection']['device'])
        self.image_processor_path = config['object_detection']['yolo-tiny_image_processor_path']
        self.model_path = config['object_detection']['yolo-tiny_model_path']
        
        self.image_processor = None
        self.model = None
        self.id2label = None
        
        # self.thread_pool = ThreadPoolExecutor(max_workers=1000)
        
        self.end_flag = False

    def run(self):
        print(f"[ObjectDetection] Start!")
        
        self.image_processor = torch.load(self.image_processor_path, map_location=self.device)
        self.model = torch.load(self.model_path, map_location=self.device)
        self.id2label = self.model.config.id2label
        
        while not self.end_flag:
            try:
                request = self.frame_queue.get(timeout=1)
            except Empty:
                continue
            
            if request is None:
                self._end()
                break
            
            # temp_thread = self.thread_pool.submit(self._infer, request)
            
            self._infer(request)
            
            if request.frame_id % 50 == 0:
                print(f"[ObjectDetection] video_id: {request.video_id}, frame_id: {request.frame_id}")
    
    def _infer(self, request):
        frame_array = request.data
        
        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        
        # inputs = self.image_processor(images=[frame_array], return_tensors="pt").to(self.device)
        inputs = {
            'pixel_values': (torch.from_numpy(frame_array.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0).to(self.device)
        }
        
        with torch.no_grad():
            # outputs = self.model(**inputs)
            outputs = self.model(inputs['pixel_values'])

        results = self.image_processor.post_process_object_detection(outputs, threshold=0.9)
        
        if request.frame_id % 10 == 0:
            print(f"[ObjectDetection] frame_array.shape = {frame_array.shape}, inputs['pixel_values'].shape = {inputs['pixel_values'].shape}")
            print(f"[ObjectDetection] video_id: {request.video_id}, frame_id: {request.frame_id}, len(results[0]['labels']) = {len(results[0]['labels'])}")
        
        # print(f"[ObjectDetection] Inference time: {round(time.time() - start_time, 4)}")
        
        for i, result in enumerate(results):
            car_number = sum([1 for label in result["labels"] if self.id2label[label.item()] == 'car'])
            person_number = sum([1 for label in result["labels"] if self.id2label[label.item()] == 'person'])
            
            # print(f"[ObjectDetection] car_number = {car_number}, person_number = {person_number}")
            
            request.car_number = car_number
            request.person_number = person_number
            
            self.car_queue.put(request)
            self.person_queue.put(request)
            
            car_id = 0
            person_id = 0
            
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                box = [round(i, 5) for i in box.tolist()]
                if self.id2label[label.item()] == 'car':
                    req = request.copy()
                    req.car_id = car_id
                    
                    req.box = box
                    req.label = self.id2label[label.item()]
                    self.car_queue.put(req)
                    
                    car_id += 1
                elif self.id2label[label.item()] == 'person':
                    req = request.copy()
                    req.person_id = person_id
                    
                    req.box = box
                    req.label = self.id2label[label.item()]
                    self.person_queue.put(req)
                    
                    person_id += 1
                    
                # print(
                #     f"Detected {self.id2label[label.item()]} with confidence "
                #     f"{round(score.item(), 3)} at location {box}"
                # )
            
    def _end(self):
        self.car_queue.put(None)
        self.person_queue.put(None)
        
        self.end_flag = True
        print(f"[ObjectDetection] Stop!")

        
if __name__ == '__main__':
    frame_queue = Queue()
    object_detection = ObjectDetection(config, frame_queue)
    object_detection.start()
    try:
        object_detection.join()
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt")
        object_detection.terminate()
        