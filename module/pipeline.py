from multiprocessing import Process, Queue

from configs import config

from video_to_frame import VideoToFrame
from object_detection import ObjectDetection
from license_recognition import LicenseRecognition
from person_recognition import PersonRecognition
from frame_to_video import FrameToVideo

#---------------------------------------------------------------------------
# Create a traffic monitoring pipeline
#---------------------------------------------------------------------------
if __name__ == '__main__':
    frame_queue = Queue()
    car_queue = Queue()
    person_queue = Queue()
    car_frame_queue = Queue()
    person_frame_queue = Queue()
    
    video_to_frame = VideoToFrame(config, frame_queue)
    object_detection = ObjectDetection(config, frame_queue, car_queue, person_queue)
    license_recognition = LicenseRecognition(config, car_queue, car_frame_queue)
    person_recognition = PersonRecognition(config, person_queue, person_frame_queue)
    frame_to_video = FrameToVideo(config, car_frame_queue, person_frame_queue)
    
    video_to_frame.start()
    object_detection.start()
    license_recognition.start()
    person_recognition.start()
    frame_to_video.start()
    
    try:
        video_to_frame.join()
        object_detection.join()
        license_recognition.join()
        person_recognition.join()
        frame_to_video.join()
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt")
        video_to_frame.terminate()
        object_detection.terminate()
        license_recognition.terminate()
        person_recognition.terminate()
        frame_to_video.terminate()

    print("[main] Pipeline end!")

