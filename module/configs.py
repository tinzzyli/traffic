config = {
    'input_video_dir': '../input_video',
    
    'output_video_dir': '../output_video',
    
    'video_number': 17,
    
    'frame_interval': 1, # ms
    
    'frame_size': (224, 224, 3),
    
    'object_detection': {
        'device': 'cuda:3',
        'yolo-tiny_image_processor_path': '../model/yolo-tiny/yolos-tiny_image_processor.pth',
        'yolo-tiny_model_path': '../model/yolo-tiny/yolos-tiny_model.pth',
    },
    
    'license_recognition': {
        'device': 'cuda:2',
        'easyocr_model_path': '../model/easyocr/easyocr_model.pth',
    },
    
    'person_recognition': {
        'device': 'cpu',
        'face_recognition_model_path': '../model/face_recognition/face_encodings.pkl',
    },
}
