import os
import torch
import pickle
import face_recognition

import numpy as np


face_image_dir = "face_image"
test_image_dir = "person_image"

face_image_files = os.listdir(face_image_dir)
face_image_files.sort(key=lambda x: int(x.split('.')[0]))
print(f"face_image_files: {face_image_files}")

test_image_files = os.listdir(test_image_dir)
test_image_files.sort(key=lambda x: int(x.split('.')[0]))
print(f"test_image_files: {test_image_files}")

# names = [
#     "Putin",
#     "Biden",
#     "Obama",
#     "Trump",
#     "Jack Ma",
#     "Jackie Chan",
#     "Stephen Chow",
#     "Musk"
# ]

# # 搜索人脸库中的所有人脸
# known_face_encodings = []
# known_face_names = []
# for face_image_file in face_image_files:
#     face_image_path = os.path.join(face_image_dir, face_image_file)
#     face_image = face_recognition.load_image_file(face_image_path)
#     face_encoding = face_recognition.face_encodings(face_image)[0]
#     known_face_encodings.append(face_encoding)
#     known_face_names.append(names[int(face_image_file.split('.')[0])])
    
# # 搜索测试图片中的所有人脸
# for test_image_file in test_image_files:
#     test_image_path = os.path.join(test_image_dir, test_image_file)
#     test_image = face_recognition.load_image_file(test_image_path)
#     test_face_encodings = face_recognition.face_encodings(test_image)
    
#     for test_face_encoding in test_face_encodings:
#         results = face_recognition.compare_faces(known_face_encodings, test_face_encoding)
#         # 输出结果，如果有匹配的人脸，则输出对应的人名，否则输出 None
#         for i, result in enumerate(results):
#             if result:
#                 print(f"{test_image_file} is {known_face_names[i]}")
#                 break
#         else:
#             print(f"{test_image_file} is None")
                
# 保存人脸编码
model_path = "/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/face_recognition/face_encodings.pkl"
# with open(model_path, "wb") as f:
#     pickle.dump(known_face_encodings, f)
#     pickle.dump(known_face_names, f)
    
# 读取人脸编码
with open(model_path, "rb") as f:
    known_face_encodings = pickle.load(f)
    known_face_names = pickle.load(f)
    
# 读取测试图片
for test_image_file in test_image_files:
    test_image_path = os.path.join(test_image_dir, test_image_file)
    test_image = face_recognition.load_image_file(test_image_path)
    # print(f"test_image = {test_image}")
    test_image = np.array(test_image)
    print(f"test_image.shape = {test_image.shape}")
    
    with torch.no_grad():
        test_face_locations = face_recognition.face_locations(test_image)
        test_face_encodings = face_recognition.face_encodings(test_image, test_face_locations)
        
    # print(f"len(test_face_encodings) = {len(test_face_encodings)}")
    
    for test_face_encoding in test_face_encodings:
        results = face_recognition.compare_faces(known_face_encodings, test_face_encoding)
        # 输出结果，如果有匹配的人脸，则输出对应的人名，否则输出 None
        for i, result in enumerate(results):
            if result:
                print(f"{test_image_file} is {known_face_names[i]}")
                break
        else:
            print(f"{test_image_file} is None")
            


# known_image = face_recognition.load_image_file("biden.jpg")
# unknown_image = face_recognition.load_image_file("unknown.jpg")

# biden_encoding = face_recognition.face_encodings(known_image)[0]
# unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# results = face_recognition.compare_faces([biden_encoding], unknown_encoding)




