# import numpy as np
# import torch
# from PIL import Image
# from torchvision import transforms

# # 加载原始图片
# image_path = "image/640_480.jpg"
# original_image = Image.open(image_path).convert('RGB')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/yolo-tiny/yolos-tiny_model.pth"
# image_processor_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/yolo-tiny/yolos-tiny_image_processor.pth"
# model = torch.load(model_path, map_location=device)
# image_processor = torch.load(image_processor_path, map_location=device)

# # # 图像处理器和模型（示例，您需要根据实际情况配置）
# # image_processor = ... # 替换为实际的图像处理器
# # model = ... # 替换为实际的模型
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)

# # 图像预处理
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# # 参数设置
# step_size = 0.01
# num_iterations = 10
# grid_size = 10
# T_conf = 0.5

# def calculate_weights(predicted_objects, grid_size, height, width):
#     """
#     Calculate grid cell weights based on the number of detected objects.
#     """
#     weights = np.ones((grid_size, grid_size))
#     cell_height = height // grid_size
#     cell_width = width // grid_size
#     for obj in predicted_objects:
#         x, y, w, h = obj['bbox']
#         grid_x = int(x / cell_width)
#         grid_y = int(y / cell_height)
#         weights[grid_y, grid_x] *= 0.5  # Decrease weight if the cell is dense
#     return weights

# def generate_adversarial_image(original_image, model, step_size, num_iterations, grid_size, T_conf):
#     """
#     Generate an adversarial image using spatial attention for latency attacks on object detection models.
#     """
#     # Initialize the adversarial image as the original image
#     adversarial_image = np.array(original_image, dtype=np.float32)

#     # Get the dimensions of the image
#     height, width, _ = adversarial_image.shape

#     for iteration in range(num_iterations):
#         # Predict objects in the current adversarial image
#         images = [Image.fromarray(adversarial_image.astype(np.uint8))]
#         inputs = image_processor(images=images, return_tensors="pt").to(device)

#         with torch.no_grad():
#             outputs = model(inputs['pixel_values'])

#         target_sizes = torch.tensor([image.size[::-1] for image in images])
#         results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)
        
#         predicted_objects = []
#         for result in results:
#             for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
#                 predicted_objects.append({
#                     'confidence': score.item(),
#                     'class_prob': score.item(),  # Simplified for example
#                     'bbox': box.tolist()
#                 })

#         # Calculate weights for each grid cell
#         weights = calculate_weights(predicted_objects, grid_size, height, width)

#         # Compute the loss function L
#         loss = 0
#         for obj in predicted_objects:
#             confidence = obj['confidence']
#             class_prob = obj['class_prob']
#             if class_prob > T_conf:
#                 loss += np.log(confidence)
#             else:
#                 loss += np.log(class_prob)

#         # Compute the gradient of the loss with respect to the image
#         gradient = np.gradient(loss)

#         # Update the adversarial image
#         cell_height = height // grid_size
#         cell_width = width // grid_size
#         for i in range(grid_size):
#             for j in range(grid_size):
#                 x_start = i * cell_height
#                 x_end = (i + 1) * cell_height
#                 y_start = j * cell_width
#                 y_end = (j + 1) * cell_width
#                 # adversarial_image[x_start:x_end, y_start:y_end] += step_size * weights[i, j] * gradient[x_start:x_end, y_start:y_end]
#                 adversarial_image[x_start:x_end, y_start:y_end] += step_size * weights[i, j] * gradient[x_start:x_end, y_start:y_end]

#     return adversarial_image

# # 生成对抗图片
# adversarial_image = generate_adversarial_image(original_image, model, step_size, num_iterations, grid_size, T_conf)

# # 保存对抗图片
# adversarial_image = Image.fromarray(adversarial_image.astype(np.uint8))
# adversarial_image.save("adversarial_image/adversarial_image.jpg")

# # 打印结果
# images = [adversarial_image]
# inputs = image_processor(images=images, return_tensors="pt").to(device)

# with torch.no_grad():
#     outputs = model(inputs['pixel_values'])

# target_sizes = torch.tensor([image.size[::-1] for image in images])
# results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)

# for i, result in enumerate(results):
#     print(f"Result {i}:")
#     for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
#         box = [round(i, 2) for i in box.tolist()]
#         print(
#             f"Detected {model.config.id2label[label.item()]} with confidence "
#             f"{round(score.item(), 3)} at location {box}"
#         )























import numpy as np

def generate_adversarial_image(original_image, model, step_size, num_iterations, grid_size, T_conf):
    """
    Generate an adversarial image using spatial attention for latency attacks on object detection models.

    Parameters:
    - original_image: the original input image.
    - model: the object detection model.
    - step_size: step size for updating the adversarial image.
    - num_iterations: number of iterations to perform.
    - grid_size: size of the grid (m x m).
    - T_conf: confidence threshold.

    Returns:
    - adversarial_image: the generated adversarial image.
    """
    # Initialize the adversarial image as the original image
    adversarial_image = np.copy(original_image)

    # Get the dimensions of the image
    height, width, _ = original_image.shape

    # Calculate the size of each grid cell
    cell_height = height // grid_size
    cell_width = width // grid_size

    def calculate_weights(predicted_objects, grid_size):
        """
        Calculate grid cell weights based on the number of detected objects.
        """
        weights = np.ones((grid_size, grid_size))
        for obj in predicted_objects:
            x, y, w, h = obj['bbox']
            grid_x = int(x / cell_width)
            grid_y = int(y / cell_height)
            weights[grid_y, grid_x] *= 0.5  # Decrease weight if the cell is dense
        return weights

    for iteration in range(num_iterations):
        # Predict objects in the current adversarial image
        predicted_objects = model.predict(adversarial_image)

        # Calculate weights for each grid cell
        weights = calculate_weights(predicted_objects, grid_size)

        # Compute the loss function L
        loss = 0
        for obj in predicted_objects:
            confidence = obj['confidence']
            class_prob = obj['class_prob']
            if class_prob > T_conf:
                loss += np.log(confidence)
            else:
                loss += np.log(class_prob)

        # Compute the gradient of the loss with respect to the image
        gradient = np.gradient(loss)

        # Update the adversarial image
        for i in range(grid_size):
            for j in range(grid_size):
                x_start = i * cell_height
                x_end = (i + 1) * cell_height
                y_start = j * cell_width
                y_end = (j + 1) * cell_width
                adversarial_image[x_start:x_end, y_start:y_end] += step_size * weights[i, j] * gradient[x_start:x_end, y_start:y_end]

    return adversarial_image


