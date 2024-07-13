from transformers import YolosImageProcessor, YolosForObjectDetection
from torchvision import transforms
from PIL import Image
import torch

import time
from pprint import pprint
import numpy as np

image_path = "../image/640_480.jpg"
image = Image.open(image_path)
image_array = np.array(image)

# Load the object detection model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/yolo-tiny/yolos-tiny_model.pth"
image_processor_path = f"/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/model/yolo-tiny/yolos-tiny_image_processor.pth"
model = torch.load(model_path, map_location=device)
image_processor = torch.load(image_processor_path, map_location=device)


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
    height, width, _ = original_image.shape # (480, 640, 3)

    # Calculate the size of each grid cell
    cell_height = height // grid_size
    cell_width = width // grid_size

    def calculate_weights(predicted_objects, grid_size):
        """
        Calculate grid cell weights based on the number of detected objects.
        """
        weights = np.ones((grid_size, grid_size))
        for obj in predicted_objects:
            x1, y1, x2, y2 = obj['bbox']
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            # 计算中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            print(f"center_x = {center_x}, center_y = {center_y}")
            # 计算中心点所在的网格
            center_grid_x = center_x // cell_width
            center_grid_y = center_y // cell_height

            # weights[grid_y, grid_x] *= 0.5  # Decrease weight if the cell is dense
            weights[center_grid_y, center_grid_x] *= 0.5  # Decrease weight if the cell is dense
        
        print(f"weights = {weights}")
        
        return weights
    
    def inference(image):
        images = [image for _ in range(1)]
        inputs = image_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = image_processor.post_process_object_detection(outputs, threshold=0.9)
        
        predicted_objects = []
        for result in results:
            for i in range(len(result['labels'])):
                obj = {
                    'bbox': result['boxes'][i].tolist(),
                    'confidence': result['scores'][i].item(),
                    'class_prob': result['labels'][i].item()
                }
                predicted_objects.append(obj)
                
        for i, result in enumerate(results): # add
            print(f"Result {i}:")
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )
        
        # pprint(f"predicted_objects = {predicted_objects}")
        
        return predicted_objects

    for iteration in range(num_iterations):
        # Predict objects in the current adversarial image
        # predicted_objects = model.predict(adversarial_image)
        predicted_objects = inference(adversarial_image)

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
        print(f"loss = {loss}, gradient = {gradient}")

        # Update the adversarial image
        for i in range(grid_size):
            for j in range(grid_size):
                x_start = i * cell_height
                x_end = (i + 1) * cell_height
                y_start = j * cell_width
                y_end = (j + 1) * cell_width
                # adversarial_image[x_start:x_end, y_start:y_end] += step_size * weights[i, j] * gradient[x_start:x_end, y_start:y_end]
                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        for c in range(3):
                            try:
                                adversarial_image[x][y][c] += step_size * weights[i, j] * gradient[x][y][c]
                            except:
                                # print(f"error: x = {x}, y = {y}, c = {c}, i = {i}, j = {j}")
                                pass
                
        # # 在检测到的目标中心画一个小圆
        # import cv2
        # for obj in predicted_objects:
        #     x1, y1, x2, y2 = obj['bbox']
        #     x1 = int(x1 * width)
        #     y1 = int(y1 * height)
        #     x2 = int(x2 * width)
        #     y2 = int(y2 * height)
        #     # 计算中心点
        #     center_x = (x1 + x2) // 2
        #     center_y = (y1 + y2) // 2
        #     print(f"center_x = {center_x}, center_y = {center_y}")
        #     # 计算中心点所在的网格
        #     center_grid_x = center_x // cell_width
        #     center_grid_y = center_y // cell_height
        #     # 画一个小圆
        #     cv2.circle(adversarial_image, (center_x, center_y), 5, (0, 0, 255), -1)

    return adversarial_image


# Generate an adversarial image
step_size = 0.01
num_iterations = 1 # 10
grid_size = 4
T_conf = 0.9
print(f"image_array.shape = {image_array.shape}")
adversarial_image = generate_adversarial_image(image_array, model, step_size, num_iterations, grid_size, T_conf)

# Save the adversarial image
adversarial_image_path = "adversarial_image/640_480.jpg"
# adversarial_image = Image.fromarray(adversarial_image.transpose(1, 2, 0))
adversarial_image = Image.fromarray(adversarial_image)
adversarial_image.save(adversarial_image_path)

print(f"Adversarial image saved to {adversarial_image_path}")
