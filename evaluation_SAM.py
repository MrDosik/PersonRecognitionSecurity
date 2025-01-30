import os
import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToTensor

# Project paths
sys.path.append("/home/dos/PycharmProjects/PersonRecognitionSecurity/NaturalisticAdversarialPatch")
sys.path.append("/home/dos/PycharmProjects/PersonRecognitionSecurity/NaturalisticAdversarialPatch/PyTorchYOLOv3")

# Are not applied?
import count_map.main as eval_map
from detect import DetectorYolov3

# **Set Device to CPU**
device = torch.device("cpu")

dataset_folder = "/home/dos/PycharmProjects/PersonRecognitionSecurity/datasets/e88v3t_after_SAM1"
ground_truth_labels_folder = "/home/dos/PycharmProjects/PersonRecognitionSecurity/NaturalisticAdversarialPatch/dataset/inria/Test/pos/yolo-labels_yolov3"
detected_labels_folder = "/home/dos/PycharmProjects/PersonRecognitionSecurity/datasets/e88v3t_after_SAM1/detected_labels"

# New directory for detected labels
os.makedirs(detected_labels_folder, exist_ok=True)

# YOLOv3 config and weights
yolo_config_path = "/home/dos/PycharmProjects/PersonRecognitionSecurity/NaturalisticAdversarialPatch/PyTorchYOLOv3/config/yolov3.cfg"
yolo_weights_path = "/home/dos/PycharmProjects/PersonRecognitionSecurity/NaturalisticAdversarialPatch/PyTorchYOLOv3/weights/yolov3.weights"

# Load YOLOv3 detector
detector = DetectorYolov3(show_detail=False, cfgfile=yolo_config_path, weightfile=yolo_weights_path)

# Every image
print("\n🚀 Running evaluation on images in:", dataset_folder)
for img_name in tqdm(os.listdir(dataset_folder)):
    if not img_name.endswith(".png"):
        continue  # Skip non-png-image files

    # Load image
    img_path = os.path.join(dataset_folder, img_name)
    image = cv2.imread(img_path)
    img_tensor = ToTensor()(image).unsqueeze(0).to(device)

    # Run YOLOv3 detection
    max_prob_obj_cls, overlap_score, bboxes = detector.detect(
        input_imgs=img_tensor, cls_id_attacked=0, with_bbox=True
    )

    # Save detected labels
    label_file = os.path.join(detected_labels_folder, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))
    with open(label_file, "w") as f:
        for box in bboxes[0]:  # Assuming batch size 1
            box_data = box.detach().cpu().numpy()

            if len(box_data) >= 6:  # Ensure valid format
                x1, y1, x2, y2, conf, class_id = box_data[:6]  # Take only first 6 elements
            else:
                print(f"Warning: Unexpected bounding box format -> {box_data}")
                continue  # Skip invalid boxes

            # Convert to YOLO format
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1

            # Write to file
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")

# Run mAP Evaluation
print("\n📊 Computing mAP (Mean Average Precision)...")
output_map = eval_map.count(
    path_ground_truth=ground_truth_labels_folder,
    path_detection_results=detected_labels_folder,
    path_images_optional=None  # Skip
)

# Print results*
print("\n🎯 Evaluation Completed! mAP Score:", output_map)
