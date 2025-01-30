import sys
import os
import cv2
import time
from datetime import datetime
import torch
import numpy as np
from torchvision.transforms import ToTensor

sys.path.append("/home/dos/PycharmProjects/PersonRecognitionSecurity/NaturalisticAdversarialPatch/PyTorchYOLOv3")

from detect import DetectorYolov3

device = torch.device("cpu")

yolo_config_path = "/home/dos/PycharmProjects/PersonRecognitionSecurity/NaturalisticAdversarialPatch/PyTorchYOLOv3/config/yolov3.cfg"
yolo_weights_path = "/home/dos/PycharmProjects/PersonRecognitionSecurity/NaturalisticAdversarialPatch/PyTorchYOLOv3/weights/yolov3.weights"

save_folder = "/home/dos/PycharmProjects/PersonRecognitionSecurity/datasets/captured_images"
label_output_folder = "/home/dos/PycharmProjects/PersonRecognitionSecurity/datasets/captured_images/yolo-labels"

os.makedirs(save_folder, exist_ok=True)
os.makedirs(label_output_folder, exist_ok=True)

# init webcam
cap = cv2.VideoCapture(0)
detector = DetectorYolov3(show_detail=False, cfgfile=yolo_config_path, weightfile=yolo_weights_path)

print("Press SPACE to capture an image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    cv2.imshow("Webcam - Press SPACE to Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # space
        print("📸 Capturing in: ", end="", flush=True)

        # countdown
        for i in range(10, 0, -1):
            ret, frame = cap.read()
            if not ret:
                print("\n❌ Failed to capture frame.")
                break
            countdown_frame = frame.copy()
            cv2.putText(countdown_frame, f"Capturing in {i}...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Webcam - Press SPACE to Capture", countdown_frame)
            cv2.waitKey(1000)

        print("\n✅ Taking picture now!")
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image")
            continue

        # Save image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(save_folder, f"image_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ Image saved: {filename}")

        # Convert image to tensor
        img_tensor = ToTensor()(frame).unsqueeze(0).to(device)

        # Run YOLO detection
        max_prob_obj_cls, overlap_score, bboxes = detector.detect(
            input_imgs=img_tensor, cls_id_attacked=0, with_bbox=True
        )

        # Save label file
        label_file = os.path.join(label_output_folder, f"image_{timestamp}.txt")

        # handle length
        with open(label_file, "w") as f:
            for box in bboxes[0]: # size 1
                box_data = box.detach().cpu().numpy()

                if len(box_data) >= 6:  # output has at least 6 elements
                    x1, y1, x2, y2, conf, class_id = box_data[:6]  # only first 6 elements
                else:
                    print(f"⚠️ Warning: Unexpected bounding box format -> {box_data}")
                    continue  # Skip

                # Convert to YOLO format (normalized values)
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1

                # Write to file
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")

        print(f"✅ Label saved: {label_file}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
