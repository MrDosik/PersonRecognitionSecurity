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

save_folder = "/home/dos/PycharmProjects/PersonRecognitionSecurity/datasets/captured_images"
label_output_folder = "/home/dos/PycharmProjects/PersonRecognitionSecurity/datasets/captured_images/yolo-labels"
output_folder = "/home/dos/PycharmProjects/PersonRecognitionSecurity/datasets/captured_images/boxed_images"

# Making sure that folder exist
os.makedirs(save_folder, exist_ok=True)
os.makedirs(label_output_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Paths to YOLOv3 config and weights
yolo_config_path = "/home/dos/PycharmProjects/PersonRecognitionSecurity/NaturalisticAdversarialPatch/PyTorchYOLOv3/config/yolov3.cfg"
yolo_weights_path = "/home/dos/PycharmProjects/PersonRecognitionSecurity/NaturalisticAdversarialPatch/PyTorchYOLOv3/weights/yolov3.weights"

# Init webcam and YOLOv3
cap = cv2.VideoCapture(0)
detector = DetectorYolov3(show_detail=False, cfgfile=yolo_config_path, weightfile=yolo_weights_path)

print("Press SPACE to capture an image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture image")
        break

    cv2.imshow("Webcam - Press SPACE to Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # space
        print("📸 Capturing in: ", end="", flush=True)

        # Countdown
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

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_name = f"image_{timestamp}.jpg"
        image_path = os.path.join(save_folder, image_name)

        # Save image
        cv2.imwrite(image_path, frame)
        print(f"✅ Image saved: {image_path}")

        # Convert image to tensor
        img_tensor = ToTensor()(frame).unsqueeze(0).to(device)

        # Run YOLO
        max_prob_obj_cls, overlap_score, bboxes = detector.detect(
            input_imgs=img_tensor, cls_id_attacked=0, with_bbox=True
        )

        # Generate label filename
        label_name = f"image_{timestamp}.txt"
        label_path = os.path.join(label_output_folder, label_name)

        # Save label file
        with open(label_path, "w") as f:
            for box in bboxes[0]:
                box_data = box.detach().cpu().numpy()

                if len(box_data) >= 6:
                    x1, y1, x2, y2, conf, class_id = box_data[:6]
                else:
                    print(f"⚠️ Warning: Unexpected bounding box format -> {box_data}")
                    continue  # Skip invalid boxes

                # Convert to YOLO format
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1


                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")

        print(f"✅ Label saved: {label_path}")

        # draw frames
        image = cv2.imread(image_path)
        h, w, _ = image.shape  # Get image dimensions

        # Read label file
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Process each bounding box
        for line in lines:
            values = line.strip().split()
            if len(values) < 5:
                continue  # Skip

            class_id = int(values[0])
            x_center, y_center, width, height, confidence = map(float, values[1:])

            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # Draw rectangle around detected person
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Person {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the new image with bounding boxes
        boxed_image_path = os.path.join(output_folder, f"boxed_{image_name}")
        cv2.imwrite(boxed_image_path, image)

        print(f"✅ Boxed Image saved at: {boxed_image_path}")

        # Display the image
        cv2.imshow("Detected Persons", image)
        cv2.waitKey(3000)  # pause

    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
