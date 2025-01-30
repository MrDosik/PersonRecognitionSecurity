## PersonRecognitionSecurity

## Overview
PersonRecognitionSecurity is a project aimed at generating, analyzing, and mitigating adversarial patches in person detection models. It consists of three main components:

1️⃣ **NaturalisticAdversarialPatch** – Generates adversarial patches to attack object detection models.
2️⃣ **PAD (Patch-based Adversarial Defense)** – Attempts to mitigate adversarial attacks.
3️⃣ **PatchCleanser** – Further cleanses the dataset by removing adversarial patches.

Each component has its own README.md and requirements.txt, providing instructions on usage.

## Project Structure
```
PersonRecognitionSecurity/
│── datasets/                        # Stores all datasets for visibility
│── NaturalisticAdversarialPatch/     # Generates adversarial patches and corrupts dataset
│── PAD/                              # Patch-based Adversarial Defense (Cleans dataset)
│── PatchCleanser/                     # Patch Cleanser (Cleans dataset further)
│── photo_only.py                     # Captures images and saves YOLO labels
│── photo_with_labels.py              # Captures images, detects persons, and saves YOLO labels
│── size_measure.py                    # Measures pixel size of patches
│── evaluation_SAM.py                  # Analyzes mAP of patched dataset (after SAM1 processing)
│── requirements.txt                    # Project dependencies
```

## Installation
### Install PatchCleanser Dependencies
```bash
pip install -r /home/dos/PycharmProjects/PersonRecognitionSecurity/PatchCleanser/requirements.txt
```

### Generate Adversarial Dataset (Start Here)
Navigate to **NaturalisticAdversarialPatch** and follow its README.md instructions to:
✅ Apply adversarial patches
✅ Generate a dataset of images with patches

```bash
cd /home/dos/PycharmProjects/PersonRecognitionSecurity/NaturalisticAdversarialPatch
```

## Workflow
1️⃣ **Generate an adversarially patched dataset**
   - Run **NaturalisticAdversarialPatch** to generate a corrupted dataset.
   - All generated images are stored in `/datasets/`.

2️⃣ **Clean the dataset using PAD and PatchCleanser**
   - Run **PAD** and **PatchCleanser** sequentially.
   - Follow their README.md instructions to process images.
   - Store the cleaned dataset in `/datasets/` for further analysis.

3️⃣ **Validate & Analyze the Cleaned Dataset**
   - Measure patch size using `size_measure.py`.
   - Evaluate detection performance using `evaluation_SAM.py`.
   - Capture real-time images using:
     - `photo_only.py` → Saves images & YOLO labels.
     - `photo_with_labels.py` → Saves images, detects persons, and saves labels.

## Key Scripts
| Script | Function |
|--------|----------|
| size_measure.py | Measures pixel size of patches before cleansing. |
| evaluation_SAM.py | Analyzes mAP of patched dataset after processing with SAM1. |
| photo_only.py | Captures a JPEG image and saves a YOLO label TXT file from the webcam. |
| photo_with_labels.py | Captures an image with person detection and bounding boxes. |

## Example Commands
### Train an Adversarial Patch (NaturalisticAdversarialPatch)
```bash
CUDA_VISIBLE_DEVICES=0 python ensemble.py --model=yolov4 --tiny
```
- `--model`: Detector model (yolov2, yolov3, yolov4, or fasterrcnn).
- `--tiny`: Enables YOLOv4-tiny mode.
- `--classBiggan`: ImageNet class used for generating the patch.

### Test an Adversarial Patch
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py --model yolov4 --tiny --patch ./patch_sample/v4tiny.png
```

### Run PatchCleanser
```bash
python pc_certification.py \
  --model vit_base_patch16_224_cutout2_128 \
  --dataset e88v3t_patched \
  --data_dir /home/dos/PycharmProjects/PersonRecognitionSecurity/datasets \
  --num_img -1 \
  --num_mask 6 \
  --patch_size 128 \
  --dump_dir /home/dos/PycharmProjects/PersonRecognitionSecurity/datasets/e88v3t_cleansed
```

## Storing Final Results
All final datasets should be stored in `/datasets/` for easy visualization and presentation of detection performance.

## Conclusion
This project identifies, defends against, and removes adversarial patches in person recognition models. By combining **NaturalisticAdversarialPatch**, **PAD**, and **PatchCleanser**, we ensure robust and secure person detection systems. 🚀