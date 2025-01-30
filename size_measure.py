import cv2

patch_path = "/home/dos/PycharmProjects/PersonRecognitionSecurity/NaturalisticAdversarialPatch/patch/e88v3t.png"
patch = cv2.imread(patch_path)

if patch is None:
    print("Error: Patch image not found!")
else:
    height, width, _ = patch.shape
    print(f"Patch size: {width}x{height} pixels")
