import cv2
import os
import time
from datetime import datetime

save_folder = "/home/dos/PycharmProjects/PersonRecognitionSecurity/datasets/captured_images"

# Create the folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

print("Press SPACE to capture an image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Show live cam
    cv2.imshow("Webcam - Press SPACE to Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # SPACE key
        print("📸 Capturing in: ", end="", flush=True)

        # **New Countdown Loop with Live Feed**
        for i in range(10, 0, -1):
            ret, frame = cap.read()  # Keep capturing frames
            if not ret:
                print("\n❌ Failed to capture frame.")
                break

            # Overlay countdown text on the cam frame
            countdown_frame = frame.copy()
            cv2.putText(countdown_frame, f"Capturing in {i}...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show countdown frame
            cv2.imshow("Webcam - Press SPACE to Capture", countdown_frame)
            cv2.waitKey(1000)  # Wait 1 second while keep updating the window

        print("\n✅ Taking picture now!")

        # Capture final frame after countdown
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image")
            continue

        # Generate filename using timestamp
        filename = os.path.join(save_folder, f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ Image saved: {filename}")

    elif key == ord('q'):  # Quit when 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
