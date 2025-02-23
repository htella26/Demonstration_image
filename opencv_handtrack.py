import cv2
import numpy as np

# Load pre-trained Haar Cascade for hand detection
hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
video_path = "video/Color_video1.MOV"  
cap = cv2.VideoCapture(video_path)

output_file = "hand_actions_opencv.txt"

prev_centroid = None
grasping = False
frame_count = 0

with open(output_file, "w") as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
 
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect hand(s) using Haar cascade
        hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in hands:
            # Compute centroid of hand bounding box
            centroid_x, centroid_y = x + w // 2, y + h // 2

            # Detect grasping based on bounding box size
            if w * h > 5000:  # Adjust threshold as needed
                if not grasping:
                    f.write(f"{frame_count},GRASPING,{x},{y},{w},{h}\n")
                    grasping = True
            else:
                if grasping:
                    f.write(f"{frame_count},RELEASING,{x},{y},{w},{h}\n")
                    grasping = False

            # Detect movement based on centroid change
            if prev_centroid:
                motion_distance = np.linalg.norm([centroid_x - prev_centroid[0], centroid_y - prev_centroid[1]])
                if motion_distance > 10:  # Adjust movement threshold
                    f.write(f"{frame_count},MOVING,{x},{y},{w},{h}\n")

            # Update previous centroid
            prev_centroid = (centroid_x, centroid_y)

            # Draw bounding box around hand
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display video (optional)
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Hand actions with dimensions saved in {output_file}")
