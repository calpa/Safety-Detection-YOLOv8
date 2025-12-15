from ultralytics import YOLO
import cv2
import cvzone
import math
import json
import os

# Open video file for reading
cap = cv2.VideoCapture("./videos/test.mp4") 

# Load YOLO model for detecting objects related to personal protective equipment (PPE)
model = YOLO("ppe.pt")

# Class names for different objects detected by the model
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

# Default color for drawing bounding boxes
myColor = (0, 0, 255)

output_dir = os.path.join(os.getcwd(), "outputs")
images_dir = os.path.join(output_dir, "images")
json_dir = os.path.join(output_dir, "json")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

frame_idx = 0

# Main loop to process each frame of the video
while True:
    # Read a frame from the video
    success, img = cap.read()

    if not success or img is None:
        break

    frame_idx += 1
    frame_timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))

    # Perform object detection using YOLO on the current frame
    results = model(img, stream=True)

    frame_detections = []

    # Process the results of object detection
    for r in results:
        # Extract bounding box information for each detected object
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Calculate confidence and class index
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            frame_detections.append({
                "class_id": cls,
                "class_name": currentClass,
                "confidence": float(conf),
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_xywh": [int(x1), int(y1), int(w), int(h)],
            })

            # Set color based on the class of the detected object
            if conf > 0.5:
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0, 255)  # Red for non-compliance
                elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                    myColor = (0, 255, 0)  # Green for compliance
                else:
                    myColor = (255, 0, 0)  # Blue for other objects

                # Display the class name and confidence on the image
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                
                # Draw bounding box around the detected object
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    frame_basename = f"frame_{frame_idx:06d}"
    image_path = os.path.join(images_dir, f"{frame_basename}.jpg")
    json_path = os.path.join(json_dir, f"{frame_basename}.json")

    cv2.imwrite(image_path, img)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "frame_index": frame_idx,
                "timestamp_ms": frame_timestamp_ms,
                "image_file": os.path.relpath(image_path, output_dir),
                "detections": frame_detections,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Display the annotated image
    cv2.imshow("Image", img)
    
    # Wait for a key press and continue the loop
    cv2.waitKey(1)
