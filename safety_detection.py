from ultralytics import YOLO
import cv2
import cvzone
import math


classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']


def load_model(weights_path: str = "ppe.pt"):
    return YOLO(weights_path)


def detect_and_annotate(
    img,
    model,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.7,
    allowed_class_names=None,
):
    myColor = (0, 0, 255)
    frame_detections = []

    results = model.predict(img, conf=conf_threshold, iou=iou_threshold, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = float(math.ceil((box.conf[0] * 100)) / 100)
            cls = int(box.cls[0])
            currentClass = classNames[cls] if 0 <= cls < len(classNames) else str(cls)

            if allowed_class_names is not None and currentClass not in allowed_class_names:
                continue

            frame_detections.append({
                "class_id": cls,
                "class_name": currentClass,
                "confidence": float(conf),
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_xywh": [int(x1), int(y1), int(w), int(h)],
            })

            if currentClass in {'NO-Hardhat', 'NO-Safety Vest', 'NO-Mask'}:
                myColor = (0, 0, 255)
            elif currentClass in {'Hardhat', 'Safety Vest', 'Mask'}:
                myColor = (0, 255, 0)
            else:
                myColor = (255, 0, 0)

            cvzone.putTextRect(
                img,
                f'{currentClass} {conf}',
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=1,
                colorB=myColor,
                colorT=(255, 255, 255),
                colorR=myColor,
                offset=5,
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    return img, frame_detections
