from ultralytics import YOLO
import cv2
from sort import *
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(r"C:\JAINIL\opencv\1338590-hd_1920_1080_30fps.mp4")
#height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#cap.set(3, 1280)  
#cap.set(4, 720)  
img = cv2.imread(r"C:\JAINIL\opencv\mask (2).png")
img = cv2.resize(img, (640, 360))
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 360))
    imgreg = cv2.bitwise_and(frame, img)
    if not ret:
        break
    detections = np.empty((0, 5))
    results = model(imgreg, stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            current_class = model.names[cls]
            if current_class not in ["person"]:
                continue
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            
            current_array=np.array([x1,y1,x2,y2,conf])
            detections = np.vstack((detections, current_array))

    result_tracker=tracker.update(detections)

    for i in range(len(result_tracker)):
        x1, y1, x2, y2, obj_id = result_tracker[i]
        print(result_tracker[i])
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {int(obj_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 Detection", frame)
    cv2.imshow("imageregion", imgreg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break