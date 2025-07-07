# OF---OnlyFeet
An advanced computer vision system for real-time person counting on escalators using YOLOv8 Nano, OpenCV, zone-based detection, and advanced tracking techniques.
### *Real-Time Escalator Person Counter using YOLOv8 Nano & SORT*

---

###  What It Does

🔹 **Detects and tracks** people on escalators in real-time
🔹 Uses **YOLOv8 Nano** for fast and efficient object detection
🔹 Tracks individuals using the **SORT** algorithm
🔹 Displays bounding boxes and ID labels for each tracked person
![WhatsApp Image 2025-07-08 at 01 14 22_b156cfbe](https://github.com/user-attachments/assets/966da389-2a18-4862-af90-7aaf1bdfdb3d)
🔹 Applies **zone-based detection** to focus only on escalator region
![WhatsApp Image 2025-07-08 at 01 16 11_f39ab699](https://github.com/user-attachments/assets/0fdacd43-95f0-433c-8bc3-a53ee2319c7c)
---

###  Key Technologies

*  **YOLOv8 Nano** – Lightweight object detection
*  **SORT Tracker** – Simple real-time object tracking
*  **OpenCV + CvZone** – Image processing and video handling
*  **Zone-based ROI Masking** – Only detect within defined region

---

###  How It Works

1.  Load video and resize to 640×360
2.  Apply **region-based mask** to filter non-escalator areas
3.  Use YOLOv8 to detect **only 'person' class**
4.  Use **SORT** to track each person across frames
5.  Show **unique ID** and bounding box per person in real-time

---

###  Requirements

```bash
pip install ultralytics opencv-python numpy cvzone
```

add `sort.py` or use a pip-based SORT library.

---

###  Run the App

```bash
python yolo.py
```
 Tested on 1080p escalator video input.

---

###  Use Cases

*  Malls & Airports – Escalator usage monitoring
*  Metro Stations – Real-time crowd flow analysis
*  Smart Infrastructure – People counting systems

---

###  Project Structure

```
EscaCount/
├── person_counter.py
├── mask.png
├── video.mp4
├── requirements.txt
└── README.md
```

---

###  Made by Jainil

*“Because real-time person tracking should be both smart and scalable.”*

---
