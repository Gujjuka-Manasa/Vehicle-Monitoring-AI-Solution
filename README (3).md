
# 🚗 Vehicle Monitoring AI Solution

An AI-powered system for real-time vehicle monitoring using automatic number plate recognition (ANPR) with YOLOv8 and Streamlit.

---

## 🎯 Objectives

- Detect and recognize vehicle number plates using AI.
- Monitor real-time vehicle activity through live feed.
- Log and alert vehicles in unauthorized locations.
- Provide an interactive UI for users to view and manage data.

---

## 🛠️ Tools and Technologies

- **Programming Language**: Python
- **Object Detection**: YOLOv8 (via Roboflow)
- **OCR**: EasyOCR, PaddleOCR
- **Web Framework**: Streamlit
- **Libraries**: OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Imutils
- **Storage**: CSV
- **Deployment**: Streamlit
- **IDE**: Jupyter Notebook

---

## ⚙️ Methodology

1. **Camera Feed** (Laptop & Mobile cameras)
2. **Image Processing**: Grayscale, Edge Detection, Contour Detection
3. **Object Detection**: YOLOv8 for number plate localization
4. **Text Recognition**: OCR on cropped number plate regions
5. **Data Logging**: Save results to CSV with timestamps and camera IDs
6. **Alert Generation**: Trigger alerts for incorrect locations
7. **Streamlit UI**: Real-time feed, alerts dashboard, and visual analytics

---

## 🏗️ Architecture

Cameras ─▶ Preprocessing ─▶ YOLOv8 Detection ─▶ OCR ─▶
CSV Storage ─▶ Location Verification ─▶ Alerts ─▶ Streamlit UI
