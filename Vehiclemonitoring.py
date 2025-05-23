import streamlit as st
import pandas as pd
import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import easyocr
import imutils
import tempfile
import time

# Set the title
st.title("Vehicle Monitoring")

# Sidebar options
st.sidebar.title("Navigation")
view = st.sidebar.radio("Choose a View", ["Dashboard", "Live Feed", "Alert Mechanism"])

# Paths for video file, output CSV, and training dataset
output_csv_path = r"C:\Users\manas\OneDrive\Desktop\text analytics\detected_number_plates.csv"  # Detected CSV path
training_csv_path = r"C:\Users\manas\OneDrive\Desktop\text analytics\Training vehicle_number plate.csv"  # Training data path

# Load data for visualization and alert mechanisms
def load_data():
    try:
        df = pd.read_csv(output_csv_path)
        return df
    except FileNotFoundError:
        st.error("Detected number plates CSV file not found!")
        return None

# Load training data
def load_training_data():
    try:
        df_training = pd.read_csv(training_csv_path)
        return df_training['Number Plate'].values.tolist()  # Only return the list of training plates
    except FileNotFoundError:
        st.error(f"Training dataset {training_csv_path} not found!")
        return []

# Dashboard Visualization
def dashboard():
    st.header("Dashboard - Number Plate Detection Insights")

    df = load_data()
    if df is not None and not df.empty:
        # Visualization 1: Bar chart for detections by date
        st.subheader("Number Plates Detected by Date")
        date_counts = df['Date'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        date_counts.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title("Number Plates Detected by Date")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Visualization 2: Pie chart of unique number plates detected
        st.subheader("Proportion of Unique Number Plates Detected")
        plate_counts = df['Number Plate'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        plate_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"), ax=ax)
        ax.set_ylabel("")  # Remove y-axis label
        ax.set_title("Proportion of Unique Number Plates Detected")
        st.pyplot(fig)

        # Visualization 3: Heatmap of detection frequency by hour
        st.subheader("Detection Frequency Heatmap (by Hour)")
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
        hour_counts = df.groupby(['Date', 'Hour']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(hour_counts, cmap="Blues", annot=True, fmt="d", cbar=True, ax=ax)
        ax.set_title("Detection Frequency Heatmap (by Hour)")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Date")
        st.pyplot(fig)

        # Visualization 4: Histogram of detection times
        st.subheader("Distribution of Detections by Hour of Day")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Hour'], bins=24, kde=False, color="orange", ax=ax)
        ax.set_title("Distribution of Detections by Hour of Day")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Frequency")
        plt.xticks(range(0, 24))
        plt.grid()
        st.pyplot(fig)
    else:
        st.warning("No data available for visualizations.")

# Live Feed Processing
def live_feed():
    st.header("Live Feed - Number Plate Detection")
    st.text("Processing uploaded video for number plate detection. Detected plates will be saved to CSV.")

    # File uploader for video
    uploaded_video = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
    if uploaded_video:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        # Initialize EasyOCR
        reader = easyocr.Reader(['en'])

        # Load training dataset to check detected plates against
        training_data = load_training_data()

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Unable to open video.")
            return

        # Placeholder for video feed
        placeholder = st.empty()

        # Prepare data for saving to CSV
        detected_plates = set()
        detected_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(bfilter, 30, 200)

            keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(keypoints)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            location = None
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 10, True)
                if len(approx) == 4:
                    location = approx
                    break

            if location is not None:
                mask = np.zeros(gray.shape, np.uint8)
                new_image = cv2.drawContours(mask, [location], 0, 255, -1)
                new_image = cv2.bitwise_and(frame, frame, mask=mask)

                (x, y) = np.where(mask == 255)
                (x1, y1) = (np.min(x), np.min(y))
                (x2, y2) = (np.max(x), np.max(y))
                cropped_image = gray[x1:x2+1, y1:y2+1]

                result = reader.readtext(cropped_image)
                if result:
                    text = result[0][1].strip()
                    # Check if the detected plate matches the training dataset
                    if text not in detected_plates and text in training_data:
                        detected_plates.add(text)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        detected_data.append({"Number Plate": text, "Date": timestamp.split(" ")[0], "Time": timestamp.split(" ")[1]})

                        # Draw rectangle and text on frame
                        cv2.putText(frame, text, (location[0][0][0], location[1][0][1] + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.rectangle(frame, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

            # Display frame in Streamlit
            placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Save detected data to CSV
        cap.release()
        if detected_data:
            detected_df = pd.DataFrame(detected_data)
            detected_df.to_csv(output_csv_path, index=False)
            st.success(f"Detection complete! Detected number plates saved to {output_csv_path}.")
        else:
            st.warning("No number plates were detected.")

# Alert Mechanism
def alert_mechanism():
    st.header("Alert Mechanism - Detected Number Plates")
    df = load_data()
    if df is not None and not df.empty:
        st.dataframe(df[['Number Plate', 'Date', 'Time']].drop_duplicates())  # Display only unique plates
    else:
        st.warning("No detected number plates to display.")

# Main Functionality
if view == "Dashboard":
    dashboard()
elif view == "Live Feed":
    live_feed()
elif view == "Alert Mechanism":
    alert_mechanism()
