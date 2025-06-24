# Sports Sponsorship Analytics using Gemini

This project is a web application that leverages the Google Gemini 1.5 Flash model to analyze sports videos and identify brand logo visibility. Users can upload a video, and the application will process it frame-by-frame to detect logos, track their appearances over time, and present a detailed analytics report.

The application is built with a Python backend for video processing and AI inference, and a Streamlit frontend for an interactive user experience.

## Features

* **AI-Powered Logo Detection**: Utilizes the multimodal capabilities of Google's Gemini 1.5 Flash to identify brand logos directly from video frames.
* **Interactive Web Interface**: A clean and user-friendly interface built with Streamlit for easy video uploading and results visualization.
* **Comprehensive Analytics Dashboard**:
    * **Logo Frequency**: A table and bar chart showing the total number of times each brand's logo appeared.
    * **Viewership Timeline**: A line graph illustrating the number of unique logos detected over the duration of the video.
* **Video Segmentation with Bounding Boxes**: Generates a short video clip from the analyzed footage, highlighting detected logos with drawn bounding boxes and labels.
* **Dynamic Frame Sampling**: Automatically adjusts the frame processing rate based on the video's length to ensure efficient analysis.
* **Robust Error Handling & Retries**: Includes logic to handle potential API errors (like rate limiting) with an exponential backoff strategy.
* **Secure API Key Management**: Safely retrieves the Google API key from environment variables, Streamlit secrets, or a direct user input field in the app's sidebar.

## How It Works

1.  **Video Upload**: The user uploads a video file (MP4, MOV, AVI, etc.) through the Streamlit interface.
2.  **Frame Extraction**: The backend, powered by OpenCV, extracts frames from the video at an intelligently determined interval (more frequent for shorter videos, less for longer ones).
3.  **Logo Detection**: Each extracted frame is sent to the Gemini 1.5 Flash model with a prompt asking it to identify visible brand logos.
4.  **Data Aggregation**: The results from all frames are collected. The `GeminiLogoDetector` class tracks the count and timestamps for each unique logo.
5.  **Analytics Generation**: The aggregated data is processed using Pandas and displayed as interactive charts and tables with Altair.
6.  **Video Segment Creation**: A 7-second clip, starting near the first logo appearance, is processed again. This time, Gemini is prompted to provide bounding box coordinates for each logo, which are then drawn onto the frames to create a new, annotated video file.
7.  **Display**: The annotated video segment and the full analytics dashboard are presented to the user.

## Technology Stack

* **Core AI Model**: Google Gemini 2.5 Flash
* **Web Framework**: Streamlit
* **Video/Image Processing**: OpenCV (`opencv-python-headless`), Pillow (PIL)
* **Data Manipulation & Visualization**: Pandas, Altair
* **API & Environment**: `google-generativeai`, `python-dotenv`

## Setup & Installation

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.8+
* A Google Gemini API Key. You can get one from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 1. Setting up
# .env file
GOOGLE_API_KEY="your-google-gemini-api-key"
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# running the app
streamlit run streamlit_app.py
