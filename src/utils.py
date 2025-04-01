import os
from typing import Optional
import cv2

def create_project_directories():
    """Create necessary project directories if they don't exist"""
    directories = ['data', 'data/videos', 'data/frames', 'data/results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video file
    """
    try:
        cap = cv2.VideoCapture(video_path)
        info = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }
        cap.release()
        return info
    except Exception as e:
        print(f"Error getting video info: {str(e)}")
        return None 