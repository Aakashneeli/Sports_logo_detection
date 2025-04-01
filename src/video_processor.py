import cv2
import numpy as np
from typing import Generator, Tuple
import os

class VideoProcessor:
    def __init__(self, video_path: str, frame_skip: int = 30):
        """
        Initialize video processor
        
        Args:
            video_path: Path to video file
            frame_skip: Number of frames to skip (30 = 1 frame per second for 30fps video)
        """
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.cap = None
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
            
    def get_frame_count(self) -> int:
        """Get total number of frames in video"""
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return max(1, (total_frames + self.frame_skip - 1) // self.frame_skip)
    
    def get_fps(self) -> float:
        """Get video FPS"""
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def process_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generate frames from video, skipping frames based on frame_skip
        
        Yields:
            Tuple[int, np.ndarray]: Frame number and frame image
        """
        total_frames = self.get_frame_count()
        processed_count = 0
        frame_number = 0
        
        while self.cap.isOpened() and processed_count < total_frames:
            ret, frame = self.cap.read()
            
            if not ret:
                break
                
            if frame_number % self.frame_skip == 0:
                processed_count += 1
                yield min(processed_count, total_frames), frame
                
            frame_number += 1 