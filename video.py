import os
import cv2


def video2frames(video_path):
    cap = cv2.VideoCapture(video_path)
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frames = []
    for frame_index in range(frame_count):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frames.append(frame)
    
    cap.release()
    
    return frames
