import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

# Initialize YOLO model (will download on first use)
model = None

def get_model():
    """Lazy load the YOLO model"""
    global model
    if model is None:
        logger.info("Loading YOLO model...")
        model = YOLO('yolov8n.pt')  # nano version for speed
        logger.info("Model loaded successfully!")
    return model

def extract_frames(video_path: str, max_frames: int = 10) -> List[np.ndarray]:
    """Extract frames from video for analysis"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)

    frame_count = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def process_detections(results, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Process YOLO results and filter for relevant objects"""
    detections = []

    # COCO class names that we're interested in
    target_classes = {
        0: 'person',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        # We'll map some objects as "package" - suitcase, backpack, handbag
        28: 'package',  # suitcase
        27: 'package',  # backpack
        26: 'package',  # handbag
        # Vehicle classes
        2: 'vehicle',   # car
        3: 'vehicle',   # motorcycle
        5: 'vehicle',   # bus
        7: 'vehicle',   # truck
        1: 'vehicle',   # bicycle
        4: 'vehicle',   # airplane
        6: 'vehicle',   # train
        8: 'vehicle',   # boat
    }

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence >= confidence_threshold and class_id in target_classes:
                    label = target_classes[class_id]

                    # Group animals under "animal" category except person
                    if class_id in [16, 17, 18, 19, 20, 21, 22, 23, 24, 25] and label != 'person':
                        label = 'animal'

                    # Vehicle classes are already labeled as 'vehicle' in target_classes
                    # No additional grouping needed for vehicles

                    detections.append({
                        "label": label,
                        "confidence": round(confidence, 2),
                        "bbox": box.xyxy[0].tolist(),  # bounding box coordinates
                        "class_id": class_id  # Include original class ID for debugging
                    })

    return detections

def run_inference_on_video(video_path: str) -> Dict[str, Any]:
    """Run YOLO inference on video and return detections"""
    try:
        logger.info(f"Processing video: {video_path}")

        # Get the model
        yolo_model = get_model()

        # Extract frames from video
        frames = extract_frames(video_path, max_frames=5)
        logger.info(f"Extracted {len(frames)} frames for analysis")

        if not frames:
            logger.warning("No frames could be extracted from video")
            return {
                "video": video_path,
                "error": "No frames could be extracted from video",
                "detections": []
            }

        # Run inference on frames
        all_detections = []
        for i, frame in enumerate(frames):
            logger.info(f"Processing frame {i+1}/{len(frames)}")
            results = yolo_model(frame)
            frame_detections = process_detections(results)
            logger.debug(f"Frame {i+1} detections: {frame_detections}")
            all_detections.extend(frame_detections)

        # Remove duplicates and get unique detections
        unique_detections = {}
        for detection in all_detections:
            label = detection["label"]
            if label not in unique_detections or detection["confidence"] > unique_detections[label]["confidence"]:
                unique_detections[label] = detection

        final_detections = list(unique_detections.values())

        logger.info(f"Found {len(final_detections)} unique detections: {[d['label'] for d in final_detections]}")

        result = {
            "video": os.path.basename(video_path),
            "frames_analyzed": len(frames),
            "detections": final_detections,
            "total_raw_detections": len(all_detections)
        }

        logger.info(f"Analysis complete for {video_path}")
        return result

    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}", exc_info=True)
        return {
            "video": video_path,
            "error": str(e),
            "detections": []
        }
