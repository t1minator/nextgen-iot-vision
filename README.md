# IoT Vision Starter

A FastAPI-based computer vision service for analyzing videos and detecting objects including people, animals, packages, and vehicles using YOLOv8.

## Features

- **Real-time object detection** using YOLOv8
- **Multi-category detection**:
  - 👤 **People** - Human detection
  - 🐾 **Animals** - Birds, cats, dogs, horses, etc.
  - 📦 **Packages** - Suitcases, backpacks, handbags
  - 🚗 **Vehicles** - Cars, trucks, motorcycles, buses, trains, boats, bicycles, airplanes
- **Video processing** - Analyzes multiple frames per video
- **REST API** with automatic documentation
- **Comprehensive logging** - File and console logging
- **Confidence scoring** and bounding box coordinates

## Requirements

- Python 3.8+
- OpenCV-compatible video formats (MP4, AVI, MOV, etc.)

## Installation

1. **Clone or download the project**:
   ```bash
   cd iot-vision-starter
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - FastAPI - Web framework
   - Uvicorn - ASGI server
   - OpenCV - Computer vision library
   - Ultralytics - YOLOv8 implementation
   - Additional dependencies (numpy, pillow, python-multipart)

3. **First run** - The YOLOv8 model will automatically download (~6MB) on first use.

## Running the Service

### Start the server:
```bash
python main.py
```

The service will start on `http://localhost:8000`

### Alternative startup:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Usage

### 1. Interactive API Documentation
Visit `http://localhost:8000/docs` in your browser for the automatic FastAPI documentation where you can:
- Upload video files directly
- Test the API interactively
- View response schemas

### 2. Command Line (curl)
```bash
curl -X POST "http://localhost:8000/analyze/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/video.mp4"
```

### 3. Python Script
```python
import requests

url = "http://localhost:8000/analyze/"
files = {"file": open("path/to/your/video.mp4", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### 4. HTML Form
```html
<!DOCTYPE html>
<html>
<body>
    <form action="http://localhost:8000/analyze/" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="video/*">
        <input type="submit" value="Analyze Video">
    </form>
</body>
</html>
```

## API Response Format

```json
{
  "video": "example.mp4",
  "frames_analyzed": 5,
  "detections": [
    {
      "label": "person",
      "confidence": 0.89,
      "bbox": [100, 150, 300, 450],
      "class_id": 0
    },
    {
      "label": "vehicle",
      "confidence": 0.76,
      "bbox": [400, 200, 600, 400],
      "class_id": 2
    }
  ],
  "total_raw_detections": 8
}
```

### Response Fields:
- `video`: Original filename
- `frames_analyzed`: Number of frames processed
- `detections`: Array of unique detections (highest confidence per category)
- `total_raw_detections`: Total detections across all frames
- `label`: Object category (person, animal, package, vehicle)
- `confidence`: Detection confidence (0.0-1.0)
- `bbox`: Bounding box coordinates [x1, y1, x2, y2]
- `class_id`: Original COCO dataset class ID

## Configuration

### Confidence Threshold
Default: 0.5 (50% confidence)
Modify in `app/inference.py`:
```python
def process_detections(results, confidence_threshold: float = 0.5):
```

### Frame Sampling
Default: 5 frames per video
Modify in `app/inference.py`:
```python
frames = extract_frames(video_path, max_frames=5)
```

### Model Selection
Default: YOLOv8 Nano (fastest)
Options in `app/inference.py`:
```python
model = YOLO('yolov8n.pt')  # nano (fastest)
model = YOLO('yolov8s.pt')  # small
model = YOLO('yolov8m.pt')  # medium
model = YOLO('yolov8l.pt')  # large
model = YOLO('yolov8x.pt')  # extra large (most accurate)
```

## Logging

Logs are written to:
- **Console**: Real-time output
- **File**: `app.log` (persistent logging)

Log levels include processing steps, detection results, and error handling.

## Supported Video Formats

- MP4, AVI, MOV, WMV, FLV, MKV
- Most common codecs supported by OpenCV

## Troubleshooting

### Common Issues:

1. **Port already in use**:
   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill the process
   kill <PID>
   ```

2. **Video file not supported**:
   - Ensure file is a valid video format
   - Check file isn't corrupted
   - Try converting to MP4

3. **Model download fails**:
   - Check internet connection
   - Model downloads automatically on first use
   - Manual download: The model will be cached locally after first download

4. **Low detection accuracy**:
   - Adjust confidence threshold
   - Use larger YOLO model (yolov8s, yolov8m, etc.)
   - Ensure good video quality

## Development

### Project Structure:
```
iot-vision-starter/
├── main.py              # FastAPI application
├── app/
│   └── inference.py     # Computer vision logic
├── requirements.txt     # Dependencies
├── app.log             # Application logs
└── README.md           # This file
```

### Adding New Detection Categories:
Modify the `target_classes` dictionary in `app/inference.py` to include additional COCO classes.

## License

This project is a starter template for IoT vision applications.
