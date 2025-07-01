from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import tempfile
import os
import cv2
import base64
import logging
import json
from pathlib import Path
from app.inference import run_inference_on_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...)):
    try:
        logger.info(f"Received video upload: {file.filename} ({file.content_type})")

        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        logger.info(f"Saved uploaded file to: {tmp_path}")

        result = run_inference_on_video(tmp_path)

        logger.info(f"Analysis complete. Response: {json.dumps(result, indent=2)}")

        os.remove(tmp_path)
        logger.info(f"Cleaned up temporary file: {tmp_path}")

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error processing video {file.filename}: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
