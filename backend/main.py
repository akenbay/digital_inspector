from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO
import os
import tempfile
from pathlib import Path

app = FastAPI(title="Document Processing API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize YOLO model
model = None

def load_model():
    """Load YOLO model. Will download if not present."""
    global model
    try:
        # Using YOLOv8n (nano) model - lightweight and fast
        # You can change to yolo11n.pt or yolov8n.pt
        model_path = "yolo11n.pt"
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to YOLOv8
        try:
            model = YOLO("yolov8n.pt")
            print("Loaded YOLOv8n model as fallback")
        except Exception as fallback_error:
            print(f"Fallback model loading failed: {fallback_error}")
            raise

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

def convert_document_to_image(file_content: bytes, filename: str) -> np.ndarray:
    """Convert uploaded document to image format for YOLO processing."""
    try:
        # Try to read as image
        nparr = np.frombuffer(file_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            # If not an image, try opening with PIL
            pil_image = Image.open(io.BytesIO(file_content))
            # Convert PIL image to OpenCV format
            img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error converting document to image: {str(e)}")

def process_document_with_yolo(image: np.ndarray) -> tuple:
    """Process image with YOLO model and return annotated image."""
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")
    
    try:
        # Run YOLO inference
        results = model(image)
        
        # Draw bounding boxes on the image
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                # Only draw boxes with confidence > 0.5
                if confidence > 0.5:
                    # Draw rectangle
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_image, len([box for result in results for box in result.boxes if float(box.conf[0].cpu().numpy()) > 0.5])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document with YOLO: {str(e)}")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document and process it with YOLO."""
    try:
        # Read file content
        file_content = await file.read()
        
        # Convert document to image
        image = convert_document_to_image(file_content, file.filename)
        
        # Process with YOLO
        annotated_image, detections_count = process_document_with_yolo(image)
        
        # Save processed image to temporary file
        temp_dir = tempfile.gettempdir()
        output_filename = f"processed_{file.filename}"
        output_path = os.path.join(temp_dir, output_filename)
        
        # Encode image to JPEG format
        _, encoded_img = cv2.imencode('.jpg', annotated_image)
        
        # Save to temporary file
        with open(output_path, 'wb') as f:
            f.write(encoded_img.tobytes())
        
        return {
            "message": "Document processed successfully",
            "filename": output_filename,
            "detections": detections_count,
            "download_url": f"/download/{output_filename}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download the processed document."""
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="image/jpeg",
        filename=filename
    )

@app.get("/")
async def root():
    return {"message": "Document Processing API with YOLO"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

