"""
FastAPI Backend for PPE Detection System
"""

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
import cv2
import numpy as np
import base64
from pathlib import Path
import shutil
import uuid
from datetime import datetime
import json
import os

from detection import PPEDetector

app = FastAPI(title="PPE Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = Path(__file__).parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
    print(f"✅ Frontend mounted: {FRONTEND_DIR}")
else:
    print(f"⚠️ Frontend not found: {FRONTEND_DIR}")

MODEL_PATH = BACKEND_DIR / 'best.pt'
print(f"🔍 Model path: {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

print("🚀 Initializing detector...")
detector = PPEDetector(str(MODEL_PATH))
print("✅ Backend ready!")

alarm_active = False

def trigger_alarm():
    global alarm_active
    alarm_active = True
    print("🚨 ALARM!")

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding='utf-8'))
    else:
        return HTMLResponse(content=f"""
        <html><body style="font-family:Arial; text-align:center; padding:50px;">
        <h1>🦺 PPE Detection Backend Running!</h1>
        <p>Frontend not found at: {FRONTEND_DIR}</p>
        <p><a href="/docs">API Docs</a></p>
        </body></html>
        """)

@app.get("/api")
async def api_info():
    return {
        "status": "running",
        "model_info": detector.get_model_info(),
        "directories": {
            "backend": str(BACKEND_DIR),
            "uploads": str(UPLOAD_DIR),
            "outputs": str(OUTPUT_DIR),
            "frontend": str(FRONTEND_DIR)
        }
    }

@app.get("/model-info")
async def get_model_info():
    return detector.get_model_info()

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    global alarm_active
    alarm_active = False
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = str(uuid.uuid4())[:8]
        input_filename = f"{timestamp}_{file_id}_input{Path(file.filename).suffix}"
        output_filename = f"{timestamp}_{file_id}_output.mp4"
        
        input_path = UPLOAD_DIR / input_filename
        output_path = OUTPUT_DIR / output_filename
        
        print(f"📥 Saving: {input_filename}")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"🔍 Processing...")
        results = detector.detect_video(
            str(input_path),
            str(output_path),
            alarm_callback=trigger_alarm
        )
        
        results['input_file'] = input_filename
        results['output_file'] = output_filename
        results['download_url'] = f"/outputs/{output_filename}"
        results['alarm_triggered'] = alarm_active
        results['status'] = 'success'
        results['timestamp'] = timestamp
        
        print(f"✅ Done: {output_filename}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.websocket("/ws/camera")
async def websocket_camera(websocket: WebSocket):
    await websocket.accept()
    print("📷 WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                if ',' in data:
                    data = data.split(',')[1]
                
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                annotated_frame, violation = detector.detect_frame(frame)
                
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                response = {
                    'frame': f'data:image/jpeg;base64,{frame_base64}',
                    'violation': violation,
                    'timestamp': datetime.now().isoformat()
                }
                
                await websocket.send_json(response)
                
                if violation:
                    print("🚨 Violation in camera!")
                
            except Exception as e:
                print(f"⚠️ Frame error: {e}")
                continue
                
    except WebSocketDisconnect:
        print("📷 Disconnected")
    except Exception as e:
        print(f"❌ WS error: {e}")

@app.get("/outputs/{filename}")
async def download_video(filename: str):
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "File not found"}
        )
    
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename
    )

@app.get("/list-videos")
async def list_videos():
    videos = []
    for video_file in OUTPUT_DIR.glob("*.mp4"):
        stat = video_file.stat()
        videos.append({
            "filename": video_file.name,
            "size_mb": round(stat.st_size / (1024*1024), 2),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "download_url": f"/outputs/{video_file.name}"
        })
    
    return {
        "total_videos": len(videos),
        "videos": sorted(videos, key=lambda x: x['created'], reverse=True)
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════╗
    ║   🦺 PPE Detection Server          ║
    ║   📡 http://localhost:8000          ║
    ╚══════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")