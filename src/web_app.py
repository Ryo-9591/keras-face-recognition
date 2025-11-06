"""
姿勢検出・猫背判定アプリ
FastAPI Webアプリケーション
"""
import cv2
import time
import threading
import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from camera_handler import CameraHandler
from pose_detector import PoseDetector
from posture_analyzer import PostureAnalyzer
from ui import UI

app = FastAPI(title="姿勢検出・猫背判定アプリ")

# グローバル変数
camera = None
pose_detector = None
posture_analyzer = None
ui = None
current_frame = None
current_posture_result = None
current_fps = 0
frame_lock = threading.Lock()


def init_modules():
    """モジュールの初期化"""
    global camera, pose_detector, posture_analyzer, ui
    
    camera = CameraHandler(camera_index=0, width=640, height=480)
    pose_detector = PoseDetector(model_name="movenet_lightning")
    posture_analyzer = PostureAnalyzer(threshold_angle=35.0)
    ui = UI(window_name="姿勢検出・猫背判定")
    
    if not camera.open():
        raise RuntimeError("カメラを開けませんでした。")
    
    print("カメラを開きました。")


def camera_loop():
    """カメラ処理ループ（バックグラウンドスレッド）"""
    global current_frame, current_posture_result, current_fps
    
    fps_counter = 0
    fps_start_time = time.time()
    
    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        # 姿勢検出
        results, annotated_frame = pose_detector.detect(frame)
        
        # ランドマークを取得
        landmarks_dict = pose_detector.get_landmarks_dict(results)
        
        # 猫背判定
        posture_result = posture_analyzer.analyze_posture(landmarks_dict)
        
        # FPS計算
        fps_counter += 1
        if fps_counter % 30 == 0:
            elapsed = time.time() - fps_start_time
            current_fps = 30 / elapsed
            fps_start_time = time.time()
        
        # UIに描画
        display_frame = ui.draw_result(annotated_frame, posture_result, current_fps)
        
        # フレームを更新
        with frame_lock:
            current_frame = display_frame
            current_posture_result = posture_result
        
        # 猫背の場合はコンソールにも出力
        if posture_result.get('is_slouched', False):
            print(f"[警告] {posture_result.get('message', '')}")
        
        time.sleep(0.01)  # CPU使用率を下げる


@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の処理"""
    print("姿勢検出・猫背判定アプリを起動しています...")
    init_modules()
    
    # カメラ処理をバックグラウンドスレッドで開始
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    print("Webサーバーが起動しました。http://localhost:8000 にアクセスしてください。")


@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーション終了時の処理"""
    global camera
    if camera:
        camera.release()
    print("アプリケーションを終了しました。")


@app.get("/", response_class=HTMLResponse)
async def root():
    """メインページ"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>姿勢検出・猫背判定アプリ</title>
        <meta charset="UTF-8">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .container {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                max-width: 900px;
                width: 100%;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .video-container {
                position: relative;
                width: 100%;
                background: #000;
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 20px;
            }
            #videoStream {
                width: 100%;
                display: block;
            }
            .status-panel {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
            }
            .status-item {
                margin: 10px 0;
                font-size: 18px;
            }
            .status-good {
                color: #28a745;
                font-weight: bold;
            }
            .status-warning {
                color: #dc3545;
                font-weight: bold;
            }
            .fps {
                color: #6c757d;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>姿勢検出・猫背判定アプリ</h1>
            <div class="video-container">
                <img id="videoStream" src="/video_feed" alt="カメラ映像">
            </div>
            <div class="status-panel">
                <div id="status" class="status-item">検出中...</div>
                <div id="angle" class="status-item"></div>
                <div id="confidence" class="status-item"></div>
                <div id="fps" class="status-item fps"></div>
            </div>
        </div>
        <script>
            // ステータス情報を定期的に取得
            setInterval(async () => {
                try {
                    const response = await fetch('/status');
                    const data = await response.json();
                    
                    const statusEl = document.getElementById('status');
                    const angleEl = document.getElementById('angle');
                    const confidenceEl = document.getElementById('confidence');
                    const fpsEl = document.getElementById('fps');
                    
                    if (data.is_slouched) {
                        statusEl.className = 'status-item status-warning';
                        statusEl.textContent = data.message;
                    } else {
                        statusEl.className = 'status-item status-good';
                        statusEl.textContent = data.message;
                    }
                    
                    if (data.angle !== null) {
                        angleEl.textContent = `角度: ${data.angle.toFixed(1)}度`;
                    }
                    
                    confidenceEl.textContent = `信頼度: ${(data.confidence * 100).toFixed(1)}%`;
                    fpsEl.textContent = `FPS: ${data.fps.toFixed(1)}`;
                } catch (error) {
                    console.error('ステータス取得エラー:', error);
                }
            }, 500);
        </script>
    </body>
    </html>
    """
    return html_content


@app.get("/video_feed")
async def video_feed():
    """MJPEGストリーミング"""
    def generate():
        while True:
            with frame_lock:
                if current_frame is None:
                    time.sleep(0.1)
                    continue
                frame = current_frame.copy()
            
            # JPEGにエンコード
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # MJPEGストリームとして送信
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # 約30fps
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
async def get_status():
    """現在の姿勢判定ステータスを取得"""
    global current_posture_result, current_fps
    
    with frame_lock:
        if current_posture_result is None:
            return {
                "is_slouched": False,
                "message": "検出中...",
                "angle": None,
                "confidence": 0.0,
                "fps": 0.0
            }
        
        result = current_posture_result.copy()
        result["fps"] = current_fps
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

