"""
姿勢検出・猫背判定アプリ
FastAPI Webアプリケーション
"""

import cv2
import base64
import json
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from src.pose_detector import PoseDetector
from src.posture_analyzer import PostureAnalyzer

app = FastAPI(title="姿勢検出・猫背判定アプリ")

# グローバル変数
pose_detector = None
posture_analyzer = None


def init_modules():
    """モジュールの初期化"""
    global pose_detector, posture_analyzer

    pose_detector = PoseDetector(model_name="movenet_lightning")
    posture_analyzer = PostureAnalyzer(threshold_angle=35.0)


@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の処理"""
    init_modules()


@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーション終了時の処理"""
    pass


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
            #videoElement {
                width: 100%;
                display: block;
            }
            #canvasElement {
                display: none;
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
            .button-container {
                text-align: center;
                margin-bottom: 20px;
            }
            button {
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                margin: 0 10px;
            }
            button:hover {
                background: #5568d3;
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>姿勢検出・猫背判定アプリ</h1>
            <div class="button-container">
                <button id="startBtn">カメラ開始</button>
                <button id="stopBtn" disabled>カメラ停止</button>
            </div>
            <div class="video-container">
                <video id="videoElement" autoplay playsinline></video>
                <canvas id="canvasElement"></canvas>
            </div>
            <div class="status-panel">
                <div id="status" class="status-item">カメラを開始してください</div>
                <div id="angle" class="status-item"></div>
                <div id="confidence" class="status-item"></div>
                <div id="fps" class="status-item fps"></div>
            </div>
        </div>
        <script>
            let websocket = null;
            let videoStream = null;
            let isProcessing = false;
            let fpsCounter = 0;
            let fpsStartTime = Date.now();
            let currentFps = 0;

            const videoElement = document.getElementById('videoElement');
            const canvasElement = document.getElementById('canvasElement');
            const canvas = canvasElement.getContext('2d');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const statusEl = document.getElementById('status');
            const angleEl = document.getElementById('angle');
            const confidenceEl = document.getElementById('confidence');
            const fpsEl = document.getElementById('fps');

            // WebSocket接続
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                websocket = new WebSocket(wsUrl);

                websocket.onopen = () => {
                    startProcessing();
                };

                websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    updateStatus(data);
                };
            }

            // カメラ開始
            async function startCamera() {
                videoStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 }
                });
                videoElement.srcObject = videoStream;
                
                // ビデオのサイズをキャンバスに設定
                videoElement.addEventListener('loadedmetadata', () => {
                    canvasElement.width = videoElement.videoWidth;
                    canvasElement.height = videoElement.videoHeight;
                });

                startBtn.disabled = true;
                stopBtn.disabled = false;
                connectWebSocket();
            }

            // カメラ停止
            function stopCamera() {
                if (videoStream) {
                    videoStream.getTracks().forEach(track => track.stop());
                    videoStream = null;
                }
                if (websocket) {
                    websocket.close();
                    websocket = null;
                }
                isProcessing = false;
                videoElement.srcObject = null;
                startBtn.disabled = false;
                stopBtn.disabled = true;
                statusEl.textContent = 'カメラを開始してください';
                angleEl.textContent = '';
                confidenceEl.textContent = '';
                fpsEl.textContent = '';
            }

            // フレーム処理開始
            function startProcessing() {
                if (isProcessing) return;
                isProcessing = true;
                processFrame();
            }

            // フレーム処理
            function processFrame() {
                if (!isProcessing || !videoStream || !websocket || websocket.readyState !== WebSocket.OPEN) {
                    return;
                }

                // キャンバスにフレームを描画
                canvas.drawImage(videoElement, 0, 0);
                
                // 画像をbase64エンコード
                const imageData = canvasElement.toDataURL('image/jpeg', 0.8);
                const base64Data = imageData.split(',')[1];

                // WebSocketで送信
                websocket.send(JSON.stringify({
                    type: 'frame',
                    data: base64Data
                }));

                // FPS計算
                fpsCounter++;
                if (fpsCounter % 30 === 0) {
                    const elapsed = (Date.now() - fpsStartTime) / 1000;
                    currentFps = 30 / elapsed;
                    fpsStartTime = Date.now();
                }

                // 次のフレームを処理（約30fps）
                setTimeout(processFrame, 33);
            }

            // ステータス更新
            function updateStatus(data) {
                if (data.is_slouched) {
                    statusEl.className = 'status-item status-warning';
                    statusEl.textContent = data.message;
                } else {
                    statusEl.className = 'status-item status-good';
                    statusEl.textContent = data.message;
                }

                if (data.angle !== null) {
                    angleEl.textContent = `角度: ${data.angle.toFixed(1)}度`;
                } else {
                    angleEl.textContent = '';
                }

                confidenceEl.textContent = `信頼度: ${(data.confidence * 100).toFixed(1)}%`;
                fpsEl.textContent = `FPS: ${currentFps.toFixed(1)}`;
            }

            // イベントリスナー
            startBtn.addEventListener('click', startCamera);
            stopBtn.addEventListener('click', stopCamera);
        </script>
    </body>
    </html>
    """
    return html_content


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocketエンドポイント - ブラウザからフレームを受信して姿勢検出"""
    await websocket.accept()

    while True:
        # ブラウザからフレームを受信
        data = await websocket.receive_text()
        message = json.loads(data)

        if message.get("type") == "frame":
            # base64デコード
            image_data = base64.b64decode(message["data"])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 姿勢検出
            results, annotated_frame = pose_detector.detect(frame)

            # ランドマークを取得
            landmarks_dict = pose_detector.get_landmarks_dict(results)

            # 猫背判定
            posture_result = posture_analyzer.analyze_posture(landmarks_dict)

            # 結果をJSONで送信
            await websocket.send_json(posture_result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
