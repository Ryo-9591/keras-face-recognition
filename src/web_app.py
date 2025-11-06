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
pose_detector = PoseDetector(model_name="movenet_lightning")
posture_analyzer = PostureAnalyzer(threshold_angle=35.0)


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
                margin: 0;
                padding: 0;
                background: #000;
            }
            .container {
                width: 100%;
                height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }
            .video-container {
                background: #000;
                display: flex;
                align-items: center;
                justify-content: center;
                max-width: 800px;
                max-height: 600px;
                width: 100%;
            }
            #videoElement {
                width: 100%;
                max-width: 800px;
                max-height: 600px;
                object-fit: contain;
            }
            #canvasElement {
                display: none;
            }
            .status-panel {
                background: #000;
                padding: 10px;
                color: #fff;
                font-size: 14px;
            }
            .status-item {
                margin: 5px 0;
            }
            .status-good {
                color: #0f0;
            }
            .status-warning {
                color: #f00;
            }
            .fps {
                color: #888;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="video-container">
                <video id="videoElement" autoplay playsinline></video>
                <canvas id="canvasElement"></canvas>
            </div>
            <div class="status-panel">
                <div id="status" class="status-item">検出中...</div>
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

                connectWebSocket();
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

            // ページ読み込み時に自動的にカメラを開始
            window.addEventListener('DOMContentLoaded', () => {
                startCamera();
            });
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
            results, _ = pose_detector.detect(frame)

            # ランドマークを取得
            landmarks_dict = pose_detector.get_landmarks_dict(results)

            # 猫背判定
            posture_result = posture_analyzer.analyze_posture(landmarks_dict)

            # 結果をJSONで送信
            await websocket.send_json(posture_result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
