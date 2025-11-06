"""
姿勢検出・猫背判定アプリ
メインアプリケーション（WebUI版）
"""

import uvicorn
from web_app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
