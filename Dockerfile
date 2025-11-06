FROM python:3.10-slim

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY src/ ./src/
COPY .dockerignore .

# Pythonパスを設定
ENV PYTHONPATH=/app

# エントリーポイント
CMD ["python", "-m", "uvicorn", "src.web_app:app", "--host", "0.0.0.0", "--port", "8000"]

