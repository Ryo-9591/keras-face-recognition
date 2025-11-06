# 姿勢検出・猫背判定アプリ

PCのカメラ映像からリアルタイムで姿勢を検出し、猫背を判定するアプリケーションです。

## 機能

- **リアルタイム姿勢検出**: TensorFlow/Keras（MoveNet）を使用して骨格キーポイントを検出
- **猫背判定**: 首・肩・腰の角度から猫背を自動判定
- **WebUI**: FastAPIベースのWebインターフェースでブラウザからアクセス可能
- **視覚的フィードバック**: 骨格ラインと判定結果をリアルタイム表示
- **Docker対応**: Dockerコンテナ内で動作

## 要件

- Docker（カメラアクセス権限付き）
- PCカメラ（内蔵またはUSB接続）

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd keras-face-recognition
```

### 2. Docker Composeで起動（推奨）

```bash
docker compose up -d
```

バックグラウンドで起動します。ログを確認するには：

```bash
docker compose logs -f
```

停止するには：

```bash
docker compose down
```

### 3. Webブラウザでアクセス

コンテナが起動したら、以下のURLにアクセスしてください：

```
http://localhost:8000
```

ブラウザでカメラ映像と姿勢判定結果がリアルタイムで表示されます。

### 4. Dockerイメージを直接ビルド・実行する場合（オプション）

#### イメージのビルド

```bash
docker build -t posture-detector .
```

#### アプリケーションの起動

```bash
docker run --rm -it --device=/dev/video0 posture-detector
```

**注意**: Windows環境では、WSL2を使用している場合に`/dev/video0`が利用可能です。Windowsネイティブの場合は、カメラデバイスのパスを調整が必要な場合があります。

### 5. ローカル実行（Dockerなし）

Dockerを使用しない場合：

```bash
pip install -r requirements.txt
python src/main.py
```

## 使用方法

1. Docker Composeでアプリケーションを起動
2. ブラウザで `http://localhost:8000` にアクセス
3. カメラの前に座り、全身が映るように調整してください
4. アプリが自動的に姿勢を検出し、猫背の場合は警告を表示します
5. ブラウザを閉じるか、`docker compose down`で停止してください

## WebUI機能

- **リアルタイム映像**: MJPEGストリーミングでカメラ映像を配信
- **姿勢判定表示**: 判定結果、角度、信頼度をリアルタイム表示
- **FPS表示**: パフォーマンス監視
- **レスポンシブデザイン**: モダンなUIデザイン

## 技術仕様

- **姿勢検出**: TensorFlow/Keras（MoveNet Lightning）
- **深層学習フレームワーク**: TensorFlow 2.15, Keras 2.15
- **Webフレームワーク**: FastAPI
- **画像処理**: OpenCV
- **プログラミング言語**: Python 3.10
- **推論速度**: 15fps以上（環境により異なります）
- **WebUIポート**: 8000

## 猫背判定ロジック

首・肩・腰の3点から角度を計算し、以下の条件で猫背を判定します：

- 正常な姿勢: 首-肩-腰の角度が145度以上
- 猫背: 首-肩-腰の角度が145度未満

閾値は`src/posture_analyzer.py`の`threshold_angle`パラメータで調整可能です。

## トラブルシューティング

### カメラが開けない

**Windows環境でのカメラアクセス:**

1. **WSL2を使用している場合**:
   ```bash
   # WSL2内でカメラデバイスを確認
   ls -la /dev/video*
   
   # カメラが存在する場合、docker-compose.ymlのdevices設定で動作します
   ```

2. **WindowsネイティブのDocker Desktopの場合**:
   - Dockerコンテナから直接Windowsのカメラにアクセスするのは困難です
   - 以下のいずれかの方法を試してください：
     - WSL2バックエンドを使用する（推奨）
     - 仮想カメラドライバ（例: OBS Virtual Camera）を使用してストリームを共有
     - ホストマシンでカメラストリームをHTTPで配信し、コンテナからアクセス

3. **一般的な対処法**:
   - カメラが他のアプリケーションで使用されていないか確認
   - 環境変数 `CAMERA_INDEX` で異なるカメラインデックスを試行
   - アプリケーションは自動的に複数のカメラインデックス（0-4）を試行します

### 姿勢が検出されない

- 十分な明るさがあるか確認してください
- 全身がカメラに映るように調整してください
- 背景とのコントラストを確保してください

### パフォーマンスが低い

- 解像度を下げる（`src/camera_handler.py`の`width`と`height`を調整）
- GPUを使用する場合は、nvidia-dockerを使用してください

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 開発者向け情報

### プロジェクト構造

```
.
├── Dockerfile              # Docker設定ファイル
├── docker-compose.yml      # Docker Compose設定
├── requirements.txt        # Python依存関係
├── README.md              # このファイル
└── src/
    ├── __init__.py
    ├── main.py            # メインアプリケーション（WebUI起動）
    ├── web_app.py         # FastAPI Webアプリケーション
    ├── camera_handler.py  # カメラ処理
    ├── pose_detector.py   # 姿勢検出
    ├── posture_analyzer.py # 猫背判定
    └── ui.py              # UI表示（描画ロジック）
```

### カスタマイズ

- **猫背判定の閾値**: `src/posture_analyzer.py`の`threshold_angle`を変更
- **カメラ解像度**: `src/camera_handler.py`の`width`と`height`を変更
- **モデル選択**: `src/pose_detector.py`の`model_name`パラメータで"movenet_lightning"（高速）または"movenet_thunder"（高精度）を選択

