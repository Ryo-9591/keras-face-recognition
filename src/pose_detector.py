"""
姿勢検出モジュール
TensorFlow/Keras（MoveNet）を使用して骨格キーポイントを検出
"""
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class PoseDetector:
    """TensorFlow/Keras（MoveNet）を使用した姿勢検出クラス"""
    
    # MoveNetのキーポイントID
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    # 骨格接続（描画用）
    CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 頭部
        (5, 6),  # 肩
        (5, 7), (7, 9),  # 左腕
        (6, 8), (8, 10),  # 右腕
        (5, 11), (6, 12),  # 胴体上部
        (11, 12),  # 腰
        (11, 13), (13, 15),  # 左足
        (12, 14), (14, 16),  # 右足
    ]
    
    def __init__(self, model_name="movenet_lightning"):
        """
        初期化
        
        Args:
            model_name: MoveNetモデル名（"movenet_lightning" または "movenet_thunder"）
        """
        # TensorFlow HubからMoveNetモデルを読み込む
        model_url = f"https://tfhub.dev/google/movenet/singlepose/{model_name}/4"
        print(f"MoveNetモデルを読み込んでいます: {model_name}")
        self.model = hub.load(model_url)
        self.input_size = 192 if model_name == "movenet_lightning" else 256
        
    def preprocess_image(self, image):
        """
        画像を前処理（リサイズと正規化）
        
        Args:
            image: BGR画像（OpenCV形式）
            
        Returns:
            processed_image: 前処理済み画像（RGB、リサイズ済み）
            original_shape: 元の画像サイズ
        """
        original_shape = image.shape[:2]
        # RGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # リサイズ
        image_resized = cv2.resize(image_rgb, (self.input_size, self.input_size))
        # 正規化（0-255 -> 0-1）
        image_normalized = image_resized.astype(np.float32) / 255.0
        # バッチ次元を追加
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch, original_shape
    
    def detect(self, image):
        """
        画像から姿勢を検出
        
        Args:
            image: BGR画像（OpenCV形式）
            
        Returns:
            results: 検出結果（キーポイント座標と信頼度）
            annotated_image: 骨格が描画された画像
        """
        # 前処理
        image_batch, original_shape = self.preprocess_image(image)
        
        # 推論
        outputs = self.model(tf.constant(image_batch))
        keypoints = outputs['output_0'].numpy()[0]
        
        # 元の画像サイズにスケール変換
        height, width = original_shape
        scale_x = width / self.input_size
        scale_y = height / self.input_size
        
        # キーポイントを元の画像サイズに変換
        keypoints_scaled = []
        for kp in keypoints:
            x = kp[1] * scale_x  # MoveNetは (y, x, confidence) の順
            y = kp[0] * scale_y
            confidence = kp[2]
            keypoints_scaled.append({
                'x': x / width,  # 正規化座標（0-1）
                'y': y / height,
                'confidence': confidence
            })
        
        # 骨格を描画
        annotated_image = image.copy()
        self._draw_skeleton(annotated_image, keypoints_scaled, width, height)
        
        return {'keypoints': keypoints_scaled}, annotated_image
    
    def _draw_skeleton(self, image, keypoints, width, height):
        """
        骨格を描画
        
        Args:
            image: 描画先の画像
            keypoints: キーポイントのリスト
            width: 画像の幅
            height: 画像の高さ
        """
        # 接続線を描画
        for connection in self.CONNECTIONS:
            idx1, idx2 = connection
            kp1 = keypoints[idx1]
            kp2 = keypoints[idx2]
            
            # 信頼度が低い場合はスキップ
            if kp1['confidence'] < 0.3 or kp2['confidence'] < 0.3:
                continue
            
            pt1 = (int(kp1['x'] * width), int(kp1['y'] * height))
            pt2 = (int(kp2['x'] * width), int(kp2['y'] * height))
            cv2.line(image, pt1, pt2, (0, 0, 255), 2)
        
        # キーポイントを描画
        for idx, kp in enumerate(keypoints):
            if kp['confidence'] < 0.3:
                continue
            
            x = int(kp['x'] * width)
            y = int(kp['y'] * height)
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    
    def get_landmark(self, results, landmark_id):
        """
        特定のランドマークを取得
        
        Args:
            results: 検出結果
            landmark_id: ランドマークID
            
        Returns:
            landmark: ランドマーク情報（x, y, confidence）またはNone
        """
        if 'keypoints' in results and landmark_id < len(results['keypoints']):
            kp = results['keypoints'][landmark_id]
            if kp['confidence'] > 0.3:
                return kp
        return None
    
    def get_landmarks_dict(self, results):
        """
        すべてのランドマークを辞書形式で取得
        
        Args:
            results: 検出結果
            
        Returns:
            landmarks_dict: ランドマークIDをキーとする辞書
        """
        landmarks_dict = {}
        if 'keypoints' in results:
            for idx, kp in enumerate(results['keypoints']):
                landmarks_dict[idx] = {
                    'x': kp['x'],
                    'y': kp['y'],
                    'confidence': kp['confidence'],
                    'visibility': kp['confidence']  # MoveNetはvisibilityの代わりにconfidenceを使用
                }
        return landmarks_dict
