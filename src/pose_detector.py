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

    def __init__(self, model_name="movenet_lightning"):
        """
        初期化

        Args:
            model_name: MoveNetモデル名（"movenet_lightning" または "movenet_thunder"）
        """
        # TensorFlow HubからMoveNetモデルを読み込む
        # モデル名をURL形式に変換（movenet_lightning -> lightning）
        if model_name == "movenet_lightning":
            hub_model_name = "lightning"
            self.input_size = 192
        elif model_name == "movenet_thunder":
            hub_model_name = "thunder"
            self.input_size = 256
        else:
            hub_model_name = "lightning"
            self.input_size = 192

        # 複数のURLを試行（バージョン4, 3, またはバージョンなし）
        model_urls = [
            f"https://tfhub.dev/google/movenet/singlepose/{hub_model_name}/4",
            f"https://tfhub.dev/google/movenet/singlepose/{hub_model_name}/3",
            f"https://tfhub.dev/google/movenet/singlepose/{hub_model_name}",
        ]

        self.model = None
        self.model_callable = None
        for model_url in model_urls:
            try:
                print(
                    f"MoveNetモデルを読み込んでいます: {model_name} (URL: {model_url})"
                )
                loaded_model = hub.load(model_url)

                # モデルがcallableかどうかを確認
                if callable(loaded_model):
                    self.model_callable = loaded_model
                    self.model = None
                elif hasattr(loaded_model, "signatures"):
                    self.model = loaded_model
                    self.model_callable = None
                else:
                    # フォールバック: 直接callableとして扱う
                    self.model_callable = loaded_model
                    self.model = None

                print(f"モデルの読み込みに成功しました: {model_url}")
                break
            except Exception as e:
                print(f"URL {model_url} の読み込みに失敗: {e}")
                continue

        if self.model is None and self.model_callable is None:
            raise RuntimeError(
                f"MoveNetモデルの読み込みに失敗しました。試行したURL: {model_urls}"
            )

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

        # TensorFlow Hubモデルを呼び出す
        if self.model_callable is not None:
            # callableな関数として呼び出す（float32で正規化済み）
            input_tensor = tf.constant(image_batch, dtype=tf.float32)
            outputs = self.model_callable(input_tensor)
        elif self.model is not None and hasattr(self.model, "signatures"):
            # SavedModelのsignaturesを使用
            signature = None
            if "serving_default" in self.model.signatures:
                signature = self.model.signatures["serving_default"]
            else:
                # 利用可能な最初のsignatureを使用
                signature_name = list(self.model.signatures.keys())[0]
                signature = self.model.signatures[signature_name]

            # signatureの入力仕様を確認
            # signature.structured_input_signatureから入力の型を取得
            expected_dtype = None
            try:
                if hasattr(signature, "structured_input_signature"):
                    input_spec = signature.structured_input_signature[1]  # kwargs部分
                    if "input" in input_spec:
                        expected_dtype = input_spec["input"].dtype
                    elif input_spec:
                        # 最初の引数の型を確認
                        expected_dtype = list(input_spec.values())[0].dtype
            except (IndexError, AttributeError, KeyError):
                pass

            # 適切な型でテンソルを作成
            # エラーログから、このsignatureはint32を要求している
            # 型が確認できない場合は、int32を試行（MoveNet v4はint32を要求）
            if expected_dtype == tf.int32 or expected_dtype is None:
                # int32が要求される場合（0-255の範囲）
                image_batch_int = (image_batch * 255.0).astype(np.uint8)
                input_tensor = tf.constant(image_batch_int, dtype=tf.int32)
            else:
                # float32で試行
                input_tensor = tf.constant(image_batch, dtype=tf.float32)

            # キーワード引数で呼び出しを試行
            try:
                outputs = signature(input=input_tensor)
            except (TypeError, ValueError) as e:
                # 位置引数でも試行
                try:
                    outputs = signature(input_tensor)
                except Exception:
                    # 型が間違っている場合は、もう一度int32で試行
                    if expected_dtype != tf.int32:
                        image_batch_int = (image_batch * 255.0).astype(np.uint8)
                        input_tensor = tf.constant(image_batch_int, dtype=tf.int32)
                        outputs = signature(input=input_tensor)
                    else:
                        raise
        else:
            raise RuntimeError("モデルを呼び出すことができません")

        # 出力の処理
        if isinstance(outputs, dict):
            # 辞書の場合は最初のキーを使用
            output_key = list(outputs.keys())[0]
            keypoints_raw = outputs[output_key].numpy()
        else:
            keypoints_raw = outputs.numpy()

        # MoveNetの出力形状は (1, 1, 17, 3) または (1, 17, 3)
        # 最初の次元を削除して (17, 3) にする
        if len(keypoints_raw.shape) == 4:
            keypoints = keypoints_raw[0][0]  # (17, 3)
        elif len(keypoints_raw.shape) == 3:
            keypoints = keypoints_raw[0]  # (17, 3)
        else:
            keypoints = keypoints_raw  # 既に (17, 3) またはそれ以外

        # 元の画像サイズにスケール変換
        height, width = original_shape
        scale_x = width / self.input_size
        scale_y = height / self.input_size

        # キーポイントを元の画像サイズに変換
        keypoints_scaled = []
        for kp in keypoints:
            # kpは (y, x, confidence) の形状
            # 各要素をスカラー値に変換
            y_val = kp[0]
            x_val = kp[1]
            conf_val = kp[2]

            # numpy配列の場合は適切にスカラー値に変換
            # サイズ1の配列の場合はitem()、それ以外は最初の要素を取得
            if isinstance(y_val, np.ndarray):
                if y_val.size == 1:
                    y_val = float(y_val.item())
                else:
                    y_val = float(y_val.flat[0])
            else:
                y_val = float(y_val)

            if isinstance(x_val, np.ndarray):
                if x_val.size == 1:
                    x_val = float(x_val.item())
                else:
                    x_val = float(x_val.flat[0])
            else:
                x_val = float(x_val)

            if isinstance(conf_val, np.ndarray):
                if conf_val.size == 1:
                    conf_val = float(conf_val.item())
                else:
                    conf_val = float(conf_val.flat[0])
            else:
                conf_val = float(conf_val)

            x = x_val * scale_x
            y = y_val * scale_y

            keypoints_scaled.append(
                {
                    "x": x / width,  # 正規化座標（0-1）
                    "y": y / height,
                    "confidence": conf_val,
                }
            )

        return {"keypoints": keypoints_scaled}, image

    def get_landmark(self, results, landmark_id):
        """
        特定のランドマークを取得

        Args:
            results: 検出結果
            landmark_id: ランドマークID

        Returns:
            landmark: ランドマーク情報（x, y, confidence）またはNone
        """
        if "keypoints" in results and landmark_id < len(results["keypoints"]):
            kp = results["keypoints"][landmark_id]
            if kp["confidence"] > 0.3:
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
        if "keypoints" in results:
            for idx, kp in enumerate(results["keypoints"]):
                landmarks_dict[idx] = {
                    "x": kp["x"],
                    "y": kp["y"],
                    "confidence": kp["confidence"],
                    "visibility": kp[
                        "confidence"
                    ],  # MoveNetはvisibilityの代わりにconfidenceを使用
                }
        return landmarks_dict
