"""
カメラ処理モジュール
PCカメラからの映像取得を管理
"""

import cv2
import os
import platform


class CameraHandler:
    """カメラ処理クラス"""

    def __init__(self, camera_device=None, width=640, height=480):
        """
        初期化

        Args:
            camera_device: カメラデバイスのパスまたはインデックス（Noneの場合は自動検出）
            width: 画像の幅
            height: 画像の高さ
        """
        self.camera_device = camera_device
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        """カメラを開く"""
        system = platform.system()

        # カメラデバイスが指定されていない場合、またはデバイスパスが存在しない場合はインデックス0を使用
        if self.camera_device is None or self.camera_device == "":
            # カメラインデックス0で開く
            print(f"カメラを開いています: インデックス 0 (プラットフォーム: {system})")
            # Windows環境ではDirectShowバックエンドを試行
            if system == "Windows" and hasattr(cv2, "CAP_DSHOW"):
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(0)
        elif isinstance(self.camera_device, str) and self.camera_device.startswith(
            "/dev/"
        ):
            # デバイスパスが存在するかチェック
            if os.path.exists(self.camera_device):
                print(f"カメラを開いています: デバイスパス {self.camera_device}")
                self.cap = cv2.VideoCapture(self.camera_device)
            else:
                # デバイスパスが存在しない場合はインデックス0で開く
                print(
                    f"デバイスパス {self.camera_device} が存在しないため、インデックス 0 で開きます"
                )
                if system == "Windows" and hasattr(cv2, "CAP_DSHOW"):
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(0)
        else:
            # 数値の場合はインデックスとして使用
            try:
                index = int(self.camera_device)
                print(
                    f"カメラを開いています: インデックス {index} (プラットフォーム: {system})"
                )
                if system == "Windows" and hasattr(cv2, "CAP_DSHOW"):
                    self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(index)
            except (ValueError, TypeError):
                # 文字列の場合はデバイスパスとして使用
                print(f"カメラを開いています: {self.camera_device}")
                self.cap = cv2.VideoCapture(self.camera_device)

        if self.cap is None or not self.cap.isOpened():
            print("警告: カメラを開けませんでした")
            print(
                "Windows環境でDockerコンテナからカメラにアクセスするには、WSL2バックエンドの使用を推奨します"
            )
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print(f"カメラを開きました: {self.width}x{self.height}")
        return True

    def read(self):
        """
        フレームを読み込む

        Returns:
            success: 成功したかどうか
            frame: 画像フレーム（BGR形式）
        """
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """カメラを解放"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self):
        """
        カメラが開いているかどうか

        Returns:
            bool: カメラが開いているかどうか
        """
        return self.cap is not None and self.cap.isOpened()
