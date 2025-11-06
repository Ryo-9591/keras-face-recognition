"""
カメラ処理モジュール
PCカメラからの映像取得を管理
"""
import cv2


class CameraHandler:
    """カメラ処理クラス"""
    
    def __init__(self, camera_index=0, width=640, height=480):
        """
        初期化
        
        Args:
            camera_index: カメラのインデックス（通常は0）
            width: 画像の幅
            height: 画像の高さ
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
    
    def open(self):
        """
        カメラを開く
        
        Returns:
            success: 成功したかどうか
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False
            
            # 解像度を設定
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            return True
        except Exception as e:
            print(f"カメラを開けませんでした: {e}")
            return False
    
    def read(self):
        """
        フレームを読み込む
        
        Returns:
            success: 成功したかどうか
            frame: 画像フレーム（BGR形式）
        """
        if self.cap is None:
            return False, None
        
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

