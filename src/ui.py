"""
UI表示モジュール
判定結果と骨格図を画面に表示
"""
import cv2
import numpy as np


class UI:
    """UI表示クラス"""
    
    def __init__(self, window_name="姿勢検出・猫背判定"):
        """
        初期化
        
        Args:
            window_name: ウィンドウ名
        """
        self.window_name = window_name
        self.alert_count = 0  # 連続アラート回数
    
    def draw_result(self, image, posture_result, fps=None):
        """
        判定結果を画像に描画
        
        Args:
            image: 画像（BGR形式）
            posture_result: 姿勢判定結果の辞書
            fps: FPS（オプション）
            
        Returns:
            annotated_image: 描画済み画像
        """
        annotated_image = image.copy()
        height, width = annotated_image.shape[:2]
        
        # 背景パネルを作成（半透明）
        overlay = annotated_image.copy()
        
        # ステータスパネル
        panel_height = 120
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)
        
        # 判定結果に応じた色を設定
        if posture_result.get('is_slouched', False):
            color = (0, 0, 255)  # 赤
            self.alert_count += 1
        else:
            color = (0, 255, 0)  # 緑
            self.alert_count = 0
        
        # メッセージを表示
        message = posture_result.get('message', '検出中...')
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # メインメッセージ
        text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 40
        cv2.putText(annotated_image, message, (text_x, text_y), 
                   font, font_scale, color, thickness)
        
        # 角度情報
        angle = posture_result.get('angle')
        if angle is not None:
            angle_text = f"角度: {angle:.1f}度"
            angle_size = cv2.getTextSize(angle_text, font, 0.6, 1)[0]
            angle_x = (width - angle_size[0]) // 2
            angle_y = 70
            cv2.putText(annotated_image, angle_text, (angle_x, angle_y), 
                       font, 0.6, (255, 255, 255), 1)
        
        # 信頼度
        confidence = posture_result.get('confidence', 0.0)
        confidence_text = f"信頼度: {confidence*100:.1f}%"
        confidence_size = cv2.getTextSize(confidence_text, font, 0.5, 1)[0]
        confidence_x = (width - confidence_size[0]) // 2
        confidence_y = 95
        cv2.putText(annotated_image, confidence_text, (confidence_x, confidence_y), 
                   font, 0.5, (200, 200, 200), 1)
        
        # FPS表示
        if fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated_image, fps_text, (10, height - 10), 
                       font, 0.5, (255, 255, 255), 1)
        
        # 猫背の場合は警告バーを表示
        if posture_result.get('is_slouched', False):
            # 上部に警告バー
            cv2.rectangle(annotated_image, (0, 0), (width, 10), color, -1)
            # 下部にも警告バー
            cv2.rectangle(annotated_image, (0, height - 10), (width, height), color, -1)
        
        return annotated_image
    
    def show(self, image):
        """
        画像を表示
        
        Args:
            image: 表示する画像
        """
        cv2.imshow(self.window_name, image)
    
    def wait_key(self, delay=1):
        """
        キー入力を待つ
        
        Args:
            delay: 待機時間（ミリ秒）
            
        Returns:
            key: 押されたキーのコード
        """
        return cv2.waitKey(delay) & 0xFF
    
    def destroy_windows(self):
        """すべてのウィンドウを閉じる"""
        cv2.destroyAllWindows()

