"""
姿勢検出モジュール
MediaPipe Poseを使用して骨格キーポイントを検出
"""
import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    """MediaPipe Poseを使用した姿勢検出クラス"""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初期化
        
        Args:
            min_detection_confidence: 検出の最小信頼度
            min_tracking_confidence: トラッキングの最小信頼度
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect(self, image):
        """
        画像から姿勢を検出
        
        Args:
            image: BGR画像（OpenCV形式）
            
        Returns:
            results: MediaPipe Poseの検出結果
            annotated_image: 骨格が描画された画像
        """
        # RGBに変換（MediaPipeはRGBを期待）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 姿勢検出
        results = self.pose.process(image_rgb)
        
        # 骨格を描画
        annotated_image = image.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        return results, annotated_image
    
    def get_landmark(self, results, landmark_id):
        """
        特定のランドマークを取得
        
        Args:
            results: MediaPipe Poseの検出結果
            landmark_id: ランドマークID（MediaPipe Pose定数）
            
        Returns:
            landmark: ランドマーク座標（x, y, z）またはNone
        """
        if results.pose_landmarks:
            return results.pose_landmarks.landmark[landmark_id]
        return None
    
    def get_landmarks_dict(self, results):
        """
        すべてのランドマークを辞書形式で取得
        
        Args:
            results: MediaPipe Poseの検出結果
            
        Returns:
            landmarks_dict: ランドマークIDをキーとする辞書
        """
        landmarks_dict = {}
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_dict[idx] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
        return landmarks_dict

