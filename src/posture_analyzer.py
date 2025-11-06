"""
猫背判定モジュール
骨格キーポイントから猫背を判定
"""
import math
import mediapipe as mp


class PostureAnalyzer:
    """猫背判定クラス"""
    
    # MediaPipe PoseのランドマークID
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    def __init__(self, threshold_angle=35.0):
        """
        初期化
        
        Args:
            threshold_angle: 猫背と判定する角度の閾値（度）
        """
        self.threshold_angle = threshold_angle
        self.mp_pose = mp.solutions.pose
    
    def calculate_angle(self, point1, point2, point3):
        """
        3点から角度を計算（point2を頂点とする角度）
        
        Args:
            point1: 点1の座標 (x, y)
            point2: 点2の座標（頂点）(x, y)
            point3: 点3の座標 (x, y)
            
        Returns:
            angle: 角度（度）
        """
        # ベクトルを計算
        vec1 = (point1[0] - point2[0], point1[1] - point2[1])
        vec2 = (point3[0] - point2[0], point3[1] - point2[1])
        
        # 内積とノルムを計算
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        norm1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
        norm2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
        
        if norm1 == 0 or norm2 == 0:
            return None
        
        # 角度を計算（ラジアンから度に変換）
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # 範囲を制限
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def analyze_posture(self, landmarks_dict):
        """
        姿勢を分析して猫背を判定
        
        Args:
            landmarks_dict: ランドマークの辞書
            
        Returns:
            result: 判定結果の辞書
                - is_slouched: 猫背かどうか（bool）
                - confidence: 信頼度（0.0-1.0）
                - angle: 計算された角度（度）
                - message: メッセージ
        """
        if not landmarks_dict:
            return {
                'is_slouched': False,
                'confidence': 0.0,
                'angle': None,
                'message': '姿勢を検出できませんでした'
            }
        
        # 必要なランドマークを取得
        left_shoulder = landmarks_dict.get(self.LEFT_SHOULDER)
        right_shoulder = landmarks_dict.get(self.RIGHT_SHOULDER)
        left_hip = landmarks_dict.get(self.LEFT_HIP)
        right_hip = landmarks_dict.get(self.RIGHT_HIP)
        nose = landmarks_dict.get(self.NOSE)
        
        # 可視性チェック
        required_landmarks = [
            (left_shoulder, '左肩'),
            (right_shoulder, '右肩'),
            (left_hip, '左腰'),
            (right_hip, '右腰')
        ]
        
        missing = [name for landmark, name in required_landmarks if not landmark or landmark.get('visibility', 0) < 0.5]
        if missing:
            return {
                'is_slouched': False,
                'confidence': 0.0,
                'angle': None,
                'message': f'検出不足: {", ".join(missing)}'
            }
        
        # 肩と腰の中点を計算
        shoulder_mid = (
            (left_shoulder['x'] + right_shoulder['x']) / 2,
            (left_shoulder['y'] + right_shoulder['y']) / 2
        )
        hip_mid = (
            (left_hip['x'] + right_hip['x']) / 2,
            (left_hip['y'] + right_hip['y']) / 2
        )
        
        # 首の位置を推定（鼻と肩の中点を使用）
        if nose and nose.get('visibility', 0) > 0.5:
            neck_estimate = (
                (nose['x'] + shoulder_mid[0]) / 2,
                (nose['y'] + shoulder_mid[1]) / 2
            )
        else:
            # 鼻が検出できない場合は肩の中点を少し上にずらす
            neck_estimate = (
                shoulder_mid[0],
                shoulder_mid[1] - 0.05
            )
        
        # 猫背角度を計算
        # 首-肩-腰の角度を計算（角度が小さいほど猫背）
        angle = self.calculate_angle(
            neck_estimate,
            shoulder_mid,
            hip_mid
        )
        
        if angle is None:
            return {
                'is_slouched': False,
                'confidence': 0.0,
                'angle': None,
                'message': '角度を計算できませんでした'
            }
        
        # 猫背判定
        # 角度が閾値より小さい場合、猫背と判定
        # 正常な姿勢では首-肩-腰の角度は大きい（150度以上）
        # 猫背では角度が小さくなる（120度以下）
        is_slouched = angle < (180 - self.threshold_angle)
        
        # 信頼度を計算（角度に基づく）
        if is_slouched:
            # 猫背の場合、角度が小さいほど信頼度が高い
            confidence = min(1.0, (180 - self.threshold_angle - angle) / 30.0)
        else:
            # 正常な場合、角度が大きいほど信頼度が高い
            confidence = min(1.0, (angle - (180 - self.threshold_angle)) / 30.0)
        
        # メッセージを生成
        if is_slouched:
            message = f'猫背です。姿勢を直してください！ (角度: {angle:.1f}度)'
        else:
            message = f'姿勢良好です (角度: {angle:.1f}度)'
        
        return {
            'is_slouched': is_slouched,
            'confidence': confidence,
            'angle': angle,
            'message': message
        }

