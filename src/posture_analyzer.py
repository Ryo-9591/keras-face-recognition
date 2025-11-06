"""
猫背判定モジュール
骨格キーポイントから猫背を判定
"""

import math


class PostureAnalyzer:
    """猫背判定クラス"""

    # MoveNetのランドマークID
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_HIP = 11
    RIGHT_HIP = 12

    def __init__(self, threshold_angle=35.0):
        """
        初期化

        Args:
            threshold_angle: 猫背と判定する角度の閾値（度）
        """
        self.threshold_angle = threshold_angle

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
        norm1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        norm2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

        if norm1 == 0 or norm2 == 0:
            return None

        # 角度を計算（ラジアンから度に変換）
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # 範囲を制限
        angle = math.degrees(math.acos(cos_angle))

        return angle

    def analyze_posture(self, landmarks_dict):
        """
        姿勢を分析して猫背を判定（座っている状態の前からのカメラ用）

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
                "is_slouched": False,
                "confidence": 0.0,
                "angle": None,
                "message": "姿勢を検出できませんでした",
            }

        # 必要なランドマークを取得
        left_shoulder = landmarks_dict.get(self.LEFT_SHOULDER)
        right_shoulder = landmarks_dict.get(self.RIGHT_SHOULDER)
        left_hip = landmarks_dict.get(self.LEFT_HIP)
        right_hip = landmarks_dict.get(self.RIGHT_HIP)
        nose = landmarks_dict.get(self.NOSE)

        # 肩の検出チェック（必須）
        if (
            not left_shoulder
            or left_shoulder.get("confidence", 0) < 0.3
            or not right_shoulder
            or right_shoulder.get("confidence", 0) < 0.3
        ):
            return {
                "is_slouched": False,
                "confidence": 0.0,
                "angle": None,
                "message": "肩を検出できませんでした",
            }

        # 肩の中点を計算
        shoulder_mid = (
            (left_shoulder["x"] + right_shoulder["x"]) / 2,
            (left_shoulder["y"] + right_shoulder["y"]) / 2,
        )

        # 首の位置を推定（鼻と肩の中点を使用）
        if nose and nose.get("confidence", 0) > 0.3:
            neck_estimate = (
                (nose["x"] + shoulder_mid[0]) / 2,
                (nose["y"] + shoulder_mid[1]) / 2,
            )
        else:
            # 鼻が検出できない場合は肩の中点を少し上にずらす
            neck_estimate = (shoulder_mid[0], shoulder_mid[1] - 0.05)

        # 腰が検出できる場合は、首-肩-腰の角度で判定（従来の方法）
        if (
            left_hip
            and left_hip.get("confidence", 0) >= 0.3
            and right_hip
            and right_hip.get("confidence", 0) >= 0.3
        ):
            hip_mid = (
                (left_hip["x"] + right_hip["x"]) / 2,
                (left_hip["y"] + right_hip["y"]) / 2,
            )

            # 首-肩-腰の角度を計算
            angle = self.calculate_angle(neck_estimate, shoulder_mid, hip_mid)

            if angle is not None:
                # 座っている状態では、角度が140度未満の場合に猫背と判定
                # 正常な姿勢では首-肩-腰の角度は大きい（150度以上）
                # 猫背では角度が小さくなる（130度以下）
                is_slouched = angle < (180 - self.threshold_angle)

                # 信頼度を計算（角度に基づく）
                if is_slouched:
                    confidence = min(1.0, (180 - self.threshold_angle - angle) / 30.0)
                else:
                    confidence = min(1.0, (angle - (180 - self.threshold_angle)) / 30.0)

                if is_slouched:
                    message = f"猫背です。姿勢を直してください！ (角度: {angle:.1f}度)"
                else:
                    message = f"姿勢良好です (角度: {angle:.1f}度)"

                return {
                    "is_slouched": is_slouched,
                    "confidence": confidence,
                    "angle": angle,
                    "message": message,
                }

        # 腰が検出できない場合（座っている状態で前からのカメラの場合が多い）
        # 首と肩の位置関係で判定
        # 首の位置が肩より前に出ている（y座標が小さい、つまり上に来ている）場合、猫背と判定
        # または、首-肩の線と水平線の角度を計算

        # 首と肩の位置関係から角度を計算
        # 水平線（右方向）と首-肩のベクトルの角度を計算
        neck_to_shoulder_vec = (
            shoulder_mid[0] - neck_estimate[0],
            shoulder_mid[1] - neck_estimate[1],
        )

        # 水平線との角度を計算（座っている状態では、首が前に出ていると角度が大きくなる）
        if neck_to_shoulder_vec[0] == 0:
            # 垂直な場合は90度
            angle = 90.0
        else:
            # atan2で角度を計算（度に変換）
            angle_rad = math.atan2(
                abs(neck_to_shoulder_vec[1]), abs(neck_to_shoulder_vec[0])
            )
            angle = math.degrees(angle_rad)

        # 座っている状態での猫背判定
        # 首が肩より前に出ている（y座標が小さい）場合、猫背の可能性が高い
        # 首と肩のy座標の差が大きいほど猫背
        y_diff = neck_estimate[1] - shoulder_mid[1]  # 首が上（yが小さい）ほど負の値

        # 首が肩より前に出ている（y座標が小さい）場合、猫背と判定
        # 座っている状態では、正常な姿勢でも首が少し上にあるため、閾値を調整
        # 閾値: 首が肩より0.015以上上にある場合（正規化座標）
        is_slouched = y_diff < -0.015

        # 角度も考慮（首-肩の線が垂直に近いほど猫背）
        # 角度が75度以上の場合も猫背の可能性が高い
        if angle > 75:
            is_slouched = True

        # 信頼度を計算
        if is_slouched:
            # 首が肩より前に出ているほど信頼度が高い
            confidence = min(1.0, abs(y_diff) / 0.08)
            # 角度も考慮
            if angle > 75:
                confidence = max(confidence, min(1.0, (angle - 75) / 15.0))
        else:
            # 正常な場合、首と肩の位置が近いほど信頼度が高い
            confidence = min(1.0, (0.015 + y_diff) / 0.03)

        # メッセージを生成（位置差をパーセンテージで表示）
        position_diff_pct = abs(y_diff) * 100
        if is_slouched:
            message = f"猫背です。姿勢を直してください！ (首-肩位置差: {position_diff_pct:.1f}%)"
        else:
            message = f"姿勢良好です (首-肩位置差: {position_diff_pct:.1f}%)"

        return {
            "is_slouched": is_slouched,
            "confidence": confidence,
            "angle": angle,
            "message": message,
        }
