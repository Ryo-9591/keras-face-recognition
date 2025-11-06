"""
姿勢検出・猫背判定アプリ
メインアプリケーション
"""
import cv2
import time
import sys
from camera_handler import CameraHandler
from pose_detector import PoseDetector
from posture_analyzer import PostureAnalyzer
from ui import UI


def main():
    """メイン関数"""
    print("姿勢検出・猫背判定アプリを起動しています...")
    
    # モジュールの初期化
    camera = CameraHandler(camera_index=0, width=640, height=480)
    pose_detector = PoseDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    posture_analyzer = PostureAnalyzer(threshold_angle=35.0)
    ui = UI(window_name="姿勢検出・猫背判定")
    
    # カメラを開く
    if not camera.open():
        print("エラー: カメラを開けませんでした。")
        print("カメラが接続されているか、他のアプリケーションが使用していないか確認してください。")
        sys.exit(1)
    
    print("カメラを開きました。")
    print("操作:")
    print("  - 'q' キー: 終了")
    print("  - ESC キー: 終了")
    
    # FPS計算用
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    try:
        while True:
            # フレームを読み込む
            ret, frame = camera.read()
            if not ret:
                print("警告: フレームを読み込めませんでした。")
                break
            
            # 姿勢検出
            results, annotated_frame = pose_detector.detect(frame)
            
            # ランドマークを取得
            landmarks_dict = pose_detector.get_landmarks_dict(results)
            
            # 猫背判定
            posture_result = posture_analyzer.analyze_posture(landmarks_dict)
            
            # FPS計算
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - fps_start_time
                current_fps = 30 / elapsed
                fps_start_time = time.time()
            
            # UIに描画
            display_frame = ui.draw_result(annotated_frame, posture_result, current_fps)
            
            # 表示
            ui.show(display_frame)
            
            # 猫背の場合はコンソールにも出力
            if posture_result.get('is_slouched', False):
                print(f"[警告] {posture_result.get('message', '')}")
            
            # キー入力チェック
            key = ui.wait_key(1)
            if key == ord('q') or key == 27:  # 'q' または ESC
                print("終了します...")
                break
    
    except KeyboardInterrupt:
        print("\n中断されました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # クリーンアップ
        camera.release()
        ui.destroy_windows()
        print("アプリケーションを終了しました。")


if __name__ == "__main__":
    main()

