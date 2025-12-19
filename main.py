import cv2
import argparse
import torch
import numpy as np
from src.model import EcoLineTracker

def run(color_mode, video_source):
    model_path = f"models/nfld_{color_mode}.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 실행 중... [{color_mode.upper()}]")

    try:
        tracker = EcoLineTracker(model_path, device=device)
    except:
        print("❌ 모델 파일 없음! python train.py를 먼저 실행하세요.")
        return

    cap = cv2.VideoCapture(video_source)
    line_col = (0, 255, 0) if color_mode == 'white' else (255, 0, 0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 추론 수행 (강력 필터링 포함)
        mask, points, curvature, offset = tracker.predict(frame)

        # 시각화
        overlay = np.zeros_like(frame)
        overlay[mask == 1] = line_col
        result = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        # 점과 경로 그리기
        if points:
            for pt in points: cv2.circle(result, pt, 5, (0, 0, 255), -1)
            cv2.polylines(result, [np.array(points, np.int32).reshape((-1, 1, 2))], False, (0, 255, 255), 2)

        # 제어 정보 표시
        direction = "STRAIGHT"
        if offset < -30: direction = "Turn LEFT  <<"
        elif offset > 30: direction = ">>  Turn RIGHT"
        
        cv2.putText(result, direction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(result, f"Off: {offset:.1f} | Curve: {curvature:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow('NFLD Final', result)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", type=str, required=True, choices=["white", "yellow"])
    # ▼▼▼ 여기에 비디오 경로를 넣으세요 ▼▼▼
    parser.add_argument("--video", type=str, default="test_video.mp4") 
    args = parser.parse_args()
    
    src = int(args.video) if args.video.isdigit() else args.video
    run(args.color, src)