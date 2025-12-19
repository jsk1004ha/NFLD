import cv2
import os

def extract(video_path):
    output_dir = "data/backgrounds"
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    print(f"🎞️ 배경 추출 중... ({video_path})")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % 10 == 0: # 10프레임마다 저장
            cv2.imwrite(f"{output_dir}/bg_{count:05d}.jpg", cv2.resize(frame, (320, 320)))
        count += 1
    print("✅ 배경 추출 완료.")

if __name__ == "__main__":
    # ▼▼▼ 실제 바닥만 찍은 영상을 여기에 입력 ▼▼▼
    video_file = "data/train_data.mp4" 
    if os.path.exists(video_file): extract(video_file)
    else: print("❌ 영상 파일이 없습니다. (무시하고 진행하면 노이즈 배경 사용됨)")