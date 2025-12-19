import cv2
import numpy as np
import os
import random
import argparse
from tqdm import tqdm
import albumentations as A

def get_perspective_matrix(w, h):
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    gap_x = random.randint(50, 100)
    gap_y = random.randint(0, 50)
    dst = np.float32([[gap_x, gap_y], [w - gap_x, gap_y], [0, h], [w, h]])
    return cv2.getPerspectiveTransform(src, dst)

def add_distractors(img):
    """모델을 속이기 위한 가짜 흰색 물체 추가 (건물, 구름 등)"""
    h, w = img.shape[:2]
    for _ in range(random.randint(1, 4)):
        color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        if random.random() > 0.5: # Rect
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (pt1[0] + random.randint(20, 80), pt1[1] + random.randint(20, 80))
            cv2.rectangle(img, pt1, pt2, color, -1)
        else: # Circle
            center = (random.randint(0, w), random.randint(0, h))
            cv2.circle(img, center, random.randint(10, 40), color, -1)
    return img

def generate_realistic(color_name, num_samples):
    bg_dir = "data/backgrounds"
    save_path = f"data/{color_name}_real"
    os.makedirs(f"{save_path}/images", exist_ok=True)
    os.makedirs(f"{save_path}/masks", exist_ok=True)
    
    bg_files = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith('.jpg')] if os.path.exists(bg_dir) else []
    base_color = (220, 220, 220) if color_name == 'white' else (0, 200, 220)
    
    augmentor = A.Compose([
        A.GaussNoise(p=0.5), A.MotionBlur(blur_limit=7, p=0.3),
        A.RandomBrightnessContrast(p=0.5),
    ])

    print(f"🚀 실전 데이터셋 생성 중 (함정 데이터 포함)... [{color_name}]")
    for i in tqdm(range(num_samples)):
        bg = cv2.imread(random.choice(bg_files)) if bg_files else np.full((320, 320, 3), 100, dtype=np.uint8)
        bg = cv2.resize(bg, (320, 320))
        
        mask_layer = np.zeros((320, 320), dtype=np.uint8)
        
        # 20% 확률로 선이 없는 '함정 데이터' 생성 (Distractors Only)
        if random.random() < 0.2:
            bg = add_distractors(bg)
            # 마스크는 0 유지
        else:
            # 정상 라인 데이터
            line_layer = np.zeros_like(bg)
            pt1, pt2 = (random.randint(-50, 370), random.randint(-50, 370)), (random.randint(-50, 370), random.randint(-50, 370))
            cv2.line(line_layer, pt1, pt2, base_color, random.randint(8, 20))
            cv2.line(mask_layer, pt1, pt2, 255, random.randint(8, 20))
            
            M = get_perspective_matrix(320, 320)
            line_layer = cv2.warpPerspective(line_layer, M, (320, 320))
            mask_layer = cv2.warpPerspective(mask_layer, M, (320, 320))
            
            bg[mask_layer > 0] = line_layer[mask_layer > 0]
            if random.random() < 0.3: bg = add_distractors(bg) # 라인 배경에도 방해물 추가

        augmented = augmentor(image=bg, mask=mask_layer)
        cv2.imwrite(f"{save_path}/images/real_{i:05d}.jpg", augmented['image'])
        cv2.imwrite(f"{save_path}/masks/real_{i:05d}.png", augmented['mask'])
    print("✅ 생성 완료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", type=str, default="white", choices=["white", "yellow"])
    parser.add_argument("--count", type=int, default=1500)
    args = parser.parse_args()
    generate_realistic(args.color, args.count)