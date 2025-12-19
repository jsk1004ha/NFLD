import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math

# --- 1. 경량화 모듈 (Ghost Module & CBAM) ---
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        scale = self.sigmoid(self.conv(torch.cat([torch.mean(out, 1, keepdim=True), torch.max(out, 1, keepdim=True)[0]], dim=1)))
        return out * scale

# --- 2. EcoLineNet (메인 네트워크) ---
class EcoLineNet(nn.Module):
    def __init__(self, num_classes=2):
        super(EcoLineNet, self).__init__()
        self.learning_to_downsample = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            GhostModule(32, 48, stride=2),
            GhostModule(48, 64, stride=2)
        )
        self.global_feature_extractor = nn.Sequential(
            GhostModule(64, 64, stride=2),
            GhostModule(64, 96, stride=1),
            GhostModule(96, 128, stride=1),
            CBAM(128)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv2d(64, num_classes, 1)
        )
    def forward(self, x):
        input_size = x.size()[2:]
        higher_res = self.learning_to_downsample(x)
        lower_res = self.global_feature_extractor(higher_res)
        fused = F.interpolate(lower_res, size=higher_res.size()[2:], mode='bilinear', align_corners=True)
        return F.interpolate(self.classifier(fused), size=input_size, mode='bilinear', align_corners=True)

# --- 3. EcoLineTracker (추론 & 필터링 엔진) ---
class EcoLineTracker:
    def __init__(self, model_path, input_size=(320, 320), device='cuda', confidence=0.5):
        self.device = device
        self.input_size = input_size
        self.model = EcoLineNet(num_classes=2).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.confidence = confidence

    def preprocess(self, img):
        h, w = img.shape[:2]
        # [강력 필터 1] 상단 40% 강제 마스킹 (하늘/건물 삭제)
        img_masked = img.copy()
        roi_limit = int(h * 0.4)
        img_masked[0:roi_limit, :] = 0
        
        img_resized = cv2.resize(img_masked, self.input_size)
        return torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

    def extract_waypoints(self, mask, num_windows=10):
        h, w = mask.shape
        window_height = int(h / num_windows)
        points = []
        for i in range(num_windows):
            y_start, y_end = h - (i + 1) * window_height, h - i * window_height
            roi = mask[y_start:y_end, :]
            white_pixels = np.where(roi >= 1)
            if len(white_pixels[0]) > 10:
                center_x = int(np.mean(white_pixels[1]))
                center_y = int((y_start + y_end) / 2)
                points.append((center_x, center_y))
        return points

    def calculate_curvature(self, points, width):
        if len(points) < 3: return 0.0, 0.0, None
        pts_x = np.array([p[0] for p in points])
        pts_y = np.array([p[1] for p in points])
        try:
            fit = np.polyfit(pts_y, pts_x, 2) # x = ay^2 + by + c
            offset = (fit[0]*np.max(pts_y)**2 + fit[1]*np.max(pts_y) + fit[2]) - (width / 2)
            return fit[0] * 1000, offset, fit
        except: return 0.0, 0.0, None

    def predict(self, frame):
        h_orig, w_orig = frame.shape[:2]
        input_tensor = self.preprocess(frame)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = F.softmax(output, dim=1)[0, 1].cpu().numpy()
            pred_mask = (prob > self.confidence).astype(np.uint8)

        # [강력 필터 2] 바닥 연결성 검사 (하늘에 뜬 덩어리 제거)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
        cleaned_mask = np.zeros_like(pred_mask)
        for i in range(1, num_labels):
            bottom_y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
            # 덩어리의 바닥이 화면 하단 20% 영역 안에 들어와야 유효
            if bottom_y > pred_mask.shape[0] * 0.8:
                cleaned_mask[labels == i] = 1

        mask_final = cv2.resize(cleaned_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        points = self.extract_waypoints(mask_final)
        curve, offset, fit = self.calculate_curvature(points, w_orig)

        return mask_final, points, curve, offset