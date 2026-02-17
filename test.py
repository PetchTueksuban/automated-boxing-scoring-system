# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import time, sys, os, csv
from collections import deque

# ====== YOLO Pose ======
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError("ติดตั้ง ultralytics ก่อน: pip install ultralytics") from e

# ===================== Global Config =====================
VIDEO_PATH_MAIN = "t2.mp4"
VIDEO_PATH_ALT  = "t1.mp4"
POSE_WEIGHTS = "yolov8s-pose.pt"
POSE_CONF    = 0.35
IMG_SIZE     = 640

# ความเร็วเล่น (1.0 ปกติ)
SLOW_FACTOR = 0.2

# ฟิวชันคะแนนสี (ทับซ้อนกับ mask + สัดส่วนสีใน bbox)
W_COV, W_COL = 0.6, 0.4
MIN_SCORE    = 0.05
COVER_DILATE = 15

# สีวาด (BGR)
COLOR_RED    = (0, 0, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (40, 40, 40)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN  = (0, 220, 0)
COLOR_RED2   = (0, 0, 255)
COLOR_CYAN   = (255, 255, 0)
COLOR_MAG    = (255, 0, 255)

# Scale px->cm และเกณฑ์
PX_PER_CM         = 3.8   #20
PUNCH_RANGE_CM    = 2.4
CLINCH_RANGE_CM   = 5.0
CLINCH_HYS_CM     = 10.0

# เวลาหน่วงหลังสลับมุมมอง (มิลลิวินาที)
SWITCH_COOLDOWN_MS = 1000

# หยุดนับหลังสลับมุม (มิลลิวินาที)
SCORE_BLOCK_AFTER_SWITCH_MS = 1000

# >>> NEW: นับหมัด "รอดึงออก"
# ต้อง "ค้างในระยะ" อย่างน้อยกี่ ms ก่อนจะนับ (กันเฟรมกระพริบ)
PUNCH_ENTER_HOLD_MS   = 100
# ต้อง "ดึงออก" เกินระยะออกอีกกี่ cm จึงถือว่ารีเซ็ต (ฮิสเทอรีซิสขาออก)
PUNCH_EXIT_ADD_CM     = 3

# จะบล็อกการนับตอนคลินช์ไหม
BLOCK_SCORING_DURING_CLINCH = True

# Morphology
MORPH_KSIZE  = 5

# ===== ค่าสีรวมศูนย์ (ตามโค้ดตัวอย่าง) =====
RED_RANGES = [
    ((0,   150, 100), (10,  255, 255)),
    ((170, 150, 100), (179, 255, 255)),
]
BLACK_RANGE = ((0, 0, 0), (180, 220, 45))  # V<=45

# Skeleton (COCO-17)
SKELETON_PAIRS = [
    (5,7),(7,9),(6,8),(8,10),
    (11,13),(13,15),(12,14),(14,16),
    (5,6),(11,12),(5,11),(6,12),
    (0,5),(0,6)
]

# Auto-switch when distance lost
MISSING_MAX_FRAMES = 8

# ================= Smooth Auto Brightness =================
class SmoothAutoBrightness:
    def __init__(self, target_luma=0.62, ema_alpha=0.10, max_step=0.06,
                 gain_min=0.65, gain_max=1.75, clahe_clip=1.4, clahe_tile=(8,8)):
        self.target = float(target_luma)
        self.alpha = float(ema_alpha)
        self.max_step = float(max_step)
        self.gain_min, self.gain_max = float(gain_min), float(gain_max)
        self._clahe = cv.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
        self.ema_luma = None
        self.gain = 1.0
        self.last_luma = 0.0
    def _estimate_luma(self, bgr):
        y = 0.114*bgr[...,0] + 0.587*bgr[...,1] + 0.299*bgr[...,2]
        lo = np.percentile(y, 5); hi = np.percentile(y, 95)
        y_clip = np.clip(y, lo, hi)
        luma = float(np.mean((y_clip - lo) / max(1e-6, (hi - lo))))
        return max(0.0, min(1.0, luma))
    def _apply_gain_safe(self, bgr, gain):
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
        L, A, B = cv.split(lab)
        Lf = L.astype(np.float32)/255.0
        gamma = 0.90 if gain > 1.0 else 1.10
        Lf = np.power(Lf, gamma) * gain
        Lf = np.clip(Lf, 0.0, 1.0)
        Lc = (Lf*255.0).astype(np.uint8)
        Lc = self._clahe.apply(Lc)
        return cv.cvtColor(cv.merge((Lc, A, B)), cv.COLOR_LAB2BGR)
    def update(self, frame_bgr):
        luma = self._estimate_luma(frame_bgr)
        self.last_luma = luma
        if self.ema_luma is None:
            self.ema_luma = luma
        else:
            self.ema_luma = (1.0 - self.alpha)*self.ema_luma + self.alpha*luma
        desired = self.gain_max if self.ema_luma < 1e-3 else float(np.clip(self.target / self.ema_luma, self.gain_min, self.gain_max))
        step = np.clip(desired - self.gain, -self.max_step, self.max_step)
        self.gain = float(np.clip(self.gain + step, self.gain_min, self.gain_max))
        return self._apply_gain_safe(frame_bgr, self.gain)

# ====== Utilities: สี ======
def make_mask_fixed(hsv):
    mask_r = None
    for lo, hi in RED_RANGES:
        m = cv.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask_r = m if mask_r is None else (mask_r | m)
    lower_k = np.array(BLACK_RANGE[0], np.uint8)
    upper_k = np.array(BLACK_RANGE[1], np.uint8)
    mask_k = cv.inRange(hsv, lower_k, upper_k)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (MORPH_KSIZE, MORPH_KSIZE))
    def clean(m):
        m = cv.morphologyEx(m, cv.MORPH_OPEN,  k, iterations=1)
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, k, iterations=1)
        m = cv.dilate(m, k, iterations=1)
        return m
    return clean(mask_r), clean(mask_k)

def box_mask_coverage(mask, xyxy, dilate_px=COVER_DILATE):
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
    mask_d = cv.dilate(mask, k, iterations=1)
    x1,y1,x2,y2 = map(int, xyxy)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(mask.shape[1]-1, x2); y2 = min(mask.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1: return 0.0
    roi = mask_d[y1:y2, x1:x2]
    return float(np.count_nonzero(roi)) / float(roi.size + 1e-6)

def color_scores_in_bbox(hsv, xyxy):
    x1,y1,x2,y2 = map(int, xyxy)
    H, W = hsv.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W-1, x2); y2 = min(H-1, y2)
    if x2 <= x1 or y2 <= y1: return 0.0, 0.0
    roi = hsv[y1:y2, x1:x2]
    red = None
    for lo, hi in RED_RANGES:
        m = cv.inRange(roi, np.array(lo, np.uint8), np.array(hi, np.uint8))
        red = m if red is None else (red | m)
    black = cv.inRange(roi, np.array(BLACK_RANGE[0], np.uint8),
                            np.array(BLACK_RANGE[1], np.uint8))
    total = roi.shape[0] * roi.shape[1] + 1e-6
    red_frac   = float(np.count_nonzero(red))   / total
    black_frac = float(np.count_nonzero(black)) / total
    return red_frac, black_frac

# -------- helpers ----------
def clamp_box_to_image(box, w, h):
    if box is None: return None
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w-1))
    x2 = max(0, min(int(x2), w-1))
    y1 = max(0, min(int(y1), h-1))
    y2 = max(0, min(int(y2), h-1))
    if x2 <= x1 or y2 <= y1: return None
    return (x1, y1, x2, y2)

def draw_skeleton(frame,kps,color,radius=3,thickness=2):
    for (x,y,c) in kps:
        if c>0.2: cv.circle(frame,(int(x),int(y)),radius,color,-1)
    for a,b in SKELETON_PAIRS:
        xa,ya,ca=kps[a]; xb,yb,cb=kps[b]
        if ca>0.2 and cb>0.2:
            cv.line(frame,(int(xa),int(ya)),(int(xb),int(yb)),color,thickness)

def get_body_box(kps):
    pts=[]
    for idx in [5,6,11,12]:
        x,y,c=kps[idx]
        if c>0.3: pts.append((int(x),int(y)))
    if len(pts)<2: return None
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return (min(xs),min(ys),max(xs),max(ys))

def get_head_box(kps):
    x,y,c=kps[0]
    if c>0.3: return (int(x-30),int(y-40),int(x+30),int(y+40))
    return None

def extend_hand(kps,wrist,elbow):
    x1,y1,c1=kps[wrist]; x2,y2,c2=kps[elbow]
    if c1<0.2 or c2<0.2: return None
    dx=x1-x2; dy=y1-y2
    return (int(x1+0.5*dx), int(y1+0.5*dy))

def box_center(b):
    if not b: return None
    x1,y1,x2,y2=b
    return ((x1+x2)//2, (y1+y2)//2)

def box_min_distance_px(b1, b2):
    if not b1 or not b2: return 1e9
    x1,y1,x2,y2=b1; a1,b1_,a2,b2_=b2
    dx=max(a1-x2, x1-a2, 0)
    dy=max(b1_-y2, y1-b2_, 0)
    return (dx*dx+dy*dy)**0.5

def point_to_box_distance_px(pt, box):
    if pt is None or box is None: return 1e9
    hx,hy=pt; x1,y1,x2,y2=box
    dx=max(x1-hx,0,hx-x2); dy=max(y1-hy,0,hy-y2)
    return (dx*dx+dy*dy)**0.5

# ===== NEW: นับแบบ "รอดึงออก" =====
def update_punch_state_pullback(state, dist_px, blocked, thresh_px,
                                hold_start_ts, now_ms):
    """
    state: "out" (นอกระยะ พร้อมนับเมื่อเข้า) หรือ "latched" (นับแล้ว รอออก)
    - ถ้าเข้า <= enter และค้าง >= PUNCH_ENTER_HOLD_MS -> นับ 1 ครั้ง แล้ว state="latched"
    - ต้องออก > exit (enter + PUNCH_EXIT_ADD_CM) ถึงจะกลับ "out" และนับรอบใหม่ได้
    blocked=True จะรีเซ็ต hold และไม่ให้เปลี่ยนสถานะ/นับ
    """
    if blocked:
        return "out", 0, 0

    enter = int(thresh_px)
    exit_ = int(thresh_px + PUNCH_EXIT_ADD_CM * PX_PER_CM)

    if state == "out":
        if dist_px <= enter:
            # เริ่ม/ต่อเนื่องการถือระยะ
            if hold_start_ts == 0:
                hold_start_ts = now_ms
            # ค้างพอหรือยัง
            if (now_ms - hold_start_ts) >= PUNCH_ENTER_HOLD_MS:
                # นับหนึ่งครั้ง แล้ว latch รอออก
                return "latched", 1, 0
            return "out", 0, hold_start_ts
        else:
            # นอกระยะ รีเซ็ตการถือ
            return "out", 0, 0

    elif state == "latched":
        # ยังไม่ออกนอก exit → รออยู่
        if dist_px > exit_:
            # ดึงออกแล้ว → พร้อมนับรอบหน้า
            return "out", 0, 0
        else:
            return "latched", 0, 0

    # fallback
    return "out", 0, 0

def draw_hand_distance(frame, hand, box, px_per_cm, punch_cm):
    if hand is None or box is None: return
    hx,hy = hand
    cx=(box[0]+box[2])//2; cy=(box[1]+box[3])//2
    dist_px = point_to_box_distance_px(hand, box)
    dist_cm = dist_px / px_per_cm
    color = COLOR_GREEN if dist_cm <= punch_cm else COLOR_RED2
    cv.circle(frame,(hx,hy),7,COLOR_YELLOW,-1)
    cv.line(frame,(hx,hy),(cx,cy),color,2)
    cv.putText(frame,f"{dist_cm:.1f}cm",(hx+8,hy-8),
               cv.FONT_HERSHEY_SIMPLEX,0.55,color,2,cv.LINE_AA)

# ===== Beep =====
ENABLE_BEEP = True
def beep():
    if not ENABLE_BEEP: return
    try:
        if sys.platform.startswith("win"):
            import winsound; winsound.Beep(1200, 90)
        else:
            print('\a', end='')
    except Exception:
        pass

# ===================== Open both videos =====================
def open_cap(path):
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {path}")
    return cap

cap_main = open_cap(VIDEO_PATH_MAIN)
cap_alt  = open_cap(VIDEO_PATH_ALT)

active_view = 'main'
missing_counter = {'main': 0, 'alt': 0}

# เก็บเวลาสลับมุมมองล่าสุด (ms) และสถานะก่อนหน้าในการตรวจ clinch
last_view_switch_ms = 0
prev_in_clinch = False

# FPS
_cap_fps = cap_main.get(cv.CAP_PROP_FPS)
if not _cap_fps or _cap_fps <= 1.0:
    _cap_fps = 30.0

# ===================== Identity anti-swap =====================
prev_boxes = {'BLACK': None, 'RED': None}
SWAP_IOU_TH   = 0.35
MARGIN_STRONG = 0.20
MARGIN_WEAK   = 0.08

def iou(boxA, boxB):
    if boxA is None or boxB is None: return 0.0
    xA = max(int(boxA[0]), int(boxB[0])); yA = max(int(boxA[1]), int(boxB[1]))
    xB = min(int(boxA[2]), int(boxB[2])); yB = min(int(boxA[3]), int(boxB[3]))
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0: return 0.0
    areaA = max(1, (int(boxA[2])-int(boxA[0]))*(int(boxA[3])-int(boxA[1])))
    areaB = max(1, (int(boxB[2])-int(boxB[0]))*(int(boxB[3])-int(boxB[1])))
    return inter / float(areaA + areaB - inter)

# ===================== Main =====================
pose_model = YOLO(POSE_WEIGHTS)
auto_exp = SmoothAutoBrightness()

paused=False
prev=time.time(); fps=0.0

score_red,score_black = 0,0

# >>> NEW: สถานะหมัดแบบ pull-back + เวลาถือระยะ
# state เริ่มที่ "out"
hand_state       = {"RED":{"L":"out","R":"out"}, "BLACK":{"L":"out","R":"out"}}
hand_hold_start  = {"RED":{"L":0,"R":0},         "BLACK":{"L":0,"R":0}}

in_clinch = False

# === History / Snapshot ===
os.makedirs("snapshots", exist_ok=True)
hit_preview = None
hit_history = deque(maxlen=50)
snap_id = 0

# === Dataset CSV ===
dataset_file = open("dataset.csv", mode="w", newline="", encoding="utf-8")
writer = csv.writer(dataset_file)
writer.writerow(["frame_id","snapshot","team","hand_side","dist_cm","label"])
dataset_file.flush()

show_masks = False  # toggle ด้วยปุ่ม m

try:
    while True:
        if not paused:
            ok1, frame_main = cap_main.read()
            ok2, frame_alt  = cap_alt.read()
            if not ok1 or not ok2:
                cap_main.set(cv.CAP_PROP_POS_FRAMES, 0)
                cap_alt.set(cv.CAP_PROP_POS_FRAMES, 0)
                ok1, frame_main = cap_main.read()
                ok2, frame_alt  = cap_alt.read()
                if not ok1 or not ok2:
                    continue

        frame = frame_main if active_view == 'main' else frame_alt

        now=time.time(); dt=now-prev; prev=now
        fps=0.9*fps+0.1*(1.0/dt if dt>0 else 0)
        now_ms = int(now*1000)

        # Auto brightness
        frame_auto = auto_exp.update(frame)

        # ====== สี fixed HSV + morphology ======
        blur = cv.GaussianBlur(frame_auto, (5,5), 0)
        hsv  = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        mask_r, mask_k = make_mask_fixed(hsv)

        out = frame_auto.copy()
        results = pose_model(out, conf=POSE_CONF, imgsz=IMG_SIZE, verbose=False)

        # ===== เลือก "ดำสุด" และ "แดงสุด" + anti-swap =====
        players=[]
        if len(results) and len(results[0].boxes):
            boxes_xyxy=results[0].boxes.xyxy.cpu().numpy()
            kps_all=results[0].keypoints
            if kps_all is not None:
                kps_all=kps_all.data.cpu().numpy()

                cands=[]
                for i,xyxy in enumerate(boxes_xyxy):
                    kps=kps_all[i]
                    cov_r = box_mask_coverage(mask_r, xyxy)
                    cov_k = box_mask_coverage(mask_k, xyxy)
                    frac_r, frac_k = color_scores_in_bbox(hsv, xyxy)
                    s_r=W_COV*cov_r+W_COL*frac_r
                    s_k=W_COV*cov_k+W_COL*frac_k
                    cands.append({"kps":kps,"box":xyxy,"s_red":float(max(0.0,s_r)),
                                  "s_blk":float(max(0.0,s_k))})

                if len(cands) >= 2:
                    idx_black = int(np.argmax([c["s_blk"] for c in cands]))
                    cand_black = cands[idx_black]
                    rest = [c for j,c in enumerate(cands) if j!=idx_black]
                    idx_red = int(np.argmax([c["s_red"] for c in rest]))
                    cand_red = rest[idx_red]

                    sorted_blk = sorted([c["s_blk"] for c in cands], reverse=True)
                    sorted_red = sorted([c["s_red"] for c in cands], reverse=True)
                    margin_black = (sorted_blk[0] - sorted_blk[1]) if len(sorted_blk)>1 else 1.0
                    margin_red   = (sorted_red[0] - sorted_red[1]) if len(sorted_red)>1 else 1.0

                    cur_BLACK = ("BLACK", COLOR_BLACK, cand_black["kps"], cand_black["box"])
                    cur_RED   = ("RED",   COLOR_RED,   cand_red["kps"],  cand_red["box"])

                    if prev_boxes['BLACK'] is not None and prev_boxes['RED'] is not None:
                        iou_B_prevB = iou(cur_BLACK[3], prev_boxes['BLACK'])
                        iou_B_prevR = iou(cur_BLACK[3], prev_boxes['RED'])
                        iou_R_prevB = iou(cur_RED[3],   prev_boxes['BLACK'])
                        iou_R_prevR = iou(cur_RED[3],   prev_boxes['RED'])
                        crossed = (iou_B_prevR > iou_B_prevB) and (iou_R_prevB > iou_R_prevR)
                        weak_pair = (margin_black < MARGIN_STRONG) and (margin_red < MARGIN_STRONG)
                        if crossed and weak_pair and (max(iou_B_prevR, iou_R_prevB) > SWAP_IOU_TH):
                            cur_BLACK, cur_RED = cur_RED, cur_BLACK

                    if prev_boxes['BLACK'] is not None and margin_black < MARGIN_WEAK:
                        if iou(cur_BLACK[3], prev_boxes['RED']) > iou(cur_BLACK[3], prev_boxes['BLACK']):
                            cur_BLACK, cur_RED = cur_RED, cur_BLACK
                    if prev_boxes['RED'] is not None and margin_red < MARGIN_WEAK:
                        if iou(cur_RED[3], prev_boxes['BLACK']) > iou(cur_RED[3], prev_boxes['RED']):
                            cur_BLACK, cur_RED = cur_RED, cur_BLACK

                    players = [cur_BLACK, cur_RED]
                    prev_boxes['BLACK'] = players[0][3] if players[0][0]=='BLACK' else players[1][3]
                    prev_boxes['RED']   = players[1][3] if players[1][0]=='RED'   else players[0][3]

        # ===== ระยะ/คลินช์/นับคะแนน =====
        dist_min = None
        if len(players)==2:
            (team1,col1,kps1,box1),(team2,col2,kps2,box2)=players
            H, W = out.shape[:2]

            body1,head1=get_body_box(kps1),get_head_box(kps1)
            body2,head2=get_body_box(kps2),get_head_box(kps2)
            body1 = clamp_box_to_image(body1, W, H)
            head1 = clamp_box_to_image(head1, W, H)
            body2 = clamp_box_to_image(body2, W, H)
            head2 = clamp_box_to_image(head2, W, H)

            dist_body=box_min_distance_px(body1,body2)
            dist_head=box_min_distance_px(head1,head2)
            dist_min=min(dist_body,dist_head)

            th = PX_PER_CM * CLINCH_RANGE_CM
            in_clinch = (dist_min <= th)

            # หากเพิ่งเข้าไปในระยะคลินช์ ให้สลับมุมมองทันที (แต่มี cooldown)
            if in_clinch and not prev_in_clinch:
                if (now_ms - last_view_switch_ms) >= SWITCH_COOLDOWN_MS:
                    active_view = 'alt' if active_view == 'main' else 'main'
                    # รีเซ็ตตัวนับการหายไปของมุมมองใหม่
                    missing_counter[active_view] = 0
                    last_view_switch_ms = now_ms

            prev_in_clinch = in_clinch

            c1, c2 = box_center(body1), box_center(body2)
            if c1 and c2:
                cv.line(out, c1, c2, COLOR_CYAN if not in_clinch else COLOR_MAG, 2)
                cm_show = dist_min / PX_PER_CM
                cv.putText(out, f"{cm_show:.1f}cm", ((c1[0]+c2[0])//2+6, (c1[1]+c2[1])//2-6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CYAN if not in_clinch else COLOR_MAG, 2, cv.LINE_AA)

            h1L,h1R=extend_hand(kps1,9,7),extend_hand(kps1,10,8)
            h2L,h2R=extend_hand(kps2,9,7),extend_hand(kps2,10,8)
            thresh_px = int(PX_PER_CM * PUNCH_RANGE_CM + 0.5)

            # บล็อกการนับตอนคลินช์หรือไม่ (รวมบล็อกหลังสลับมุม 200ms)
            blocked_for_scoring = (BLOCK_SCORING_DURING_CLINCH and in_clinch) or ((now_ms - last_view_switch_ms) < SCORE_BLOCK_AFTER_SWITCH_MS)

            # team1 -> team2
            for side,hand_pt in (("L",h1L),("R",h1R)):
                d_body=point_to_box_distance_px(hand_pt,body2)
                d_head=point_to_box_distance_px(hand_pt,head2)
                d_min=min(d_body,d_head)

                state0 = hand_state[team1][side]
                hold0  = hand_hold_start[team1][side]
                state1,score,hold1 = update_punch_state_pullback(
                    state0, d_min, blocked_for_scoring, thresh_px, hold0, now_ms
                )
                hand_state[team1][side]=state1
                hand_hold_start[team1][side]=hold1

                if score and not blocked_for_scoring:
                    if team1=="RED": score_red+=1
                    else: score_black+=1
                    beep()
                    snap_id += 1
                    fname = f"snapshots/hit_{snap_id:04d}_{team1}.jpg"
                    writer.writerow([snap_id, fname, team1, side, f"{d_min/PX_PER_CM:.2f}", 1])
                    dataset_file.flush()
                    cv.imwrite(fname, out)
                    img = cv.imread(fname)
                    if img is not None:
                        scale_w = 480
                        ratio = min(scale_w / float(img.shape[1]), 1.0)
                        hit_preview = cv.resize(img, (int(img.shape[1]*ratio), int(img.shape[0]*ratio)))
                        hit_history.append(img)

                draw_hand_distance(out,hand_pt,body2,PX_PER_CM,PUNCH_RANGE_CM)
                draw_hand_distance(out,hand_pt,head2,PX_PER_CM,PUNCH_RANGE_CM)

            # team2 -> team1
            for side,hand_pt in (("L",h2L),("R",h2R)):
                d_body=point_to_box_distance_px(hand_pt,body1)
                d_head=point_to_box_distance_px(hand_pt,head1)
                d_min=min(d_body,d_head)

                state0 = hand_state[team2][side]
                hold0  = hand_hold_start[team2][side]
                state1,score,hold1 = update_punch_state_pullback(
                    state0, d_min, blocked_for_scoring, thresh_px, hold0, now_ms
                )
                hand_state[team2][side]=state1
                hand_hold_start[team2][side]=hold1

                if score and not blocked_for_scoring:
                    if team2=="RED": score_red+=1
                    else: score_black+=1
                    beep()
                    snap_id += 1
                    fname = f"snapshots/hit_{snap_id:04d}_{team2}.jpg"
                    writer.writerow([snap_id, fname, team2, side, f"{d_min/PX_PER_CM:.2f}", 1])
                    dataset_file.flush()
                    cv.imwrite(fname, out)
                    img = cv.imread(fname)
                    if img is not None:
                        scale_w = 480
                        ratio = min(scale_w / float(img.shape[1]), 1.0)
                        hit_preview = cv.resize(img, (int(img.shape[1]*ratio), int(img.shape[0]*ratio)))
                        hit_history.append(img)

                draw_hand_distance(out,hand_pt,body1,PX_PER_CM,PUNCH_RANGE_CM)
                draw_hand_distance(out,hand_pt,head1,PX_PER_CM,PUNCH_RANGE_CM)

            # วาด bbox ช่วย debug
            for box in [body1,body2]:
                if box: cv.rectangle(out,(box[0],box[1]),(box[2],box[3]),(0,255,255),2)
            for box in [head1,head2]:
                if box: cv.rectangle(out,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)

            good_distance = (dist_min is not None) and (dist_min < 1e8)
            if good_distance: missing_counter[active_view] = 0
            else:             missing_counter[active_view] += 1
        else:
            missing_counter[active_view] += 1

        if missing_counter[active_view] >= MISSING_MAX_FRAMES:
            # ตรวจ cooldown ก่อนสลับโดย auto-missing
            if (now_ms - last_view_switch_ms) >= SWITCH_COOLDOWN_MS:
                active_view = 'alt' if active_view == 'main' else 'main'
                missing_counter[active_view] = 0
                last_view_switch_ms = now_ms

        # วาด skeleton
        for team,col,kps,xyxy in players:
            draw_skeleton(out,kps,col)

        # HUD
        cv.putText(out,f"FPS:{fps:.1f}",(10,28),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv.putText(out,f"RED:{score_red}",(10,60),cv.FONT_HERSHEY_SIMPLEX,0.9,COLOR_RED,2)
        cv.putText(out,f"BLACK:{score_black}",(10,95),cv.FONT_HERSHEY_SIMPLEX,0.9,COLOR_WHITE,3,cv.LINE_AA)
        cv.putText(out,f"BLACK:{score_black}",(10,95),cv.FONT_HERSHEY_SIMPLEX,0.9,COLOR_BLACK,1,cv.LINE_AA)
        cv.putText(out,f"Punch:{PUNCH_RANGE_CM:.1f}cm ({int(PX_PER_CM*PUNCH_RANGE_CM+0.5)}px)",(10,130),cv.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)
        cv.putText(out,f"Clinch<{CLINCH_RANGE_CM:.1f}cm (+{CLINCH_HYS_CM:.1f})  {'CLINCHING' if in_clinch else ''}",
                   (10,165),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,200,200) if not in_clinch else (0,0,255),2)
        view_name = os.path.basename(VIDEO_PATH_MAIN if active_view=='main' else VIDEO_PATH_ALT)
        cv.putText(out, f"VIEW: {view_name}", (10,198), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv.putText(out, f"Luma:{auto_exp.last_luma:.2f}  Gain:{auto_exp.gain:.2f}",
                   (10, 224), cv.FONT_HERSHEY_SIMPLEX, 0.6, (180,220,255), 2)

        # แสดงผลหลัก
        cv.imshow("Boxing", out)

        # Overlay masks (กด m เพื่อ toggle)
        if show_masks:
            cv.imshow("mask_red",   mask_r)
            cv.imshow("mask_black", mask_k)
        else:
            try:
                if cv.getWindowProperty("mask_red", 0) >= 0:   cv.destroyWindow("mask_red")
                if cv.getWindowProperty("mask_black", 0) >= 0: cv.destroyWindow("mask_black")
            except:
                pass

        # Preview + History
        if hit_preview is not None:
            cv.imshow("Hit-Preview", hit_preview)
        if len(hit_history)>0:
            show = list(hit_history)[-10:]
            thumbs=[]
            for i,img in enumerate(show, start=1):
                th=cv.resize(img,(160,120))
                cv.putText(th,str(i),(6,22),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv.LINE_AA)
                thumbs.append(th)
            strip=np.hstack(thumbs)
            cv.imshow("History (1-10)", strip)

        # ชะลอความเร็วเล่นวิดีโอ
        target_frame_time = (1.0 / max(_cap_fps, 1e-6)) / max(SLOW_FACTOR, 1e-6)
        extra_sleep = target_frame_time - (time.time() - now)
        if extra_sleep > 0:
            time.sleep(extra_sleep)

        # คีย์ลัด
        key=cv.waitKey(1)&0xFF
        if key==ord('q') or key==27: break
        elif key==ord('p') or key==ord('s'): paused=not paused
        elif key==ord('b'): ENABLE_BEEP = not ENABLE_BEEP
        elif key==ord('m'): show_masks = not show_masks
        elif key==ord('['): PUNCH_RANGE_CM = max(1.0, PUNCH_RANGE_CM-0.2)
        elif key==ord(']'): PUNCH_RANGE_CM = PUNCH_RANGE_CM+0.2
        elif key==ord('{'): CLINCH_RANGE_CM = max(0.0, CLINCH_RANGE_CM-2.0)
        elif key==ord('}'): CLINCH_RANGE_CM = CLINCH_RANGE_CM+2.0
        elif key==ord('='): PX_PER_CM = PX_PER_CM + 0.1
        elif key==ord('-'): PX_PER_CM = max(0.5, PX_PER_CM - 0.1)

finally:
    try: cap_main.release()
    except: pass
    try: cap_alt.release()
    except: pass
    try: cv.destroyAllWindows()
    except: pass
    try: dataset_file.close()
    except: pass
