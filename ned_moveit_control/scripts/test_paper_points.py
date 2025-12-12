#!/usr/bin/env python3
import rospy, sys, cv2, base64, numpy as np
import time, os
import math
import threading
import io
from std_msgs.msg import ColorRGBA, Empty
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import Marker
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import String
from tf.transformations import quaternion_from_euler
import pyautogui
import tf
import json

PAPER_WIDTH  = 0.21     # A4 width
PAPER_HEIGHT = 0.297    # A4 height
ORIGIN_X = 0.245
ORIGIN_Y = -0.1485
DRAW_HEIGHT = 0.31

marker_pub = None

is_paused = False
is_drawing = False
is_finished = False
latest_image_path = None
latest_mode = None
current_image_path = None
current_mode = None
paths = []
marker_id = 0


STATE_DIR = "/home/nut/sketchbot_saves"
os.makedirs(STATE_DIR, exist_ok=True)

def upload_to_firebase(image_path, user_id=None):
    import firebase_admin
    from firebase_admin import credentials, storage, firestore
    import os, time, traceback, rospy

    try:
        if not image_path or not isinstance(image_path, str) or not os.path.exists(image_path):
            rospy.logwarn(f"Skip upload_to_firebase: invalid image_path ({image_path})")
            return

        cred_path = os.path.expanduser("~/catkin_ws/src/ned_moveit_control/scripts/serviceAccountKey.json")

        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'sketchbot-app.firebasestorage.app'
            })

        bucket = storage.bucket()

        if not user_id:
            uid_paths = [
                "/media/sf_Downloads/current_uid.txt",  
                os.path.expanduser("~/catkin_ws/src/ned_moveit_control/scripts/current_uid.txt")
            ]

            found_uid = None
            for path in uid_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        found_uid = f.read().strip()
                        break

            if found_uid:
                user_id = found_uid
                rospy.loginfo(f"Loaded UID: {user_id}")
            else:
                rospy.logwarn("ไม่พบ current_uid.txt — ใช้ค่า default_user")
                user_id = "default_user"

        # ==================  UPLOAD TO FIREBASE STORAGE ==================
        filename = f"drawn_result_{int(time.time())}.png"
        blob = bucket.blob(f"{user_id}/{filename}")
        blob.upload_from_filename(image_path)
        blob.make_public()
        url = blob.public_url

        rospy.loginfo(f"Uploaded to Firebase Storage: {url}")

        # ==================  SAVE URL TO FIRESTORE ==================
        db = firestore.client()
        db.collection("users").document(user_id).collection("my_images").add({
            "url": url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename
        })

        rospy.loginfo("URL saved to Firestore successfully.")

    except Exception as e:
        rospy.logerr(f"Upload failed: {e}\n{traceback.format_exc()}")



def save_drawing_state(group, marker_id, paths, mode, is_paused, is_finished, image_path=None):
    import json, os, rospy, time

    save_dir = "/home/nut/sketchbot_saves"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = int(time.time())
    filepath = os.path.join(save_dir, f"state_{mode}_{timestamp}.json")

    img_path = image_path or globals().get("current_image_path") or globals().get("latest_image_path")

    if img_path and os.path.exists(img_path):
        rospy.loginfo(f"เก็บ image_path แบบ local: {img_path}")

    clean_paths = []
    for p in paths:
        if isinstance(p, list):
            clean_paths.append([list(map(float, pt)) for pt in p])

    state = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "mode": mode,
        "image_path": img_path,
        "marker_id": marker_id,
        "paths": clean_paths,
        "total_paths": len(clean_paths),
        "is_paused": is_paused,
        "is_finished": is_finished
    }

    try:
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        rospy.loginfo(f"บันทึก state เสร็จ → {filepath}")
        rospy.loginfo(f"marker_id={marker_id}, total_paths={len(clean_paths)}")
        rospy.loginfo(f"image_path ที่ถูกบันทึก: {img_path}")
        
        saved_list = sorted(glob.glob(os.path.join(save_dir, "state_*.json")))
        pub_saved_states.publish(json.dumps(saved_list))
        rospy.loginfo(f"Published updated saved_states list ({len(saved_list)} files).")
    except Exception as e:
        rospy.logerr(f"Save command failed: {e}")

    return filepath


def show_previous_drawn_paths(paths, upto_index, marker_pub):
    if not paths or upto_index <= 0:
        return

    rospy.loginfo(f"แสดงเส้นก่อนหน้า {upto_index} เส้นที่วาดไปแล้ว...")
    for i, path in enumerate(paths[:upto_index]):
        marker = Marker()
        marker.header.frame_id = "base_link"   
        marker.header.stamp = rospy.Time.now()
        marker.ns = "drawn_lines"            
        marker.id = 1000 + i
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.0025
        marker.color = ColorRGBA(0.0, 0.0, 0.0, 1.0)
        marker.pose.orientation.w = 1.0

        for p in path:
            marker.points.append(Point(x=p[0], y=p[1], z=p[2]))

        marker_pub.publish(marker)
        rospy.sleep(0.01)

    rospy.loginfo("แสดงเส้นก่อนหน้าเสร็จสิ้น")

def load_drawing_state(filepath):
    if not os.path.exists(filepath):
        rospy.logerr(f"ไม่พบไฟล์ state: {filepath}")
        return None
    try:
        with open(filepath, "r") as f:
            state = json.load(f)
        rospy.loginfo(f"โหลด state จาก {filepath}")
        return state
    except Exception as e:
        rospy.logerr(f"โหลด state ล้มเหลว: {e}")
        return None

def sort_contours_by_area(contours, reverse=True):
    return sorted(contours, key=cv2.contourArea, reverse=reverse)

def publish_marker_progress(path, marker_id=0, delay=0.02):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.ns = "drawn_lines"
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.0015
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.pose.orientation.w = 1.0
    marker.lifetime = rospy.Duration(0.0)

    points = []
    for (x, y, z) in path:
        points.append(Point(x=x, y=y, z=z+0.002))
        marker.points = points[:]
        marker.header.stamp = rospy.Time.now()
        marker_pub.publish(marker)
        rospy.sleep(delay)


def preprocess_basic(img_gray):
    _, binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    return binary

def preprocess_anime(img_gray):
    binary = cv2.adaptiveThreshold(
        img_gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        9, 2
    )
    return binary
    

def preprocess_sketch(img_gray):
    edges = cv2.Canny(img_gray, 80, 180)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(edges, kernel, iterations=1)
    return binary


def extract_outline_lines(img_gray, min_length=10, epsilon=2.0):
    img = cv2.GaussianBlur(img_gray.copy(), (3, 3), 0)
    edges_canny = cv2.Canny(img, 80, 150)
    lap = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
    _, edges_lap = cv2.threshold(lap, 20, 255, cv2.THRESH_BINARY)
    edges = cv2.bitwise_or(edges_canny, edges_lap)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        if cv2.arcLength(cnt, True) < min_length:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        for i in range(len(approx) - 1):
            x1, y1 = approx[i][0]
            x2, y2 = approx[i+1][0]
            lines.append((int(x1), int(y1), int(x2), int(y2)))
    return lines

def build_object_mask(img_gray, outline_lines):
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    for (x1, y1, x2, y2) in outline_lines:
        cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_filled = mask.copy()
    h, w = mask.shape
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(mask_filled, flood_mask, (0,0), 128)
    object_mask = cv2.bitwise_not(mask_filled)
    object_mask[mask > 0] = 255
    return object_mask

def lines_to_paths(lines, offset_z=DRAW_HEIGHT, scale=1.0, offset_x=ORIGIN_X, offset_y=ORIGIN_Y):
    paths = []
    for (x1,y1,x2,y2) in lines:
        path = [
            (offset_x + x1*scale, offset_y + y1*scale, offset_z),
            (offset_x + x2*scale, offset_y + y2*scale, offset_z)
        ]
        paths.append(path)
    return paths

def extract_contours_from_image(img_path, offset_z=DRAW_HEIGHT, mode="basic"):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("ไม่พบไฟล์ภาพ: {}".format(img_path))

    img = cv2.resize(img, (300, 300))
    img = cv2.flip(img, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if mode == "basic":
        binary = preprocess_basic(img)
    elif mode == "anime":
        binary = preprocess_anime(img)
    else:
        binary = preprocess_anime(img)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sort_contours_by_area(contours, reverse=True)

    all_paths = []
    if contours:
        xs = [p[0][0] for c in contours for p in c]
        ys = [p[0][1] for c in contours for p in c]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        scale_x = PAPER_WIDTH  / (max_x - min_x)
        scale_y = PAPER_HEIGHT / (max_y - min_y)
        scale   = min(scale_x, scale_y) * 0.70

        offset_x = ORIGIN_X + (PAPER_WIDTH  - (max_x - min_x) * scale) / 2
        offset_y = ORIGIN_Y + (PAPER_HEIGHT - (max_y - min_y) * scale) / 2

        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                continue
            epsilon = 0.0001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            path_points = []
            for p in approx:
                x, y = p[0]
                wx = offset_x + (x - min_x) * scale
                wy = offset_y + (y - min_y) * scale
                wz = offset_z
                path_points.append((wx, wy, wz))
            if len(path_points) >= 2:
                all_paths.append(path_points)

    return all_paths


# โหมด PORTRAIT 

def extract_portrait_contours(img_path, offset_z=DRAW_HEIGHT):
    import cv2, numpy as np

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {img_path}")

    h, w = img.shape[:2]
    max_side = 700
    scale_ratio = max_side / max(h, w)
    new_w, new_h = int(w * scale_ratio), int(h * scale_ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img = cv2.flip(img, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        9, 2
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    PAPER_WIDTH, PAPER_HEIGHT = 0.21, 0.297
    all_paths = []

    if contours:
        xs = [p[0][0] for c in contours for p in c]
        ys = [p[0][1] for c in contours for p in c]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        img_ratio = (max_x - min_x) / (max_y - min_y)
        paper_ratio = PAPER_WIDTH / PAPER_HEIGHT
        if img_ratio > paper_ratio:
            scale = PAPER_WIDTH / (max_x - min_x)
        else:
            scale = PAPER_HEIGHT / (max_y - min_y)
        scale *= 0.70  

        offset_x = ORIGIN_X + (PAPER_WIDTH - (max_x - min_x) * scale) / 2
        offset_y = ORIGIN_Y + (PAPER_HEIGHT - (max_y - min_y) * scale) / 2

        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                continue
            epsilon = 0.0001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            path_points = []
            for p in approx:
                x, y = p[0]
                wx = offset_x + (x - min_x) * scale
                wy = offset_y + (y - min_y) * scale
                wz = offset_z
                path_points.append((wx, wy, wz))
            if len(path_points) > 2:
                all_paths.append(path_points)

    rospy.loginfo(f"Portrait (Scaled 1:1 Real Ratio) contours: {len(all_paths)}")
    return all_paths

def extract_pet_contours(img_path, offset_z=DRAW_HEIGHT):
    import cv2, numpy as np

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {img_path}")

    h, w = img.shape[:2]
    max_side = 700
    scale_ratio = max_side / max(h, w)
    new_w, new_h = int(w * scale_ratio), int(h * scale_ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img = cv2.flip(img, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        9, 2
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    PAPER_WIDTH, PAPER_HEIGHT = 0.21, 0.297
    all_paths = []

    if contours:
        xs = [p[0][0] for c in contours for p in c]
        ys = [p[0][1] for c in contours for p in c]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        img_ratio = (max_x - min_x) / (max_y - min_y)
        paper_ratio = PAPER_WIDTH / PAPER_HEIGHT
        if img_ratio > paper_ratio:
            scale = PAPER_WIDTH / (max_x - min_x)
        else:
            scale = PAPER_HEIGHT / (max_y - min_y)
        scale *= 0.70  # เว้นขอบกระดาษ

        offset_x = ORIGIN_X + (PAPER_WIDTH - (max_x - min_x) * scale) / 2
        offset_y = ORIGIN_Y + (PAPER_HEIGHT - (max_y - min_y) * scale) / 2

        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                continue
            epsilon = 0.0001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            path_points = []
            for p in approx:
                x, y = p[0]
                wx = offset_x + (x - min_x) * scale
                wy = offset_y + (y - min_y) * scale
                wz = offset_z
                path_points.append((wx, wy, wz))
            if len(path_points) > 2:
                all_paths.append(path_points)

    rospy.loginfo(f"Pet (Scaled 1:1 Real Ratio) contours: {len(all_paths)}")
    return all_paths

# Save drawn image
def rotate_points(path, angle_deg, cx, cy):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotated = []
    for (x, y, z) in path:
        dx, dy = x - cx, y - cy
        rx = dx * cos_a - dy * sin_a + cx
        ry = dx * sin_a + dy * cos_a + cy
        rotated.append((rx, ry, z))
    return rotated    

def save_drawn_image(paths, filename=None, img_width=800, rotate_inner=0): 
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"/home/nut/Downloads/drawn_result_{timestamp}.png"

    img_height = int(img_width * PAPER_HEIGHT / PAPER_WIDTH)
    canvas = np.ones((img_height, img_width), dtype=np.uint8) * 255
    all_points = [pt for path in paths for pt in path]
    if not all_points:
        cv2.imwrite(filename, canvas)
        rospy.logwarn("ไม่มีจุดให้วาด เซฟผ้าขาวไว้ที่ %s", filename)
        return

    xs = [p[0] for p in all_points]; ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if rotate_inner != 0:
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        paths = [rotate_points(path, rotate_inner, cx, cy) for path in paths]
        rotated_points = [pt for path in paths for pt in path]
        xs = [p[0] for p in rotated_points]; ys = [p[1] for p in rotated_points]
        min_x, max_x = min(xs), max(xs); min_y, max_y = min(ys), max(ys)

    scale_x = (img_width - 20) / (max_x - min_x)
    scale_y = (img_height - 20) / (max_y - min_y)
    scale = min(scale_x, scale_y)
    offset_x = (img_width - (max_x - min_x) * scale) / 2
    offset_y = (img_height - (max_y - min_y) * scale) / 2

    for path in paths:
        for i in range(len(path)-1):
            x1 = int((path[i][0]-min_x) * scale + offset_x)
            y1 = int((path[i][1]-min_y) * scale + offset_y)
            x2 = int((path[i+1][0]-min_x) * scale + offset_x)
            y2 = int((path[i+1][1]-min_y) * scale + offset_y)
            cv2.line(canvas, (x1, y1), (x2, y2), (0,), 2)

    canvas = cv2.flip(canvas, 0)
    cv2.imwrite(filename, canvas)
    rospy.loginfo("เซฟไว้ที่ %s (A4 %dx%d px)", filename, img_width, img_height)
    return filename
    
def publish_preview_on_board(image_path):
    from tf.transformations import quaternion_from_euler
    pub = rospy.Publisher("/preview_marker", Marker, queue_size=1, latch=True)
    rospy.sleep(1.0)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        rospy.logwarn("ไม่พบภาพ preview ที่ %s", image_path)
        return

    img = cv2.resize(img, (300, 300))
    img = cv2.flip(img, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    marker = Marker()
    marker.header.frame_id = "board_link"
    marker.ns = "preview_lines"
    marker.id = 0
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.002
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    q = quaternion_from_euler(3.15, 0, 0)
    marker.pose.orientation.x = q[0]
    marker.pose.orientation.y = q[1]
    marker.pose.orientation.z = q[2]
    marker.pose.orientation.w = q[3]

    xs = [p[0][0] for c in contours for p in c]
    ys = [p[0][1] for c in contours for p in c]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    scale_x = PAPER_WIDTH / (max_x - min_x)
    scale_y = PAPER_HEIGHT / (max_y - min_y)
    scale = min(scale_x, scale_y) * 1.8   

    img_w_scaled = (max_x - min_x) * scale
    img_h_scaled = (max_y - min_y) * scale
    offset_x = -img_w_scaled / 2
    offset_y = -img_h_scaled / 2

    for cnt in contours:
        pts = cnt.squeeze()

        if len(pts.shape) < 2 or len(pts) < 2:
            continue

        if cv2.contourArea(cnt) < 5:
            continue

        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            marker.points.append(Point(
                x=offset_x + (x1 - min_x) * scale,
                y=offset_y + (y1 - min_y) * scale,
                z=0.005
            ))
            marker.points.append(Point(
                x=offset_x + (x2 - min_x) * scale,
                y=offset_y + (y2 - min_y) * scale,
                z=0.005
            ))

    pub.publish(marker)
    rospy.loginfo("แสดงพรีวิวเส้นบน board_link สำเร็จ (%s)", image_path)

    
def publish_image_on_board(image_path):
    pub = rospy.Publisher("/image_marker", Marker, queue_size=1, latch=True)
    rospy.sleep(1.0)

    img = cv2.imread(image_path)
    if img is None:
        rospy.logwarn("ไม่พบไฟล์ภาพ: %s", image_path)
        return
    
    img = cv2.rotate(img, cv2.ROTATE_180)

    try:
        img = cv2.resize(img, (300, 424))
    except Exception as e:
        rospy.logwarn(f"ปรับขนาดภาพไม่สำเร็จ: {e}")
        return

    if img.size == 0:
        rospy.logwarn("ข้อมูลภาพว่างเปล่า (image size == 0)")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        small = cv2.resize(img_rgb, (60, 84), interpolation=cv2.INTER_LANCZOS4)
    except Exception as e:
        rospy.logwarn(f"resize (Lanczos) ล้มเหลว: {e}")
        return

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    try:
        small = cv2.filter2D(small, -1, kernel)
    except Exception as e:
        rospy.logwarn(f"sharpen filter ล้มเหลว: {e}")
        return

    try:
        colors = small.reshape(-1, 3) / 255.0
    except Exception as e:
        rospy.logwarn(f"reshape สีผิดพลาด: {e}")
        return

    marker = Marker()
    marker.header.frame_id = "world"
    marker.ns = "image_display"
    marker.id = 999
    marker.type = Marker.POINTS
    marker.action = Marker.ADD

    marker.scale.x = 0.007   
    marker.scale.y = 0.007  

    marker.pose.position.x = 0.5038
    marker.pose.position.y = -0.448
    marker.pose.position.z = 0.3050

    q = quaternion_from_euler(1.97, 0, 0)
    marker.pose.orientation.x = q[0]
    marker.pose.orientation.y = q[1]
    marker.pose.orientation.z = q[2]
    marker.pose.orientation.w = q[3]

    h, w, _ = small.shape
    for j in range(h):
        for i in range(w):
            color = colors[j * w + i]
            if np.any(np.isnan(color)) or np.any(np.isinf(color)):
                continue

            px, py = i, j
            marker.points.append(Point(
                x=(px - w / 2) * (0.42 / 60.0),
                y=(py - h / 2) * (0.594 / 84.0),
                z=0
            ))
            marker.colors.append(ColorRGBA(color[0], color[1], color[2], 1.0))

    if len(marker.points) == 0:
        rospy.logwarn("ไม่มีข้อมูลพิกเซลถูกเพิ่มใน marker (image ว่างหรือ error ตอน loop)")
        return

    pub.publish(marker)
    rospy.loginfo("แสดงภาพ %s บน world สำเร็จ (POINTS Mode: Safe & Smooth)", image_path)

# RViz Capture Thread 
def start_rviz_camera_publisher(fps=5, region=None):
    pub = rospy.Publisher("/camera/image_raw/compressed", CompressedImage, queue_size=1)
    bridge = CvBridge()

    def capture_loop():
        rate = rospy.Rate(5)
        rospy.loginfo("เริ่มส่งภาพ RViz ไปยัง /camera/image_raw/compressed (%d fps)", fps)
        while not rospy.is_shutdown():
            try:
                screenshot = pyautogui.screenshot(region=region)
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

                frame = cv2.resize(frame, (1280, 720))

                msg = CompressedImage()
                msg.header.stamp = rospy.Time.now()
                msg.format = "jpeg"
                msg.data = np.array(cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1]).tobytes()

                pub.publish(msg)
            except Exception as e:
                rospy.logwarn("❌ Capture error: %s", str(e))
            rate.sleep()

    t = threading.Thread(target=capture_loop)
    t.daemon = True
    t.start()
    
def interpolate_path(path_xyz, max_step=0.005):
    dense = []
    for i in range(len(path_xyz) - 1):
        x1,y1,z1 = path_xyz[i]
        x2,y2,z2 = path_xyz[i+1]
        dx,dy,dz = (x2-x1),(y2-y1),(z2-z1)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        n = max(1, int(dist / max_step))
        for k in range(n):
            t = k/float(n)
            dense.append((x1+dx*t, y1+dy*t, z1+dz*t))
    dense.append(path_xyz[-1])
    return dense

def cartesian_plan_with_retries(group, waypoints, tries=4):
    cfgs = [
        (0.003, False),
        (0.005, False),
        (0.007, True),
        (0.010, True),
    ][:tries]

    for (eef_step, avoid_collisions) in cfgs:
        plan, fraction = group.compute_cartesian_path(
            waypoints,
            eef_step,
            avoid_collisions
        )
        rospy.loginfo("   try eef_step=%.3f avoid_collisions=%s -> fraction=%.2f",
                      eef_step, avoid_collisions, fraction)
        if fraction > 0.95:
            return plan, fraction

    return plan, fraction  

is_paused = False

def command_callback(msg):
    global is_paused, is_drawing, is_finished, latest_image_path, latest_mode, group
    global paths, marker_id
    cmd = msg.data.lower().strip()
    rospy.loginfo(f"Received external command: {cmd}")

    if cmd == "stop":
        if is_drawing and not is_paused:
            is_paused = True
            rospy.loginfo("Drawing paused (robot stopped temporarily)")
        else:
            rospy.loginfo("Stop ignored: robot not drawing or already paused")

    elif cmd == "start":
        if is_drawing and is_paused:
            is_paused = False
            rospy.loginfo("Resume drawing from paused state")
        elif not is_drawing:
            rospy.logwarn("No drawing in progress to resume")

    elif cmd == "save":
        rospy.loginfo("Received SAVE command — saving current drawing state...")
        rospy.sleep(0.3)
        try:
            if "paths" in globals() and globals()["paths"]:
                current_paths = globals()["paths"]
                current_marker = globals().get("marker_id", 0)
                save_drawing_state(group, current_marker, current_paths, latest_mode or "basic", is_paused, is_finished)
            else:
                rospy.logwarn("ไม่มีข้อมูล paths ใน memory — เซฟเฉพาะ state ทั่วไป")
                save_drawing_state(group, 0, [], latest_mode or "basic", is_paused, is_finished)
        except Exception as e:
            rospy.logerr(f"Save command failed: {e}")

    else:
        rospy.loginfo(f"Unknown command: {cmd}")


def save_current_pose(group, save_dir="/tmp/pose_saves"):
    import json, time
    os.makedirs(save_dir, exist_ok=True)
    
    pose = group.get_current_pose().pose
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "position": {
            "x": round(pose.position.x, 5),
            "y": round(pose.position.y, 5),
            "z": round(pose.position.z, 5),
        },
        "orientation": {
            "x": round(pose.orientation.x, 5),
            "y": round(pose.orientation.y, 5),
            "z": round(pose.orientation.z, 5),
            "w": round(pose.orientation.w, 5),
        }
    }

    filename = os.path.join(save_dir, f"pose_{int(time.time())}.json")
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    rospy.loginfo(f"บันทึก Pose ลงไฟล์: {filename}")

    pub = rospy.Publisher("/sketchbot/pose_saved", String, queue_size=1)
    rospy.sleep(0.5)
    pub.publish("pose_saved")
    rospy.loginfo("ส่งสัญญาณ pose_saved กลับไปยัง Flutter สำเร็จ")


# MAIN

def main():
    import argparse
    global marker_pub, is_paused, is_drawing, is_finished, latest_image_path, latest_mode, current_image_path, current_mode
    global paths, marker_id 

    roscpp_initialize(sys.argv)
    rospy.init_node("draw_from_image_shadow_only", anonymous=True)
    marker_pub = rospy.Publisher("/drawing_marker", Marker, queue_size=10)
    rospy.Subscriber("/sketchbot_command", String, command_callback)
    rospy.sleep(1.0)  

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=False, help="absolute image path or downloaded tmp path")
    parser.add_argument("--mode", required=False, choices=["basic", "anime", "pet", "portrait"], help="drawing mode")
    parser.add_argument("--preview-only", action="store_true", help="only generate preview, skip robot motion")
    parser.add_argument("--resume-file", help="path of saved state file to resume from")
    parser.add_argument("--save-only", action="store_true", help="Save current state without drawing")

    args = parser.parse_args()
    current_image_path = args.path
    current_mode = args.mode
    
    #เขียน last_image_path.txt
    try:
        with open(os.path.expanduser("~/catkin_ws/src/ned_moveit_control/scripts/last_image_path.txt"), "w") as f:
            f.write(current_image_path or "")
    except Exception:
        pass
        
    if args.save_only:
        rospy.loginfo("Save-only mode: capturing current state without drawing...")

        img_path = current_image_path or "/tmp/placeholder.png"
        mode_used = current_mode or "anime"

        save_drawing_state(None, 0, [], mode_used, False, False, image_path=img_path)
        rospy.loginfo("Save-only completed.")
        sys.exit(0)

    if not args.path or not args.mode:
        rospy.logerr("ต้องระบุ --path และ --mode เมื่อไม่ได้ใช้ --save-only")
        sys.exit(1)
        
    if args.resume_file:
        rospy.loginfo(f"Resume mode active: loading from {args.resume_file}")
        state = load_drawing_state(args.resume_file)
        if not state:
            rospy.logerr("โหลด state ไม่สำเร็จ → ออกจากโปรแกรม")
            sys.exit(1)

        image_path = state.get("image_path", args.path)
        mode = state.get("mode", args.mode)
        paths = state.get("paths", [])
        start_marker = state.get("marker_id", 0) + 1
        show_previous_drawn_paths(paths, start_marker, marker_pub)
        
        if not image_path or image_path in [None, "", "null", "/tmp/placeholder.png"]:
            try:
                last_file = os.path.expanduser("~/catkin_ws/src/ned_moveit_control/scripts/last_image_path.txt")
                if os.path.exists(last_file):
                    with open(last_file, "r") as f:
                        image_path = f.read().strip()
                        rospy.loginfo(f"ใช้ path ล่าสุดจาก last_image_path.txt → {image_path}")
            except Exception as e:
                rospy.logwarn(f"อ่าน last_image_path.txt ไม่ได้: {e}")

        if not image_path or not os.path.exists(image_path):
            rospy.logwarn("ไม่พบไฟล์ภาพต้นฉบับ → ใช้ /tmp/placeholder.png ชั่วคราว")
            placeholder_path = "/tmp/placeholder.png"
            if not os.path.exists(placeholder_path):
                cv2.imwrite(placeholder_path, np.ones((200, 200, 3), dtype=np.uint8) * 255)
            image_path = placeholder_path

        if start_marker >= len(paths):
            rospy.logwarn(f"Resume marker_id ({start_marker}) ≥ total paths ({len(paths)}) → adjusting")
            start_marker = max(0, len(paths) - 1)
    
        rospy.loginfo(f"Resume จาก marker_id={start_marker}, total paths={len(paths)}")
    else:
        image_path = args.path
        mode = args.mode
        start_marker = 0
        
    is_drawing = True
    is_finished = False
    latest_image_path = image_path
    latest_mode = mode

    rospy.loginfo(f"เริ่มวาดภาพจาก {image_path} (mode={mode})")

    if not args.preview_only:
        start_rviz_camera_publisher(fps=8, region=(610, 130, 780, 440))

    check_img = cv2.imread(image_path)
    if check_img is None:
        rospy.logerr("❌ ไม่พบไฟล์ภาพ หรือไฟล์เสีย: %s", image_path)
        sys.exit(1)

    try:
        global group
        group = MoveGroupCommander("arm")
        group.set_planning_time(45.0)
        group.set_max_velocity_scaling_factor(0.05)
        group.set_max_acceleration_scaling_factor(0.05)
        group.set_goal_position_tolerance(0.0015)
        group.set_goal_orientation_tolerance(0.01)
    except Exception as e:
        rospy.logerr("สร้าง MoveGroup ไม่ได้: %s", str(e))
        sys.exit(1)

    rospy.sleep(1.0)
    orientation = group.get_current_pose().pose.orientation

    if mode == "basic":
        paths = extract_portrait_contours(image_path, offset_z=DRAW_HEIGHT)
        delay = 0.03
        globals()["paths"] = paths
    elif mode == "anime":
        paths = extract_portrait_contours(image_path, offset_z=DRAW_HEIGHT)
        delay = 0.03
        globals()["paths"] = paths
    elif mode == "pet":
        rospy.loginfo("ใช้โหมด pet (ลายเส้นเรียบ สมดุล)")
        paths = extract_pet_contours(image_path, offset_z=DRAW_HEIGHT)
        delay = 0.03
        globals()["paths"] = paths
    elif mode == "portrait":
        rospy.loginfo("ใช้โหมด PORTRAIT (ลายเส้นเรียบ สมดุล)")
        paths = extract_portrait_contours(image_path, offset_z=DRAW_HEIGHT)
        delay = 0.03
        globals()["paths"] = paths
    else:
        rospy.logerr("mode ไม่ถูกต้อง: %s", mode)
        sys.exit(1)

    try:
        preview_path = save_drawn_image(paths, filename="/tmp/preview.png", rotate_inner=0)
        rospy.loginfo("แสดงพรีวิวบนบอร์ด (โหมด %s)...", mode)
        publish_image_on_board(image_path)
        publish_preview_on_board("/tmp/preview.png")
    except Exception as e:
        rospy.logwarn("ไม่สามารถแสดงพรีวิวได้: %s", str(e))

    hover_dz = 0.010
    marker_id = 0
    rospy.loginfo("Starting actual drawing process (%d paths total)", len(paths))

    try:
        for marker_id, path in enumerate(paths[start_marker:], start=start_marker):
            globals()["marker_id"] = marker_id
            globals()["paths"] = paths 
            if not path:
                continue

            while is_paused and not rospy.is_shutdown():
                rospy.loginfo_throttle(2.0, "Waiting for resume command...")
                rospy.sleep(0.5)

            dense_xyz = interpolate_path(path, max_step=0.005)

            start_x, start_y, start_z = dense_xyz[0]
            approach = Pose()
            approach.position.x, approach.position.y, approach.position.z = start_x, start_y, start_z + hover_dz
            approach.orientation = orientation
            group.set_pose_target(approach)
            group.go(wait=True)
            group.stop()
            group.clear_pose_targets()

            waypoints = []
            for (x, y, z) in dense_xyz:
                p = Pose()
                p.position.x, p.position.y, p.position.z = x, y, z
                p.orientation = orientation
                waypoints.append(p)

            plan, fraction = cartesian_plan_with_retries(group, waypoints, tries=4)
            rospy.loginfo("Path %d fraction = %.2f", marker_id, fraction)

            if fraction < 0.9:
                alt_waypoints = []
                for (x, y, z) in dense_xyz:
                    p = Pose()
                    p.position.x, p.position.y, p.position.z = x, y, z + 0.005
                    p.orientation = orientation
                    alt_waypoints.append(p)
                rospy.logwarn(f"Path {marker_id} fraction={fraction:.2f} -> replan raised Z")
                plan, fraction = cartesian_plan_with_retries(group, alt_waypoints, tries=4)
                rospy.loginfo(f"Path {marker_id} (raised Z) fraction={fraction:.2f}")

            if fraction > 0.9:
                rospy.loginfo(f"Executing path {marker_id} ...")

                while is_paused and not rospy.is_shutdown():
                    rospy.loginfo_throttle(2.0, "Paused during execution...")
                    rospy.sleep(0.5)

                group.execute(plan, wait=True)
                publish_marker_progress(path, marker_id, delay)
            else:
                rospy.logwarn(f"Path {marker_id} planning failed after retries (fraction={fraction:.2f})")

            marker_id += 1

        final_image_path = save_drawn_image(paths, rotate_inner=90) 
        save_current_pose(group)
        rospy.loginfo("Shadow draw เสร็จสมบูรณ์ (cleaned)")
        upload_to_firebase(final_image_path, user_id="jirapat0716@gmail.com")
        
        try:
            status_pub = rospy.Publisher("/sketchbot/status", String, queue_size=10)
            rospy.sleep(0.5)  
            status_pub.publish("upload_complete")
            rospy.loginfo("Published upload_complete → /sketchbot/status")
        except Exception as e:
            rospy.logwarn(f"ส่งสถานะ upload_complete ไม่สำเร็จ: {e}")        
            
        topics_to_clear = ["/drawing_marker", "/preview_marker", "/image_marker"]
        for topic in topics_to_clear:
            pub = rospy.Publisher(topic, Marker, queue_size=1, latch=False)
            rospy.sleep(0.2)
            clear_marker = Marker()
            clear_marker.action = Marker.DELETEALL
            pub.publish(clear_marker)
            rospy.sleep(0.3)
            pub.publish(clear_marker)
            rospy.loginfo(f"ล้าง {topic} เรียบร้อยแล้ว (double clear)")
        rospy.sleep(1.0)
        rospy.loginfo("ล้าง marker ทั้งหมดสำเร็จ!")

        is_drawing = False
        is_finished = True
        rospy.loginfo("Drawing complete. Ready for replay on next 'start' command.")

    finally:
        if not is_finished:
            rospy.logwarn("Drawing interrupted before completion.")
        roscpp_shutdown()


if __name__ == "__main__":
    main()  
