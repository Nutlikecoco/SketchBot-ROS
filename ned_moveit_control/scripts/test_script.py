#!/usr/bin/env python3
import rospy, sys, cv2, numpy as np
import time, os, math
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import Marker
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from tf.transformations import quaternion_from_euler

# ============================================================
# 🧾 CONFIG
# ============================================================
PAPER_WIDTH  = 0.21     # A4 width (m)
PAPER_HEIGHT = 0.297    # A4 height (m)
ORIGIN_X = 0.245
ORIGIN_Y = -0.1485
DRAW_HEIGHT = 0.31

marker_pub = None

# ============================================================
# 🧩 Utility
# ============================================================
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

# ============================================================
# 🎨 Preprocess
# ============================================================
def preprocess_basic(img_gray):
    _, binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    return binary

def preprocess_anime(img_gray):
    return cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 9, 2
    )

def preprocess_sketch(img_gray):
    edges = cv2.Canny(img_gray, 80, 180)
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)

# ============================================================
# ✏️ Outline Extraction
# ============================================================
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

# ============================================================
# 🧮 Convert lines → robot paths
# ============================================================
def lines_to_paths(lines, offset_z=DRAW_HEIGHT, scale=1.0, offset_x=ORIGIN_X, offset_y=ORIGIN_Y):
    return [[
        (offset_x + x1*scale, offset_y + y1*scale, offset_z),
        (offset_x + x2*scale, offset_y + y2*scale, offset_z)
    ] for (x1, y1, x2, y2) in lines]

# ============================================================
# 🖼️ Extract contours from image
# ============================================================
def extract_contours_from_image(img_path, offset_z=DRAW_HEIGHT, mode="basic"):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {img_path}")

    img = cv2.resize(img, (300, 300))
    img = cv2.flip(img, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if mode == "basic":
        binary = preprocess_basic(img)
    elif mode == "anime":
        binary = preprocess_anime(img)
    else:
        binary = preprocess_sketch(img)

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
        offset_x = ORIGIN_X + (PAPER_WIDTH  - (max_x - min_x)*scale)/2
        offset_y = ORIGIN_Y + (PAPER_HEIGHT - (max_y - min_y)*scale)/2

        for cnt in contours:
            if cv2.contourArea(cnt) < 10: continue
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

# ============================================================
# 💾 Save preview/drawn image
# ============================================================
def save_drawn_image(paths, filename=None, img_width=800, rotate_inner=0): 
    if filename is None:
        filename = f"/tmp/preview.png"
    img_height = int(img_width * PAPER_HEIGHT / PAPER_WIDTH)
    canvas = np.ones((img_height, img_width), dtype=np.uint8) * 255
    all_points = [pt for path in paths for pt in path]
    if not all_points:
        cv2.imwrite(filename, canvas)
        return filename

    xs = [p[0] for p in all_points]; ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale_x = (img_width - 20) / (max_x - min_x)
    scale_y = (img_height - 20) / (max_y - min_y)
    scale = min(scale_x, scale_y)
    offset_x = (img_width - (max_x - min_x) * scale) / 2
    offset_y = (img_height - (max_y - min_y) * scale) / 2
    for path in paths:
        for i in range(len(path)-1):
            x1 = int((path[i][0]-min_x)*scale + offset_x)
            y1 = int((path[i][1]-min_y)*scale + offset_y)
            x2 = int((path[i+1][0]-min_x)*scale + offset_x)
            y2 = int((path[i+1][1]-min_y)*scale + offset_y)
            cv2.line(canvas, (x1,y1), (x2,y2), (0,), 2)
    canvas = cv2.flip(canvas, 0)
    cv2.imwrite(filename, canvas)
    rospy.loginfo("🖼️ เซฟ preview ที่ %s", filename)
    return filename

# ============================================================
# 📄 แสดงพรีวิวบน board_link
# ============================================================
def publish_preview_on_board(image_path):
    from tf.transformations import quaternion_from_euler
    pub = rospy.Publisher("/preview_marker", Marker, queue_size=1, latch=True)
    rospy.sleep(1.0)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        rospy.logwarn("❌ ไม่พบภาพ preview ที่ %s", image_path)
        return

    # 🔄 เตรียมภาพให้แนวเดียวกับกระดาษ
    img = cv2.resize(img, (300, 300))
    img = cv2.flip(img, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 🎯 เตรียม Marker
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

    # 🧭 หมุนให้ตรงกับ board_link
    q = quaternion_from_euler(3.15, 0, 0)
    marker.pose.orientation.x = q[0]
    marker.pose.orientation.y = q[1]
    marker.pose.orientation.z = q[2]
    marker.pose.orientation.w = q[3]

    # 📏 คำนวณขนาดและสเกล
    xs = [p[0][0] for c in contours for p in c]
    ys = [p[0][1] for c in contours for p in c]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    scale_x = PAPER_WIDTH / (max_x - min_x)
    scale_y = PAPER_HEIGHT / (max_y - min_y)
    scale = min(scale_x, scale_y) * 1.5   # ✅ ขยายใหญ่ขึ้น ~2 เท่า (จาก 0.7 → 1.4)

    # ✅ คำนวณ offset ใหม่ให้ภาพอยู่กลางบอร์ดจริง ๆ
    img_w_scaled = (max_x - min_x) * scale
    img_h_scaled = (max_y - min_y) * scale
    offset_x = -img_w_scaled / 2
    offset_y = -img_h_scaled / 2

    # 🧩 วาดเส้นพรีวิว
    for cnt in contours:
        pts = cnt.squeeze()
        for i in range(len(pts)-1):
            x1, y1 = pts[i]
            x2, y2 = pts[i+1]
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
    rospy.loginfo("✅ แสดงพรีวิวเส้นบน board_link สำเร็จ (%s)", image_path)

# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    global marker_pub
    roscpp_initialize(sys.argv)
    rospy.init_node("draw_from_image_with_preview", anonymous=True)
    marker_pub = rospy.Publisher("/drawing_marker", Marker, queue_size=10)
    rospy.sleep(1.0)

    image_path = input("กรุณากรอก path ของรูปภาพ: ").strip()
    print("เลือกประเภทของภาพ:\n[1] Line Art\n[2] Anime\n[3] Sketch")
    image_type = input("เลือก [1/2/3]: ").strip()

    group = MoveGroupCommander("arm")
    group.set_planning_time(30.0)
    group.set_max_velocity_scaling_factor(0.1)
    group.set_max_acceleration_scaling_factor(0.1)
    orientation = group.get_current_pose().pose.orientation

    # ==========================================
    # 🔹 สร้างพรีวิวจากผลประมวลผลจริง
    mode = "basic" if image_type == "1" else "anime" if image_type == "2" else "sketch"
    preview_paths = extract_contours_from_image(image_path, mode=mode)
    preview_path = save_drawn_image(preview_paths, filename="/tmp/preview.png", rotate_inner=90)
    rospy.loginfo("🖼️ แสดงพรีวิวบนบอร์ด (โหมด %s)...", mode)
    publish_preview_on_board(preview_path)
    # ==========================================

    # 🔹 เริ่มวาดจริงบนกระดาษ
    delay = 0.02 if mode == "basic" else 0.04 if mode == "anime" else 0.05
    marker_id = 0
    for path in preview_paths:
        if not path: continue
        waypoints = [Pose(position=Point(x, y, z), orientation=orientation) for (x, y, z) in path]
        (plan, fraction) = group.compute_cartesian_path(waypoints, 0.02, False)
        rospy.loginfo("Path %d fraction=%.2f", marker_id, fraction)
        if fraction > 0.9:
            group.execute(plan, wait=False)
            publish_marker_progress(path, marker_id, delay)
        marker_id += 1

    save_drawn_image(preview_paths, rotate_inner=90)
    rospy.loginfo("✅ Draw completed.")
    roscpp_shutdown()

if __name__ == "__main__":
    main()

