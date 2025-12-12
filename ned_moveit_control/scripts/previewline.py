#!/usr/bin/env python3
from flask import Flask, request, send_file, jsonify
import io, os, cv2, numpy as np, math, time

# ============================================================
# 📄 ค่ากระดาษจำลอง (เมตร)
# ============================================================
PAPER_WIDTH = 0.21
PAPER_HEIGHT = 0.297


# ============================================================
# ✏️ ฟังก์ชันช่วยจัดเรียงและดึงเส้นขอบ
# ============================================================
def sort_contours_by_area(contours, reverse=True):
    return sorted(contours, key=cv2.contourArea, reverse=reverse)


def extract_contours_from_image(img_path, mode="anime"):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {img_path}")

    img = cv2.resize(img, (300, 300))
    img = cv2.flip(img, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if mode == "anime":
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 9, 2
        )
    else:
        edges = cv2.Canny(img, 80, 180)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sort_contours_by_area(contours)

    all_paths = []
    if contours:
        xs = [p[0][0] for c in contours for p in c]
        ys = [p[0][1] for c in contours for p in c]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        scale_x = PAPER_WIDTH / (max_x - min_x)
        scale_y = PAPER_HEIGHT / (max_y - min_y)
        scale = min(scale_x, scale_y) * 0.70

        offset_x = (PAPER_WIDTH - (max_x - min_x) * scale) / 2
        offset_y = (PAPER_HEIGHT - (max_y - min_y) * scale) / 2

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
                path_points.append((wx, wy))
            all_paths.append(path_points)
    return all_paths


# ============================================================
# 👤 โหมด PORTRAIT (คน) — เวอร์ชันปรับใหม่ ลายเส้นเรียบเหมือนภาพซ้าย
# ============================================================
def extract_portrait_contours(img_path, mode="portrait"):
    import cv2, numpy as np

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ภาพ: {img_path}")

    # === ✅ Resize โดยคงสัดส่วนจริง (ไม่บี้) ===
    h, w = img.shape[:2]
    max_side = 700
    scale_ratio = max_side / max(h, w)
    new_w, new_h = int(w * scale_ratio), int(h * scale_ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # === หมุน + กลับภาพถ้าต้องการ ===
    img = cv2.flip(img, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # === ใช้ adaptive threshold เหมือน anime ===
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        9, 2
    )

    # === หา contours ===
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # === แปลงพิกัดเป็นหน่วยจริงของกระดาษ A4 ===
    PAPER_WIDTH, PAPER_HEIGHT = 0.21, 0.297
    all_paths = []

    if contours:
        xs = [p[0][0] for c in contours for p in c]
        ys = [p[0][1] for c in contours for p in c]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # ✅ ใช้อัตราส่วนของภาพจริงในการ scale
        img_ratio = (max_x - min_x) / (max_y - min_y)
        paper_ratio = PAPER_WIDTH / PAPER_HEIGHT
        if img_ratio > paper_ratio:
            scale = PAPER_WIDTH / (max_x - min_x)
        else:
            scale = PAPER_HEIGHT / (max_y - min_y)
        scale *= 0.70  # เว้นขอบกระดาษ

        offset_x = (PAPER_WIDTH - (max_x - min_x) * scale) / 2
        offset_y = (PAPER_HEIGHT - (max_y - min_y) * scale) / 2

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
                path_points.append((wx, wy))
            all_paths.append(path_points)

    print(f"✅ Portrait (Scaled 1:1 Real Ratio) contours: {len(all_paths)}")
    return all_paths





# ============================================================
# 🐶 โหมด PET (ไม่แก้)
# ============================================================
def extract_pet_contours(img_path):
    import cv2, numpy as np

    # === โหลดภาพขาวดำ ===
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ภาพ: {img_path}")

    # === ✅ Resize โดยคงสัดส่วนจริง (ไม่บี้) ===
    h, w = img_gray.shape[:2]
    max_side = 700
    scale_ratio = max_side / max(h, w)
    new_w, new_h = int(w * scale_ratio), int(h * scale_ratio)
    img_resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # === หมุนและกลับภาพให้เหมือนระบบคน ===
    img_resized = cv2.flip(img_resized, 1)
    img_resized = cv2.rotate(img_resized, cv2.ROTATE_90_CLOCKWISE)

    # === ใช้ adaptive threshold แบบ anime ===
    binary = cv2.adaptiveThreshold(
        img_resized, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        9, 2
    )

    # === หา contours ===
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # === แปลงพิกัดเป็นหน่วยจริงของกระดาษ A4 ===
    PAPER_WIDTH, PAPER_HEIGHT = 0.21, 0.297
    all_paths = []

    if contours:
        xs = [p[0][0] for c in contours for p in c]
        ys = [p[0][1] for c in contours for p in c]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # ✅ ใช้อัตราส่วนของภาพจริงในการ scale
        img_ratio = (max_x - min_x) / (max_y - min_y)
        paper_ratio = PAPER_WIDTH / PAPER_HEIGHT
        if img_ratio > paper_ratio:
            scale = PAPER_WIDTH / (max_x - min_x)
        else:
            scale = PAPER_HEIGHT / (max_y - min_y)
        scale *= 0.70  # เว้นขอบกระดาษเล็กน้อย

        offset_x = (PAPER_WIDTH - (max_x - min_x) * scale) / 2
        offset_y = (PAPER_HEIGHT - (max_y - min_y) * scale) / 2

        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue
            epsilon = 0.0001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            path_points = []
            for p in approx:
                x, y = p[0]
                wx = offset_x + (x - min_x) * scale
                wy = offset_y + (y - min_y) * scale
                path_points.append((wx, wy))
            all_paths.append(path_points)

    print(f"✅ Pet (Anime-like & True Scale) contours: {len(all_paths)}")
    return all_paths


# ============================================================
# 🖼️ ฟังก์ชันวาดและบันทึกภาพ
# ============================================================
def save_drawn_image(paths, filename="preview.png", img_width=800, rotate_inner=0):
    img_height = int(img_width * PAPER_HEIGHT / PAPER_WIDTH)
    canvas = np.ones((img_height, img_width), dtype=np.uint8) * 255
    all_points = [pt for path in paths for pt in path]
    if not all_points:
        cv2.imwrite(filename, canvas)
        return filename

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    scale_x = (img_width - 20) / (max_x - min_x)
    scale_y = (img_height - 20) / (max_y - min_y)
    scale = min(scale_x, scale_y)
    offset_x = (img_width - (max_x - min_x) * scale) / 2
    offset_y = (img_height - (max_y - min_y) * scale) / 2

    for path in paths:
        for i in range(len(path) - 1):
            x1 = int((path[i][0] - min_x) * scale + offset_x)
            y1 = int((path[i][1] - min_y) * scale + offset_y)
            x2 = int((path[i + 1][0] - min_x) * scale + offset_x)
            y2 = int((path[i + 1][1] - min_y) * scale + offset_y)
            cv2.line(canvas, (x1, y1), (x2, y2), (0,), 2)

    if rotate_inner != 0:
        rot = cv2.getRotationMatrix2D((img_width / 2, img_height / 2), rotate_inner, 1)
        canvas = cv2.warpAffine(canvas, rot, (img_width, img_height), borderValue=255)
    canvas = cv2.flip(canvas, 1)

    cv2.imwrite(filename, canvas)
    print(f"✅ Saved preview to {filename}")
    return filename


# ============================================================
# 🚀 Flask API
# ============================================================
app = Flask(__name__)

@app.route('/preview_line', methods=['POST'])
def preview_line():
    try:
        image_file = request.files['image']
        mode = request.form.get('mode', 'anime')

        os.makedirs("temp_inputs", exist_ok=True)
        temp_path = os.path.join("temp_inputs", "input_temp.png")
        image_file.save(temp_path)

        if mode == "pet":
            paths = extract_pet_contours(temp_path)
        elif mode == "portrait":
            paths = extract_portrait_contours(temp_path)
        else:
            paths = extract_contours_from_image(temp_path, mode=mode)

        output_path = os.path.join("temp_inputs", "preview_result.png")
        save_drawn_image(paths, output_path, img_width=1200, rotate_inner=90)

        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🏁 Run Server
# ============================================================
# ============================================================
# 🎨 MAIN (สำหรับรันจาก command line)
# ============================================================
if __name__ == "__main__":
    import argparse
    import sys
    if len(sys.argv) == 1:
        print("🌐 Starting Flask API server at http://0.0.0.0:8100 ...")
        app.run(host="0.0.0.0", port=8100, debug=True)
        sys.exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path ของรูปภาพ")
    parser.add_argument("--mode", required=True, choices=["anime", "sketch", "pet", "portrait"])
    parser.add_argument("--output", default="preview_result.png")
    args = parser.parse_args()

    if args.mode == "pet":
        paths = extract_pet_contours(args.path)
    elif args.mode == "portrait":
        paths = extract_portrait_contours(args.path)
    else:
        paths = extract_contours_from_image(args.path, mode=args.mode)

    save_drawn_image(paths, args.output, img_width=1200, rotate_inner=90)

