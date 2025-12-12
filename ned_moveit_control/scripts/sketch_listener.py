#!/usr/bin/env python3
import rospy, json, subprocess, os, requests, time
from std_msgs.msg import String
import tf
import urllib.request, cv2
import numpy as np
from geometry_msgs.msg import Pose, Quaternion
from moveit_commander import MoveGroupCommander
from tf.transformations import quaternion_from_euler
import math
SAVE_DIR = "/tmp/sketchbot_images"
STATE_DIR = "/home/nut/sketchbot_saves"   
LAST_PATH_FILE = os.path.expanduser("~/catkin_ws/src/ned_moveit_control/scripts/last_image_path.txt")
TEST_SCRIPT = os.path.expanduser("~/catkin_ws/src/ned_moveit_control/scripts/test_paper_points.py")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

def draw_from_paths(paths):
    rospy.loginfo(f" เริ่มวาด {len(paths)} segments จาก Flutter")

    try:
        move_group = MoveGroupCommander("arm")
        move_group.set_max_velocity_scaling_factor(0.2)
        move_group.set_max_acceleration_scaling_factor(0.2)

        for i, segment in enumerate(paths):
            if not segment:
                continue

            waypoints = []
            for pt in segment:
                x, y, z = pt
                pose = Pose()
                pose.position.x = float(x)
                pose.position.y = float(y)
                pose.position.z = float(z)
                pose.orientation = Quaternion(*quaternion_from_euler(0, math.pi, 0))
                waypoints.append(pose)

            (plan, fraction) = move_group.compute_cartesian_path(
                waypoints,
                eef_step=0.01,
                jump_threshold=0.0,
                avoid_collisions=False
            )

            if fraction > 0:
                move_group.execute(plan, wait=True)
                move_group.stop()
                rospy.loginfo(f"เส้นที่ {i+1}/{len(paths)} วาดสำเร็จ (fraction={fraction:.2f})")
            else:
                rospy.logwarn(f"เส้นที่ {i+1} คำนวณ path ไม่สำเร็จ (fraction={fraction:.2f})")

            rospy.sleep(0.05)

        rospy.loginfo("วาดเสร็จทุกเส้นเรียบร้อยแล้ว")

    except Exception as e:
        rospy.logerr(f"draw_from_paths error: {e}")



# DOWNLOAD IMAGE TO TMP (from URL)
def download_to_tmp(url: str) -> str:
    """ดาวน์โหลดไฟล์ภาพจาก URL มาเก็บใน /tmp แล้วคืน path"""
    local_path = os.path.join(SAVE_DIR, f"img_{int(time.time())}.png")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(r.content)
    return local_path

#  Util
def ensure_local_image_path(maybe_path_or_url: str) -> str:
    if not maybe_path_or_url:
        return None
    s = str(maybe_path_or_url).strip()
    if s.startswith("http://") or s.startswith("https://"):
        try:
            return download_to_tmp(s)
        except Exception as e:
            rospy.logwarn(f"ดาวน์โหลดจาก URL ไม่สำเร็จ: {e}")
            return None
    if os.path.exists(s):
        return s
    return None

def write_last_image_path(path: str):
    try:
        with open(LAST_PATH_FILE, "w") as f:
            f.write(path or "")
    except Exception as e:
        rospy.logwarn(f"เขียน last_image_path.txt ไม่สำเร็จ: {e}")

def read_last_image_path() -> str:
    try:
        if os.path.exists(LAST_PATH_FILE):
            with open(LAST_PATH_FILE, "r") as f:
                return f.read().strip()
    except Exception as e:
        rospy.logwarn(f"อ่าน last_image_path.txt ไม่ได้: {e}")
    return None

def make_placeholder_if_needed() -> str:
    placeholder_path = "/tmp/placeholder.png"
    try:
        if not os.path.exists(placeholder_path):
            cv2.imwrite(placeholder_path, np.ones((200,200,3), dtype=np.uint8) * 255)
    except Exception as e:
        rospy.logerr(f"สร้าง placeholder.png ไม่สำเร็จ: {e}")
        return None
    return placeholder_path

#  SAVE CURRENT DRAWING STATE (manual)
is_saving = False 

def manual_save_state():
    global is_saving
    if is_saving:
        rospy.logwarn("กำลังบันทึกอยู่แล้ว — ข้ามคำสั่ง save ซ้ำ")
        return

    try:
        is_saving = True
        rospy.loginfo("Manual save requested — saving full drawing state...")

        pub = rospy.Publisher("/sketchbot_command", String, queue_size=1)
        rospy.sleep(0.3)
        pub.publish("save")
        rospy.loginfo("✅ ส่งคำสั่ง save ไปยัง node test_paper_points.py สำเร็จ")

        rospy.sleep(1.5)
        list_saved_states()

        pub_done = rospy.Publisher("/sketchbot/save_done", String, queue_size=1)
        rospy.sleep(0.2)
        pub_done.publish("manual_save_done")

    except Exception as e:
        rospy.logerr(f"Manual save error: {e}")
    finally:
        is_saving = False  


# 💾 SAVE CURRENT ROBOT POSE (base_link → pencil_link)
def save_current_pose():
    import tf
    import math
    from geometry_msgs.msg import Pose

    pose_dir = "/tmp/pose_saves"
    os.makedirs(pose_dir, exist_ok=True)
    listener = tf.TransformListener()

    try:
        rospy.loginfo("waiting for transform base_link → pencil_link ...")
        listener.waitForTransform("base_link", "pencil_link", rospy.Time(), rospy.Duration(3.0))
        (trans, rot) = listener.lookupTransform("base_link", "pencil_link", rospy.Time(0))

        import tf.transformations as tft
        (roll, pitch, yaw) = tft.euler_from_quaternion(rot)

        data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "x": round(trans[0], 4),
            "y": round(trans[1], 4),
            "z": round(trans[2], 4),
            "roll": round(roll, 4),
            "pitch": round(pitch, 4),
            "yaw": round(yaw, 4),
        }

        fname = os.path.join(pose_dir, f"pose_{int(time.time())}.json")
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)

        rospy.loginfo(f"✅ Pose saved to {fname}")
        pub = rospy.Publisher("/sketchbot/pose_saved", String, queue_size=1)
        rospy.sleep(0.5)
        pub.publish("pose_saved")

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logerr(f"TF lookup failed: {e}")
    except Exception as e:
        rospy.logerr(f"save_current_pose error: {e}")


# LIST SAVED STATES
def list_saved_states():
    """ส่งลิสต์ state ทั้งหมดให้ Flutter"""
    global pub_saved_states
    try:
        files = [os.path.join(STATE_DIR, f) for f in os.listdir(STATE_DIR) if f.endswith(".json")]
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        msg = json.dumps(files)
        pub_saved_states.publish(msg)
        rospy.loginfo(f"ส่งรายการ state ทั้งหมด ({len(files)} รายการ)")
    except Exception as e:
        rospy.logerr(f"list_saved_states error: {e}")



# RESUME FROM SAVED STATE FILE
def resume_from_file(filepath: str):
    import os, json, cv2, subprocess, numpy as np, rospy

    if not os.path.exists(filepath):
        rospy.logwarn(f"ม่พบไฟล์ state: {filepath}")
        return

    rospy.loginfo(f"Resume จากไฟล์ {filepath}")
    script = os.path.expanduser("~/catkin_ws/src/ned_moveit_control/scripts/test_paper_points.py")

    mode = "anime"
    original_path = None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        mode = data.get("mode", "anime")
        original_path = data.get("image_path", None)
        rospy.loginfo(f"อ่าน mode={mode}, image_path={original_path}")
    except Exception as e:
        rospy.logwarn(f"อ่าน state file ไม่ได้: {e}")

    if original_path and (str(original_path).startswith("http://") or str(original_path).startswith("https://")):
        try:
            original_path = download_to_tmp(original_path)
            rospy.loginfo(f"ดาวน์โหลดภาพจาก URL → {original_path}")
        except Exception as e:
            rospy.logwarn(f"ดาวน์โหลดจาก URL ไม่สำเร็จ: {e}")
            try:
                file_name = os.path.basename(original_path)
                local_fallback = f"/media/sf_Downloads/{file_name}"
                if os.path.exists(local_fallback):
                    rospy.loginfo(f"ใช้ภาพ fallback จาก local: {local_fallback}")
                    original_path = local_fallback
                else:
                    original_path = None
            except Exception:
                original_path = None

    if not original_path or not isinstance(original_path, str) or original_path.strip() == "":
        last_path_file = os.path.expanduser("~/catkin_ws/src/ned_moveit_control/scripts/last_image_path.txt")
        if os.path.exists(last_path_file):
            with open(last_path_file, "r") as f:
                original_path = f.read().strip()
                rospy.loginfo(f"ใช้ path ล่าสุดจาก last_image_path.txt → {original_path}")

    if not original_path or not os.path.exists(original_path):
        placeholder_path = "/tmp/placeholder.png"
        if not os.path.exists(placeholder_path):
            cv2.imwrite(placeholder_path, np.ones((200, 200, 3), dtype=np.uint8) * 255)
        original_path = placeholder_path
        rospy.logwarn(f"ใช้ placeholder.png แทน (path={original_path})")

    if not isinstance(original_path, str):
        rospy.logerr("❌ original_path ไม่เป็น string — ยกเลิกการ resume")
        return

    cmd = [
        "python3",
        script,
        "--path", str(original_path),
        "--mode", str(mode),
        "--resume-file", str(filepath),
    ]
    rospy.loginfo(f"🚀 เรียก test_paper_points.py ด้วยโหมด resume: {cmd}")

    try:
        subprocess.Popen(cmd)
    except Exception as e:
        rospy.logerr(f"❌ เรียก script resume ไม่ได้: {e}")


#  HANDLE COMMANDS FROM FLUTTER
def handle_command(msg):
    try:
        data = None
        try:
            data = json.loads(msg.data)
        except Exception:
            cmd_simple = msg.data.strip().lower()
            rospy.loginfo(f"📩 Received simple command: {cmd_simple}")
            if cmd_simple == "save":
                manual_save_state()
                return
            elif cmd_simple in ("stop", "start", "resume"):
                pub = rospy.Publisher("/sketchbot_command", String, queue_size=1)
                rospy.sleep(0.3)
                pub.publish(cmd_simple)
                rospy.loginfo(f"Forwarded simple command '{cmd_simple}' to test_paper_points node")
                return
            else:
                rospy.logwarn(f"Unknown simple command: {cmd_simple}")
                return

        cmd = data.get("cmd")
        mode = data.get("mode")
        path = data.get("path")
        url = data.get("url")

        rospy.loginfo(f"recv cmd={cmd} mode={mode} path={path} url={url}")


        if cmd == "list_saves":
            list_saved_states()
            return


        if cmd == "resume_from_file":
            filepath = data.get("path")
            resume_from_file(filepath)
            return


        if cmd == "save":
            manual_save_state()
            return


        if cmd == "delete_save":
            filepath = data.get("path")
            if not filepath or not os.path.exists(filepath):
                rospy.logwarn(f"ไม่พบไฟล์ที่จะลบ: {filepath}")
                return
            try:
                os.remove(filepath)
                rospy.loginfo(f"ลบไฟล์สำเร็จ: {filepath}")
                rospy.sleep(0.5)
                list_saved_states()
            except Exception as e:
                rospy.logerr(f"ลบไฟล์ไม่สำเร็จ: {e}")
            return


        if cmd == "preview":
            if (not path) and url:
                path = download_to_tmp(url)
            local_path = ensure_local_image_path(path)
            if not local_path:
                rospy.logerr(f"image path ไม่ถูกต้อง: {path}")
                return
            try:
                subprocess.call(["python3", TEST_SCRIPT, "--path", local_path, "--mode", mode, "--preview-only"])
                rospy.loginfo("started preview mode only")
            except Exception as e:
                rospy.logerr(f"preview call error: {e}")
            return


        if cmd == "draw":
            if (not path) and url:
                path = download_to_tmp(url)
            local_path = ensure_local_image_path(path)
            if not local_path:
                rospy.logerr(f"❌ image path ไม่ถูกต้อง: {path}")
                return

            write_last_image_path(local_path)

            try:
                time.sleep(1.0)
                subprocess.Popen(["python3", TEST_SCRIPT, "--path", local_path, "--mode", str(mode)])
                rospy.loginfo("started test_paper_points.py")
            except Exception as e:
                rospy.logerr(f"draw call error: {e}")
            return
            
        if cmd == "draw_paths":
            rospy.loginfo("รับคำสั่ง draw_paths จาก Flutter")

            paths = data.get("paths", [])
            if not paths:
                rospy.logwarn("ไม่มีข้อมูล paths จาก Flutter")
                return

            try:
                draw_from_paths(paths)
            except Exception as e:
                rospy.logerr(f"draw_paths error: {e}")
            return
        rospy.logwarn(f"Unrecognized cmd: {cmd}")

    except Exception as e:
        rospy.logerr(f"handle_command error: {e}")


# MAIN
def main():
    rospy.init_node("sketch_listener", anonymous=True)
    rospy.loginfo("sketch_listener ready (listening /sketchbot_command)")

    global pub_saved_states
    pub_saved_states = rospy.Publisher(
        "/sketchbot/saved_states",
        String,
        queue_size=10,
        latch=True
    )

    listener = tf.TransformListener()
    try:
        listener.waitForTransform("base_link", "board_link", rospy.Time(), rospy.Duration(2.0))
        rospy.loginfo("TF ready (base_link → board_link)")
    except tf.Exception:
        rospy.logwarn("TF not ready, using fallback transform")

    rospy.Subscriber("/sketchbot_command", String, handle_command)
    rospy.spin()


if __name__ == "__main__":
    main()

