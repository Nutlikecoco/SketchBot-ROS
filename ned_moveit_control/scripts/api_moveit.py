#!/usr/bin/env python3
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- ROS/MoveIt ---
from moveit_commander import roscpp_initialize, MoveGroupCommander, RobotCommander
import rospy
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from std_srvs.srv import Trigger

app = Flask(__name__)
CORS(app)

# เริ่ม ROS/MoveIt หนึ่งครั้ง
roscpp_initialize(sys.argv)
rospy.init_node('api_bridge', anonymous=True)

# RobotCommander เพื่อเช็ค group ที่มี
robot = RobotCommander()

# Arm group
arm = MoveGroupCommander("arm")

# Gripper service (แทน MoveIt group)
rospy.wait_for_service('/niryo_robot/commander/open_gripper')
rospy.wait_for_service('/niryo_robot/commander/close_gripper')
open_gripper_srv = rospy.ServiceProxy('/niryo_robot/commander/open_gripper', Trigger)
close_gripper_srv = rospy.ServiceProxy('/niryo_robot/commander/close_gripper', Trigger)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify(ok=True)

@app.route("/home", methods=["POST"])
def home():
    arm.set_named_target("home")
    arm.go(wait=True)
    arm.stop()
    return jsonify(ok=True)

@app.route("/move_joints", methods=["POST"])
def move_joints():
    data = request.get_json(force=True)
    joints = data.get("joints", None)
    if not joints or len(joints) != 6:
        return jsonify(ok=False, error="Require 6 joints"), 400
    arm.go(joints, wait=True)
    arm.stop()
    return jsonify(ok=True)

@app.route("/move_pose", methods=["POST"])
def move_pose():
    data = request.get_json(force=True)
    req = ["x","y","z","roll","pitch","yaw"]
    if not all(k in data for k in req):
        return jsonify(ok=False, error=f"Require {req}"), 400
    q = quaternion_from_euler(data["roll"], data["pitch"], data["yaw"])
    pose = Pose()
    pose.position.x = data["x"]; pose.position.y = data["y"]; pose.position.z = data["z"]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
    arm.set_pose_target(pose)
    ok = arm.go(wait=True)
    arm.stop(); arm.clear_pose_targets()
    return jsonify(ok=bool(ok))

@app.route("/open_gripper", methods=["POST"])
def open_gripper():
    try:
        resp = open_gripper_srv()
        return jsonify(ok=resp.success, message=resp.message)
    except rospy.ServiceException as e:
        return jsonify(ok=False, error=str(e)), 500

@app.route("/close_gripper", methods=["POST"])
def close_gripper():
    try:
        resp = close_gripper_srv()
        return jsonify(ok=resp.success, message=resp.message)
    except rospy.ServiceException as e:
        return jsonify(ok=False, error=str(e)), 500

if __name__ == "__main__":
    # สำคัญ: ฟังทุก iface เพื่อให้คอมอื่นเข้าถึงได้
    app.run(host="0.0.0.0", port=5000)


