#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()
current_pose = None

def pose_callback(msg):
    global current_pose
    current_pose = msg

def main():
    rospy.init_node("virtual_camera_node")
    rospy.Subscriber("/virtual_camera_pose", Pose, pose_callback)
    pub = rospy.Publisher("/virtual_camera_image", Image, queue_size=1)

    rospy.loginfo("📷 Virtual camera active...")
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if current_pose:
            # 👇 ตัวอย่าง: สร้างภาพจำลอง / frame ที่ได้จากกล้องมุมนี้
            img = cv2.imread("/home/nut/catkin_ws/src/ned_moveit_control/resources/current_scene.png")
            if img is not None:
                pub.publish(bridge.cv2_to_imgmsg(img, encoding="bgr8"))
        rate.sleep()

if __name__ == "__main__":
    main()

