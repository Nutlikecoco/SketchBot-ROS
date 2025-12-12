#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState

def talker():
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rospy.init_node('fake_joint_state_pub', anonymous=True)
    rate = rospy.Rate(10)  # ส่งทุก 0.1 วินาที (10Hz)

    while not rospy.is_shutdown():
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']
        # เซ็ตค่าเริ่มต้น เช่น joint_3 = 1.57 rad (ท่าตั้งตรง)
        js.position = [0.0, 0.0, 1.57, 0.0, 0.0, 0.0]
        js.velocity = []
        js.effort = []
        pub.publish(js)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

