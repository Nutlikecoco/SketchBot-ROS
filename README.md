# SketchBot-ROS

ROS package for controlling the Niryo Ned robot to draw images on paper using MoveIt motion planning and OpenCV image processing.

This package is part of the SketchBot system, which converts digital images into robot-drawn sketches.

---

# System Overview

SketchBot consists of two main components:

1. SketchBot-Application (Frontend + Image Processing)
2. SketchBot-ROS (Robot Motion Control)

Workflow:

Image → SketchBot-Application → Processing → ROS → Niryo Ned → Drawing on Paper

---

# Hardware Requirements

- Niryo Ned / Ned1 / Ned2 robot
- Pencil attached to end effector
- Paper workspace
- Ubuntu 20.04
- ROS Noetic

---

# Software Requirements

- ROS Noetic
- MoveIt
- OpenCV
- Python 3

Install dependencies:

```bash
sudo apt install ros-noetic-moveit
sudo apt install ros-noetic-cv-bridge
sudo apt install python3-opencv

```

# Running the Robot
-roslaunch niryo_robot_moveit_config webviz_demo.launch
-rosrun robot_state_publisher robot_state_publisher
-rosrun joint_state_publisher_gui joint_state_publisher_gui
-rosrun rviz rviz -d ~/my_robot_config.rviz
