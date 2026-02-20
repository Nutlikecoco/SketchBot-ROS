# SketchBot-ROS

ROS package for controlling a Niryo Ned robot to draw images on paper.

## Package
- ned_moveit_control: Motion planning and drawing scripts using MoveIt and OpenCV

This project uses Niryo Ned robot description from the official Niryo ROS packages.

งานนี้มีฟังก์ชันในการขยับหุ่นยนต์ Niryo ned1 ให้สามารถวาดรูปภาพตามที่ผู้ใช้ add เข้าไปในแอปพลิเคชันได้โดยจะวาดรูปเป็นภาพขาวดำ และวาดเรียงลำดับจากเส้นใหญ่สุดไปเส้นเล็กสุด
มีไฟล์ python ที่เป็น scripts วาดรูปและเชื่อมต่อกับแอปพลิเคชัน
