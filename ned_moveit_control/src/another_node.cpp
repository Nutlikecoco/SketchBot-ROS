#include <ros/ros.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "another_node");
    ros::NodeHandle nh;
    ROS_INFO("another_node started!");
    ros::spin();
    return 0;
}
