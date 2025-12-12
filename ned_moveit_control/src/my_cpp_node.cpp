#include <ros/ros.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "my_cpp_node");
    ros::NodeHandle nh;
    ROS_INFO("my_cpp_node started!");
    ros::spin();
    return 0;
}
