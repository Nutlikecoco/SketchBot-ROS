#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sstream>

int main(int argc, char** argv)
{
    // 1. Initialize ROS node
    ros::init(argc, argv, "ned_cpp_node");
    ros::NodeHandle nh;

    // 2. Create a publisher
    ros::Publisher pub = nh.advertise<std_msgs::String>("chatter", 10);

    // 3. Set loop rate
    ros::Rate loop_rate(1); // 1 Hz

    int count = 0;
    while (ros::ok())
    {
        // 4. Prepare message
        std_msgs::String msg;
        std::stringstream ss;
        ss << "Hello ROS! " << count;
        msg.data = ss.str();

        // 5. Publish message
        pub.publish(msg);

        // 6. Log info
        ROS_INFO("%s", msg.data.c_str());

        // 7. Spin and sleep
        ros::spinOnce();
        loop_rate.sleep();
        count++;
    }

    return 0;
}

