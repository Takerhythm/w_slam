#include <gflags/gflags.h>
#include <ros/ros.h>
#include <ros/time.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <sophus/se3.hpp>
#include "System.h"



DEFINE_string(config_file, "/home/wmh/catkin_ws/src/slam/config/euroc.yaml", "config file path");

using namespace std;

class ImageGrabber
{
private:
    slam::System* pSLAM;

public:
    ImageGrabber(slam::System* SLAM):pSLAM(SLAM){};

    void GrabStereo(const sensor_msgs::ImageConstPtr& msgLeft, const sensor_msgs::ImageConstPtr& msgRight);

};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Stereo");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    if (argc > 1)
    {
        ROS_WARN ("Arguments supplied via command line are ignored.");
    }

    const std::string& node_name = ros::this_node::getName();

    ros::NodeHandle node_handler;
    image_transport::ImageTransport image_transport(node_handler);

    std::string voc_file, settings_file;
    node_handler.param<std::string>(node_name + "/voc_file", voc_file, "file_not_set");
    node_handler.param<std::string>(node_name + "/settings_file", settings_file, "file_not_set");

    /*if (voc_file == "file_not_set" || settings_file == "file_not_set")
    {
        ROS_ERROR("Please provide voc_file and settings_file in the launch file");
        ros::shutdown();
        return 1;
    }*/

//    node_handler.param<std::string>(node_name + "/world_frame_id", world_frame_id, "map");
//    node_handler.param<std::string>(node_name + "/cam_frame_id", cam_frame_id, "camera");

    bool enable_pangolin;
    node_handler.param<bool>(node_name + "/enable_pangolin", enable_pangolin, true);


    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    slam::System::eSensor sensor_type = slam::System::STEREO;
    auto* pSLAM = new slam::System(FLAGS_config_file , voc_file, sensor_type, enable_pangolin);
    ImageGrabber igb(pSLAM);

    message_filters::Subscriber<sensor_msgs::Image> sub_img_left(node_handler, "/cam0/image_raw", 10);
    message_filters::Subscriber<sensor_msgs::Image> sub_img_right(node_handler, "/cam1/image_raw", 10);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), sub_img_left, sub_img_right);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabStereo, &igb, _1, _2));

    ros::spin();

    // Stop all threads
    //pSLAM->Shutdown();
    ros::shutdown();

    return 0;
}

//////////////////////////////////////////////////
// Functions
//////////////////////////////////////////////////

void ImageGrabber::GrabStereo(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight)
{
    ros::Time msg_time = msgLeft->header.stamp;

    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrLeft, cv_ptrRight,cv_ptrD;
    try
    {
        cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
        cv_ptrRight = cv_bridge::toCvShare(msgRight);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat left = cv_ptrLeft->image.clone();
    cv::Mat right = cv_ptrRight->image.clone();
    // ORB-SLAM3 runs in TrackStereo()
    Sophus::SE3d Tcw = pSLAM->TrackStereo(cv_ptrLeft->image, cv_ptrRight->image);
}
