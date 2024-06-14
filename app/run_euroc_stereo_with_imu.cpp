#include <gflags/gflags.h>
#include <ros/ros.h>
#include <ros/time.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Imu.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <sophus/se3.hpp>
#include <queue>
#include "System.h"



DEFINE_string(config_file, "/home/wmh/catkin_ws/src/slam/config/euroc.yaml", "config file path");

using namespace std;

class ImuGrabber
{
public:
    ImuGrabber(){};

    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
private:
    slam::System* pSLAM;
public:
    ImageGrabber(ImuGrabber *pImuGb,slam::System* SLAM): mpImuGb(pImuGb), pSLAM(SLAM){}

    void GrabImageLeft(const sensor_msgs::ImageConstPtr& msg);
    void GrabImageRight(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf;
    std::mutex mBufMutexLeft,mBufMutexRight;
    ImuGrabber *mpImuGb;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Stereo_Inertial");
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
    bool enable_pangolin;
    node_handler.param<bool>(node_name + "/enable_pangolin", enable_pangolin, true);


    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    slam::System::eSensor sensor_type = slam::System::IMU_STEREO;
    auto* pSLAM = new slam::System(FLAGS_config_file , voc_file, sensor_type, enable_pangolin);
    ImuGrabber imugb;
    ImageGrabber igb(&imugb,pSLAM);

    ros::Subscriber sub_imu = node_handler.subscribe("/imu", 1000, &ImuGrabber::GrabImu, &imugb);
    ros::Subscriber sub_img_left = node_handler.subscribe("/camera/left/image_raw", 100, &ImageGrabber::GrabImageLeft, &igb);
    ros::Subscriber sub_img_right = node_handler.subscribe("/camera/right/image_raw", 100, &ImageGrabber::GrabImageRight, &igb);

    std::thread sync_thread(&ImageGrabber::SyncWithImu, &igb);

    ros::spin();
    // Stop all threads
    //pSLAM->Shutdown();
    ros::shutdown();

    return 0;
}

//////////////////////////////////////////////////
// Functions
//////////////////////////////////////////////////


void ImageGrabber::GrabImageLeft(const sensor_msgs::ImageConstPtr &img_msg)
{
    mBufMutexLeft.lock();
    if (!imgLeftBuf.empty())
        imgLeftBuf.pop();
    imgLeftBuf.push(img_msg);
    mBufMutexLeft.unlock();
}

void ImageGrabber::GrabImageRight(const sensor_msgs::ImageConstPtr &img_msg)
{
    mBufMutexRight.lock();
    if (!imgRightBuf.empty())
        imgRightBuf.pop();
    imgRightBuf.push(img_msg);
    mBufMutexRight.unlock();
}


void ImageGrabber::SyncWithImu()
{
    const double maxTimeDiff = 0.01;
    while(1)
    {
        cv::Mat imLeft, imRight;
        double tImLeft = 0, tImRight = 0;
        if (!imgLeftBuf.empty()&&!imgRightBuf.empty()&&!mpImuGb->imuBuf.empty())
        {
            tImLeft = imgLeftBuf.front()->header.stamp.toSec();
            tImRight = imgRightBuf.front()->header.stamp.toSec();

            this->mBufMutexRight.lock();
            while((tImLeft-tImRight)>maxTimeDiff && imgRightBuf.size()>1)
            {
                imgRightBuf.pop();
                tImRight = imgRightBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexRight.unlock();

            this->mBufMutexLeft.lock();
            while((tImRight-tImLeft)>maxTimeDiff && imgLeftBuf.size()>1)
            {
                imgLeftBuf.pop();
                tImLeft = imgLeftBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexLeft.unlock();

            if((tImLeft-tImRight)>maxTimeDiff || (tImRight-tImLeft)>maxTimeDiff)
            {
                // std::cout << "big time difference" << std::endl;
                continue;
            }
            if(tImLeft>mpImuGb->imuBuf.back()->header.stamp.toSec())
                continue;

            this->mBufMutexLeft.lock();
            imLeft = GetImage(imgLeftBuf.front());
            ros::Time msg_time = imgLeftBuf.front()->header.stamp;
            imgLeftBuf.pop();
            this->mBufMutexLeft.unlock();

            this->mBufMutexRight.lock();
            imRight = GetImage(imgRightBuf.front());
            imgRightBuf.pop();
            this->mBufMutexRight.unlock();

            //三维向量
            vector<IMU::Point> vImuMeas;
            Eigen::Vector3f Wbb;
            mpImuGb->mBufMutex.lock();
            if(!mpImuGb->imuBuf.empty())
            {
                // Load imu measurements from buffer
                vImuMeas.clear();
                double time_offset = 0.0;
                while(!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec()<=tImLeft-time_offset)
                {
                    double t = mpImuGb->imuBuf.front()->header.stamp.toSec();

                    cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);

                    cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);

                    vImuMeas.push_back(IMU::Point(acc,gyr,t));

                    Wbb << mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z;

                    mpImuGb->imuBuf.pop();
                }
            }
            mpImuGb->mBufMutex.unlock();

            // ORB-SLAM3 runs in TrackStereo()
            Sophus::SE3f Tcw = pSLAM->TrackStereoIMU(imLeft,imRight,tImLeft,vImuMeas);


            std::chrono::milliseconds tSleep(1);
            std::this_thread::sleep_for(tSleep);
        }
    }
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    mBufMutex.lock();
    imuBuf.push(imu_msg);
    mBufMutex.unlock();
}