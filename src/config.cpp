#include "config.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include "camera.h"

namespace slam {
bool Config::Init() {
    // read camera intrinsics and extrinsics
    Mat33 K1,K2;
    K1 << Get<float>("Camera1.fx"), 0, Get<float>("Camera1.cx"),
         0, Get<float>("Camera1.fy"), Get<float>("Camera1.cy"),
         0, 0, 1;
    K2 << Get<float>("Camera2.fx"), 0, Get<float>("Camera2.cx"),
            0, Get<float>("Camera2.fy"), Get<float>("Camera2.cy"),
            0, 0, 1;
    std::string dataset = Get<std::string>("Dataset");
    Vec3 t1,t2;
    SO3 R1 = SO3(), R2 ;
    t1<<0,0,0;
    cv::Mat dist_coeff1 = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat dist_coeff2 = cv::Mat::zeros(1, 5, CV_64F);
    if(dataset == "KITTI"){
        float b = Get<float>("Stereo.b");
        t2<<-b*Get<float>("Camera2.fx"),0,0;
        t1 = K1.inverse() * t1;
        t2 = K2.inverse() * t2;
        R2 = SO3();
    }else if(dataset == "Euroc"){
        /*cv::Mat cv_T;
        cv_T = Get<float>("Stereo.T_c1_c2");
        // 检查矩阵大小和类型，并进行类型转换
        if (cv_T.rows != 4 || cv_T.cols != 4) {
            std::cerr << "Error: Matrix size is incorrect. Expected 4x4 but got " << cv_T.rows << "x" << cv_T.cols << std::endl;
            return false;
        }
        if (cv_T.type() != CV_64F) {
            cv_T.convertTo(cv_T, CV_64F);
        }
        cv::cv2eigen(cv_T, eigen_T);
         */
        Mat44 eigen_Tbc,eigen_cam_T12;
        eigen_Tbc << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                0.999557249008, 0.0149672133247, 0.025715529948,  -0.064676986768,
                -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                0, 0, 0, 1;
        eigen_cam_T12 << 0.999997256477797,-0.002317135723275,-0.000343393120620,0.110074137800478,
                0.002312067192432,0.999898048507103,-0.014090668452683,-0.000156612054392,
                0.000376008102320,0.014089835846691,0.999900662638081,0.000889382785432,
                0,0,0,1.0;
        SE3 T = SE3(eigen_cam_T12);
        T = T.inverse();
        R2 = T.so3();
        t2 = T.translation();
        dist_coeff1.at<double>(0, 0) = Get<float>("Camera1.k1");
        dist_coeff1.at<double>(0, 1) = Get<float>("Camera1.k2");
        dist_coeff1.at<double>(0, 2) = Get<float>("Camera1.p1");
        dist_coeff1.at<double>(0, 3) = Get<float>("Camera1.p2");
        dist_coeff2.at<double>(0, 0) = Get<float>("Camera2.k1");
        dist_coeff2.at<double>(0, 1) = Get<float>("Camera2.k2");
        dist_coeff2.at<double>(0, 2) = Get<float>("Camera2.p1");
        dist_coeff2.at<double>(0, 3) = Get<float>("Camera2.p2");
    }
    K1 = K1 * 0.5;
    K2 = K2 * 0.5;
    Camera::Ptr camera0(new Camera(K1(0, 0), K1(1, 1), K1(0, 2), K1(1, 2), t1.norm(), SE3(R1, t1), dist_coeff1));
    cameras_.push_back(camera0);
    LOG(INFO) << "Camera " << 0 << " extrinsics: " ;
    std::cout<< camera0->pose().matrix()<<std::endl;
    std::cout <<"distortion: "<<std::endl;
    std::cout <<dist_coeff1<<std::endl;
    Camera::Ptr camera1(new Camera(K2(0, 0), K2(1, 1), K2(0, 2), K2(1, 2), t2.norm(), SE3(R2, t2), dist_coeff2));
    cameras_.push_back(camera1);

    LOG(INFO) << "Camera " << 1 << " extrinsics: " ;
    std::cout<< camera1->pose().matrix()<<std::endl;
    std::cout <<"distortion: "<<std::endl;
    std::cout <<dist_coeff2<<std::endl;
    return true;
}

bool Config::SetParameterFile(const std::string &filename) {
    if (config_ == nullptr)
        config_ = std::make_shared<Config>();
    config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
    if (!config_->file_.isOpened()) {
        LOG(ERROR) << "parameter file " << filename << " does not exist.";
        config_->file_.release();
        return false;
    }
    return true;
}



Config::~Config() {
    if (file_.isOpened())
        file_.release();
}

std::shared_ptr<Config> Config::config_ = nullptr;

}
