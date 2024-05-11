#include "config.h"

#include <memory>
#include "camera.h"

namespace slam {
bool Config::Init() {
    // read camera intrinsics and extrinsics
    Mat33 K;
    K << Get<float>("Camera1.fx"), 0, Get<float>("Camera1.cx"),
         0, Get<float>("Camera1.fy"), Get<float>("Camera1.cy"),
         0, 0, 1;
    float b = Get<float>("Stereo.b");
    Vec3 t1,t2;
    t1<<0,0,0;
    t2<<-b*Get<float>("Camera1.fx"),0,0;
    t1 = K.inverse()*t1;
    t2 = K.inverse()*t2;
    K = K * 0.5;
    Camera::Ptr camera0(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),t1.norm(), SE3(SO3(), t1)));
    cameras_.push_back(camera0);
    //LOG(INFO) << "Camera " << 0 << " extrinsics: " << t.transpose();
    Camera::Ptr camera1(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),t2.norm(), SE3(SO3(), t2)));
    cameras_.push_back(camera1);
//    LOG(INFO) << "Camera " << 0 << " extrinsics: " << t1.transpose();
//    LOG(INFO) << "Camera " << 0 << " extrinsics: " << camera0->pose().matrix();
    LOG(INFO) << "Camera " << 1 << " extrinsics: " << camera1->pose().matrix();
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
