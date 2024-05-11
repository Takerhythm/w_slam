#include <chrono>
#include <memory>
#include <string>
#include "System.h"
#include "config.h"
#include "common_include.h"

namespace slam {

bool System::Init() {
    // read from config file
    if (!Config::SetParameterFile(config_file_path_)) {
        std::cout<<"!Config::SetParameterFile(config_file_path_)"<<std::endl;
        return false;
    }

    /*dataset_ = Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));
    CHECK_EQ(dataset_->Init(), true);*/

    // create components and links
    frontend_ = std::make_shared<Frontend>();
    backend_ = std::make_shared<Backend>();
    map_ = std::make_shared<Map>();
    viewer_ = std::make_shared<Viewer>();
    config_ = std::make_shared<Config>();
    config_->Init();

    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    frontend_->SetViewer(viewer_);
    frontend_->SetCameras(config_->GetCamera(0), config_->GetCamera(1));

    backend_->SetMap(map_);
    backend_->SetCameras(config_->GetCamera(0), config_->GetCamera(1));

    viewer_->SetMap(map_);

    std::cout<<"wait data"<<std::endl;
    return true;
}

/*void System::Run() {
    while (1) {
        LOG(INFO) << "VO is running";
        if (Step() == false) {
            break;
        }
    }

    backend_->Stop();
    viewer_->Close();

    LOG(INFO) << "VO exit";
}

bool System::Step() {
    Frame::Ptr new_frame = dataset_->NextFrame();
    if (new_frame == nullptr) return false;

    auto t1 = std::chrono::steady_clock::now();
    bool success = frontend_->AddFrame(new_frame);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
    return success;
}*/

System::System(std::string &config_path, std::string &voc_file_path, System::eSensor sensor_type,bool enable_pangolin):
config_file_path_(config_path)
{
    Init();
}

Sophus::SE3d System::TrackStereo(const Mat& left, const Mat& right) {
    auto new_frame = frontend_->createFrame(left, right);
    auto t1 = std::chrono::steady_clock::now();
    Sophus::SE3d Tcw = frontend_->AddFrame(new_frame);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
    return Tcw;
}

}  // namespace slam
