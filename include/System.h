#pragma once
#ifndef SLAM_SYSTEM_H
#define SLAM_SYSTEM_H

#include "backend.h"
#include "common_include.h"
//#include "dataset.h"
#include "frontend.h"
#include "viewer.h"
#include "config.h"

namespace slam {

/**
 * VO 对外接口
 */
class System {
public:
    enum eSensor{
        NOT_SET=-1,
        MONOCULAR=0,
        STEREO=1,
        RGBD=2,
        IMU_MONOCULAR=3,
        IMU_STEREO=4,
        IMU_RGBD=5,
    };


public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /// constructor with config file
    System(std::string &config_path, std::string &voc_file_path, eSensor sensor_type,
           bool enable_pangolin);

    typedef std::shared_ptr<System> Ptr;

    /**
     * do initialization things before run
     * @return true if success
     */
    bool Init();

    /**
     * start vo in the dataset
     */
    void Run();

    /**
     * Make a step forward in dataset
     */
    bool Step();

    Sophus::SE3d TrackStereo(const Mat& left, const Mat& right);

    /// 获取前端状态
    FrontendStatus GetFrontendStatus() const { return frontend_->GetStatus(); }

   private:
    bool inited_ = false;
    std::string config_file_path_{};
    std::string voc_file_path_{};
    eSensor sensor_type_;
    bool enable_pangolin_;

    Frontend::Ptr frontend_ = nullptr;
    Backend::Ptr backend_ = nullptr;
    Map::Ptr map_ = nullptr;
    Viewer::Ptr viewer_ = nullptr;
    Config::Ptr config_ = nullptr;

    // dataset
    //Dataset::Ptr dataset_ = nullptr;
};
}  // namespace slam

#endif  // SLAM_SYSTEM_H
