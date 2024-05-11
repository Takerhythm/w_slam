#pragma once
#ifndef SLAM_CONFIG_H
#define SLAM_CONFIG_H

#include "common_include.h"
#include "camera.h"

namespace slam {

/**
 * 配置类，使用SetParameterFile确定配置文件
 * 然后用Get得到对应值
 * 单例模式
 */
class Config {
private:
    // private constructor makes a singleton
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;

public:
    /// 初始化，返回是否成功
    bool Init();
    std::vector<Camera::Ptr> cameras_;

    typedef std::shared_ptr<Config> Ptr;

    ~Config();  // close the file when deconstructing

    // set a new config file
    static bool SetParameterFile(const std::string &filename);

    // access the parameter values
    template <typename T>
    static T Get(const std::string &key) {
        return T(Config::config_->file_[key]);
    }

    /// get camera by id
    Camera::Ptr GetCamera(int camera_id) const {
        return cameras_.at(camera_id);
    }

    Config() {}
};
}  // namespace slam

#endif  // SLAM_CONFIG_H
