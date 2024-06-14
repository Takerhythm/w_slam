#pragma once

#ifndef SLAM_CAMERA_H
#define SLAM_CAMERA_H

#include <utility>

#include "common_include.h"

namespace slam {

// Pinhole stereo camera model
class Camera {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Camera> Ptr;

    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0,
           baseline_ = 0;  // Camera intrinsics
    SE3 pose_;             // extrinsic, from stereo camera to single camera
    SE3 pose_inv_;         // inverse of extrinsics

    cv::Mat distortion_coeff_;  // distortion coefficients

    Camera();

    Camera(double fx, double fy, double cx, double cy, double baseline,
           const SE3 &pose)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose) {
        pose_inv_ = pose_.inverse();
    }

    Camera(double fx, double fy, double cx, double cy, double baseline,
           const SE3 &pose, cv::Mat distortion_coeff)
            : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose), distortion_coeff_(std::move(distortion_coeff)) {
        pose_inv_ = pose_.inverse();
    }

    SE3 pose() const { return pose_; }

    // return intrinsic matrix
    Mat33 K() const {
        Mat33 k;
        k << fx_, 0, cx_,
             0, fy_, cy_,
             0, 0, 1;
        return k;
    }

    cv::Mat K_Mat() const{
        cv::Mat k = cv::Mat::eye(3, 3, CV_64F);
        k.at<double>(0, 0) = fx_;
        k.at<double>(1, 1) = fy_;
        k.at<double>(0, 2) = cx_;
        k.at<double>(1, 2) = cy_;
        return k;
    }

    // coordinate transform: world, camera, pixel
    Vec3 world2camera(const Vec3 &p_w, const SE3 &T_c_w);

    Vec3 camera2world(const Vec3 &p_c, const SE3 &T_c_w);

    Vec2 camera2pixel(const Vec3 &p_c);

    Vec3 pixel2camera(const Vec2 &p_p, double depth = 1);

    Vec3 pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth = 1);

    Vec2 world2pixel(const Vec3 &p_w, const SE3 &T_c_w);
};

}  // namespace slam
#endif  // SLAM_CAMERA_H
