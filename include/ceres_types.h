#ifndef SLAM_CERES_TYPES_H
#define SLAM_CERES_TYPES_H

#include "common_include.h"
#include "ceres/ceres.h"
#include "rotation.h"


namespace slam {
/// ceres ba
class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(const cv::KeyPoint observed, const Mat33 &K, const SE3 &cam_ext) :
    observed_x(observed.pt.x),observed_y(observed.pt.y) ,_K(K),_cam_ext(cam_ext)
    {}

    template<typename T>
    bool operator()(
             const T *const v7, const T *const v3,
                    T *residuals) const {
        //pose se3
        T camera[3];
        QuaternionToAngleAxis(v7,camera);
        T p[3];
        AngleAxisRotatePoint(camera, v3, p);
        // camera[3,4,5] are the translation
        p[0] += v7[4];
        p[1] += v7[5];
        p[2] += v7[6];
        double fx = _K(0,0);
        double fy = _K(1,1);
        double cx = _K(0,2);
        double cy = _K(1,2);
        T predictions[2];
        predictions[0] = fx*p[0]/p[2]+cx;
        predictions[1] = fy*p[1]/p[2]+cy;
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);
        return true;
    }


    static ceres::CostFunction *Create(const cv::KeyPoint observed, const Mat33 &K, const SE3 &cam_ext) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3>(
                new SnavelyReprojectionError(observed,K,cam_ext)));
    }

    private:
        double observed_x;
        double observed_y;
        Mat33 _K;
        SE3 _cam_ext;
    };


}  // namespace slam

#endif  // SLAM_CERES_TYPES_H
