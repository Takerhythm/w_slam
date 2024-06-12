#ifndef SLAM_CERES_TYPES_H
#define SLAM_CERES_TYPES_H

#include <utility>

#include "common_include.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "rotation.h"


namespace slam {
/// ceres ba
class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(const double observed_x, const double observed_y, Mat33 K, const SE3 &cam_ext) :
    observed_x(observed_x),observed_y(observed_y),_K(std::move(K)),_cam_ext(cam_ext)
    {}

    template<typename T>
    bool operator()(
            const T *const q, const T *const t, const T *const point,
            T *residuals) const {

        /*Eigen::Quaterniond quaterniond;
        quaterniond<<q[0],q[1],q[2],q[3];
        Vec3 v3;
        v3<<t[0],t[1],t[2];
        Vec3 pw;
        pw<<point[0],point[1],point[2];
        SE3 se3(quaterniond,v3);
        Vec3 pos_cam = _K * (_cam_ext * se3 * pw);*/
        //pose Quaternion
        T p[3];
        //QuaternionToAngleAxis(v4, q);
        ceres::QuaternionRotatePoint(q, point, p);
        //RotatePoint
        //AngleAxisRotatePoint(camera, point, p);
        // add translation
        p[0] += t[0];
        p[1] += t[1];
        p[2] += t[2];
        //switch camera
//        double b = _cam_ext.matrix()(0,3);
//        p[0] += b;
        double fx = _K(0,0);
        double fy = _K(1,1);
        double cx = _K(0,2);
        double cy = _K(1,2);

        //predictions
        T predictions[2];
        predictions[0] = fx*p[0]/p[2]+cx;
        predictions[1] = fy*p[1]/p[2]+cy;
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);
        return true;
    }


    static ceres::CostFunction* Create(const double observed_x, const double observed_y, const Mat33 &K, const SE3 &cam_ext) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 4, 3, 3>(
                new SnavelyReprojectionError(observed_x,observed_y,K,cam_ext)));
    }

    private:
        double observed_x;
        double observed_y;
        Mat33 _K;
        SE3 _cam_ext;
    };


}  // namespace slam

#endif  // SLAM_CERES_TYPES_H
