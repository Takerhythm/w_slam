#include "sophus/se3.hpp"
#include "vertex_pose.h"
#include "edge_reprojection.h"
#include "utility.h"

#include <iostream>

namespace slam {
namespace backend {

/**
    std::vector<std::shared_ptr<Vertex>> verticies_; // 该边对应的顶点
    VecX residual_;                 // 残差
    std::vector<MatXX> jacobians_;  // 雅可比，每个雅可比维度是 residual x vertex[i]
    MatXX information_;             // 信息矩阵
    VecX observation_;              // 观测信息
*/

void EdgeReprojection::ComputeResidual() {
//    std::cout << pts_i_.transpose() <<" "<<pts_j_.transpose()  <<std::endl;

    double inv_dep_i = verticies_[0]->Parameters()[0];

    VecX param_i = verticies_[1]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();

    VecX param_j = verticies_[2]->Parameters();
    Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
    Vec3 Pj = param_j.head<3>();

    VecX param_ext = verticies_[3]->Parameters();
    Qd qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
    Vec3 tic = param_ext.head<3>();

    Vec3 pts_camera_i = pts_i_ / inv_dep_i;
    Vec3 pts_imu_i = qic * pts_camera_i + tic;
    Vec3 pts_w = Qi * pts_imu_i + Pi;
    Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);

    double dep_j = pts_camera_j.z();
    residual_ = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();   /// J^t * J * delta_x = - J^t * r
//    residual_ = information_ * residual_;   // remove information here, we multi information matrix in problem solver
}

//void EdgeReprojection::SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_) {
//    qic = qic_;
//    tic = tic_;
//}

void EdgeReprojection::ComputeJacobians() {
    double inv_dep_i = verticies_[0]->Parameters()[0];

    VecX param_i = verticies_[1]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();

    VecX param_j = verticies_[2]->Parameters();
    Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
    Vec3 Pj = param_j.head<3>();

    VecX param_ext = verticies_[3]->Parameters();
    Qd qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
    Vec3 tic = param_ext.head<3>();

    Vec3 pts_camera_i = pts_i_ / inv_dep_i;
    Vec3 pts_imu_i = qic * pts_camera_i + tic;
    Vec3 pts_w = Qi * pts_imu_i + Pi;
    Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);

    double dep_j = pts_camera_j.z();

    Mat33 Ri = Qi.toRotationMatrix();
    Mat33 Rj = Qj.toRotationMatrix();
    Mat33 ric = qic.toRotationMatrix();
    Mat23 reduce(2, 3);
    reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
        0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
//    reduce = information_ * reduce;

    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    Eigen::Matrix<double, 3, 6> jaco_i;
    jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
    jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Sophus::SO3d::hat(pts_imu_i);
    jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

    Eigen::Matrix<double, 2, 6> jacobian_pose_j;
    Eigen::Matrix<double, 3, 6> jaco_j;
    jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
    jaco_j.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_j);
    jacobian_pose_j.leftCols<6>() = reduce * jaco_j;

    Eigen::Vector2d jacobian_feature;
    jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_ * -1.0 / (inv_dep_i * inv_dep_i);

    Eigen::Matrix<double, 2, 6> jacobian_ex_pose;
    Eigen::Matrix<double, 3, 6> jaco_ex;
    jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
    Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
    jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                             Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
    jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;

    jacobians_[0] = jacobian_feature;
    jacobians_[1] = jacobian_pose_i;
    jacobians_[2] = jacobian_pose_j;
    jacobians_[3] = jacobian_ex_pose;

    ///------------- check jacobians -----------------
//    {
//        std::cout << jacobians_[0] <<std::endl;
//        const double eps = 1e-6;
//        inv_dep_i += eps;
//        Eigen::Vector3d pts_camera_i = pts_i_ / inv_dep_i;
//        Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
//        Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
//        Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
//        Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
//
//        Eigen::Vector2d tmp_residual;
//        double dep_j = pts_camera_j.z();
//        tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();
//        tmp_residual = information_ * tmp_residual;
//        std::cout <<"num jacobian: "<<  (tmp_residual - residual_) / eps <<std::endl;
//    }

}

void EdgeReprojectionXYZ::ComputeResidual() {
    VecX pose_so3 = verticies_[0]->Parameters();
    Vec3 point = verticies_[1]->Parameters();

    _pred = _K * (_cam_ext*SE3::exp(pose_so3)*point);
    residual_ = _obs.head<2>() - _pred.head<2>()/_pred[2];
    /*Qd Qi(pose_so3[6], pose_so3[3], pose_so3[4], pose_so3[5]);
    Vec3 Pi = pose_so3.head<3>();
    Vec3 pts_imu_i = Qi.inverse() * (point - Pi);
    Vec3 pts_camera_i = qic.inverse() * (pts_imu_i - tic);
    double dep_i = pts_camera_i.z();
    residual_ = (pts_camera_i / dep_i).head<2>() - _obs.head<2>();*/
}

void EdgeReprojectionXYZ::SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_) {
    qic = qic_;
    tic = tic_;
}

void EdgeReprojectionXYZ::ComputeJacobians() {
    VecX pose_so3 = verticies_[0]->Parameters();
    Vec3 point = verticies_[1]->Parameters();
    Eigen::Matrix<double,2,6> J_pose(Eigen::Matrix<double,2,6>::Zero(2, 6));
    Eigen::Matrix<double,2,3> J_point(Eigen::Matrix<double,2,3>::Zero(2, 3));

    Mat33 R_ = (_cam_ext*SE3::exp(pose_so3)).rotationMatrix();
    double x = _pred[0], y = _pred[1], z = _pred[2], z_2 = z * z;
    double fx = _K(0, 0), fy = _K(1, 1);
    // Jacobian of pose : のf/のT
    // 只优化位姿，所以只有位姿导数，没有坐标点导数
    J_pose(0, 0) = -1. / z * fx;
    J_pose(0, 1) = 0;
    J_pose(0, 2) = x / z_2 * fx;
    J_pose(0, 3) = x * y / z_2 * fx;
    J_pose(0, 4) = -(1 + (x * x / z_2)) * fx;
    J_pose(0, 5) = y / z * fx;
    J_pose(1, 0) = 0;
    J_pose(1, 1) = -1. / z * fy;
    J_pose(1, 2) = y / z_2 * fy;
    J_pose(1, 3) = (1 + (y * y / z_2)) * fy;
    J_pose(1, 4) = -x * y / z_2 * fy;
    J_pose(1, 5) = -x / z * fy;

    // Jacobian of point : のf/のp 坐标导数
    J_point(0, 0) = -fx / z;
    J_point(0, 1) = 0;
    J_point(0, 2) = fx * x / z_2;
    J_point(1, 0) = 0;
    J_point(1, 1) = -fy / z;
    J_point(1, 2) = fy * y / z_2;
    J_point *= R_;

    jacobians_[0] = J_pose;
    jacobians_[1] = J_point;
    /*Qd Qi(pose_so3[6], pose_so3[3], pose_so3[4], pose_so3[5]);
    Vec3 Pi = pose_so3.head<3>();
    Vec3 pts_imu_i = Qi.inverse() * (point - Pi);
    Vec3 pts_camera_i = qic.inverse() * (pts_imu_i - tic);
    double dep_i = pts_camera_i.z();
    Mat33 Ri = Qi.toRotationMatrix();
    Mat33 ric = qic.toRotationMatrix();
    Mat23 reduce(2, 3);
    reduce << 1. / dep_i, 0, -pts_camera_i(0) / (dep_i * dep_i),
        0, 1. / dep_i, -pts_camera_i(1) / (dep_i * dep_i);
    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    Eigen::Matrix<double, 3, 6> jaco_i;
    jaco_i.leftCols<3>() = ric.transpose() * -Ri.transpose();
    jaco_i.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_i);
    jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
    Eigen::Matrix<double, 2, 3> jacobian_feature;
    jacobian_feature = reduce * ric.transpose() * Ri.transpose();
    jacobians_[0] = jacobian_feature;
    jacobians_[1] = jacobian_pose_i;*/

}

void EdgeReprojectionPoseOnly::ComputeResidual() {
    VecX pose_params = verticies_[0]->Parameters();
    Sophus::SE3d pose(
        Qd(pose_params[6], pose_params[3], pose_params[4], pose_params[5]),
        pose_params.head<3>()
    );

    Vec3 pc = pose * landmark_world_;
    pc = pc / pc[2];
    Vec2 pixel = (K_ * pc).head<2>() - observation_;
    // TODO:: residual_ = ????
    residual_ = pixel;
}

void EdgeReprojectionPoseOnly::ComputeJacobians() {
    // TODO implement jacobian here
}

}
}