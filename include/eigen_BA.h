#ifndef SLAM_EIGEN_BA_H
#define SLAM_EIGEN_BA_H

#include <utility>

#include "common_include.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "rotation.h"
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <eigen3/Eigen/SparseCholesky>

namespace slam {

struct Edge {
    //局部BA的全局位姿索引和3D点索引
    int pose_index, point_index;
    unsigned long kf_id;
    unsigned long landmark_id;
    bool is_left;
    double u,v;
    //位姿观测的第几个点
    int edge_index;
    Edge(int pose_index, int point_index, unsigned long kf_id, unsigned long landmark_id, bool is_left, double u, double v, int edge_index) :
    pose_index(pose_index), point_index(point_index), kf_id(kf_id), landmark_id(landmark_id), is_left(is_left), u(u), v(v), edge_index(edge_index) {}
};
/// eigen ba
void ComputeJacobian(const std::map<unsigned long, std::shared_ptr<Vec6>> &poses,
                     const std::map<unsigned long, std::shared_ptr<Vec3>> &points,
                     const std::map<std::pair<unsigned long, unsigned long>, Edge> &pose2points,
                     const std::vector<int> &pose_points_num,
                     const Mat33 &K, Eigen::SparseMatrix<double> &J) {
    // 待优化的位姿节点数量
    int N_poses = poses.size();
    // 3D坐标点的节点数量
    int N_points = points.size();
    std::cout<<"pose size: "<<N_poses<<std::endl;
    std::cout<<"N_points size: "<<N_points<<std::endl;
    //计算行数
    int rows = pose_points_num.size() * 2;
    std::vector<int> dp(N_poses, 0);
    std::cout << "obs_pair.second: " <<std::endl;
    for (int i = 0; i < pose_points_num.size(); i++)  {
        for (int j = i+1; j < pose_points_num.size(); j++) {
            dp[j] += pose_points_num[i] * 2;
        }
        //rows += pose_points_num[i]*2;
        std::cout <<i<<":"<< pose_points_num[i]<<", ";
    }
    std::cout << std::endl;
    //观测到的点
    J.resize(rows, 6 * N_poses+3* N_points);
    J.setZero();
    std::cout << "JacobianPose : rows(" << J.rows() <<  ") cols(" << J.cols() << ")" << std::endl;

    /*for (int i = 0; i < N_poses; i++) {
        std::cout << "dp: " << dp[i] << std::endl;
    }*/

    double fx = K(0,0);
    double fy = K(1,1);

    // 循环用的Jacobian，放在循环外部初始化
    Eigen::Matrix<double,2,6> J_ij_pose(Eigen::Matrix<double,2,6>::Zero(2, 6));
    Eigen::Matrix<double,2,3> J_ij_point(Eigen::Matrix<double,2,3>::Zero(2, 3));

    int point_index;
    int pose_index;
    for (const auto& obs_pair:pose2points) {
        // P' = R*P + t
        SE3 T = SE3::exp(*(poses.find(obs_pair.first.first)->second));
        Mat33 R_ = T.rotationMatrix();
        Vec3 t_ = T.translation();
        Vec3 P_ = R_ * (*(points.find(obs_pair.first.second)->second)) + t_;
        double x = P_(0, 0), y = P_(1, 0), z = P_(2, 0), z_2 = z * z;

        // Jacobian of pose : のf/のT
        // 只优化位姿，所以只有位姿导数，没有坐标点导数
        J_ij_pose(0, 0) = -1. / z * fx;
        J_ij_pose(0, 1) = 0;
        J_ij_pose(0, 2) = x / z_2 * fx;
        J_ij_pose(0, 3) = x * y / z_2 * fx;
        J_ij_pose(0, 4) = -(1 + (x * x / z_2)) * fx;
        J_ij_pose(0, 5) = y / z * fx;
        J_ij_pose(1, 0) = 0;
        J_ij_pose(1, 1) = -1. / z * fy;
        J_ij_pose(1, 2) = y / z_2 * fy;
        J_ij_pose(1, 3) = (1 + (y * y / z_2)) * fy;
        J_ij_pose(1, 4) = -x * y / z_2 * fy;
        J_ij_pose(1, 5) = -x / z * fy;
        //J.block<2, 6>(row_ij, 6 * i) = J_ij_pose;

        //该位姿观测到的点在landmark_indexs中的索引 全局索引 代表是第几个点
        Edge edge = obs_pair.second;
        pose_index = edge.pose_index;
        point_index = edge.point_index;
        int row_ij = dp[pose_index]+edge.edge_index*2;
        //std::cout << "row_ij: " << row_ij << std::endl;
        J.insert(row_ij, 6 * pose_index) = J_ij_pose.coeffRef(0,0);
        J.insert(row_ij, 6 * pose_index+1) = J_ij_pose.coeffRef(0,1);
        J.insert(row_ij, 6 * pose_index+2) = J_ij_pose.coeffRef(0,2);
        J.insert(row_ij, 6 * pose_index+3) = J_ij_pose.coeffRef(0,3);
        J.insert(row_ij, 6 * pose_index+4) = J_ij_pose.coeffRef(0,4);
        J.insert(row_ij, 6 * pose_index+5) = J_ij_pose.coeffRef(0,5);
        J.insert(row_ij+1, 6 * pose_index) = J_ij_pose.coeffRef(1,0);
        J.insert(row_ij+1, 6 * pose_index+1) = J_ij_pose.coeffRef(1,1);
        J.insert(row_ij+1, 6 * pose_index+2) = J_ij_pose.coeffRef(1,2);
        J.insert(row_ij+1, 6 * pose_index+3) = J_ij_pose.coeffRef(1,3);
        J.insert(row_ij+1, 6 * pose_index+4) = J_ij_pose.coeffRef(1,4);
        J.insert(row_ij+1, 6 * pose_index+5) = J_ij_pose.coeffRef(1,5);

        // Jacobian of point : のf/のp 坐标导数
        J_ij_point(0, 0) = -fx / z;
        J_ij_point(0, 1) = 0;
        J_ij_point(0, 2) = fx * x / z_2;
        J_ij_point(1, 0) = 0;
        J_ij_point(1, 1) = -fy / z;
        J_ij_point(1, 2) = fy * y / z_2;
        J_ij_point *= R_;
        //J.block<2, 3>(row_ij, 6*N_poses + 3 * point_index) = J_ij_point;
        int col_ij = 6*N_poses + 3 * point_index;
        J.insert(row_ij,col_ij) = J_ij_point.coeffRef(0, 0);
        J.insert(row_ij,col_ij + 1) = J_ij_point.coeffRef(0, 1);
        J.insert(row_ij,col_ij + 2) = J_ij_point.coeffRef(0, 2);
        J.insert(row_ij+1,col_ij) = J_ij_point.coeffRef(1, 0);
        J.insert(row_ij+1,col_ij + 1) = J_ij_point.coeffRef(1, 1);
        J.insert(row_ij+1,col_ij + 2) = J_ij_point.coeffRef(1, 2);
    }
    /*if (N_poses == 2){
        std::cout << "JacobianPose : " << std::endl << J << std::endl;
    }*/
}

void ComputeError(const std::map<unsigned long, std::shared_ptr<Vec6>> &poses,
                  const std::map<unsigned long, std::shared_ptr<Vec3>> &points,
                  const std::map<std::pair<unsigned long, unsigned long>, Edge> &pose2points,
                  const std::vector<int> &pose_points_num,
                  const Mat33 &K,const SE3 &left_ext,const SE3 &right_ext, Eigen::VectorXd &Error,double &error){
    int N_poses = poses.size();
    int rows = 0;
    std::vector<int> dp(N_poses, 0);
    for (int i = 0; i < pose_points_num.size(); i++)  {
        for (int j = i+1; j < pose_points_num.size(); j++) {
            dp[j] += pose_points_num[i] * 2;
        }
        rows += pose_points_num[i]*2;
    }
    Error = Eigen::VectorXd::Zero(rows, 1);
    std::cout << "Error rows: " << Error.rows()  << std::endl;
    SE3 ext;

    for (const auto& obs_pair:pose2points) {
        Edge edge = obs_pair.second;
        //如果在右图，则计算右图的重投影误差，需要乘以右图的外参
        if (edge.is_left ){
            ext = left_ext;
        }else{
            ext = right_ext;
        }
        // P' = R*P + t
        SE3 T = ext*SE3::exp(*(poses.find(obs_pair.first.first)->second));
        Mat33 R_ = T.rotationMatrix();
        Vec3 t_ = T.translation();
        Vec3 P_ = R_ * (*(points.find(obs_pair.first.second)->second)) + t_;
        Vec3 P_proj = K * P_;
        double x_proj = P_proj(0, 0) / P_proj(2, 0);
        double y_proj = P_proj(1, 0) / P_proj(2, 0);

        double x_err = edge.u-x_proj ;
        double y_err = edge.v-y_proj ;
        int row_i = dp[edge.pose_index]+edge.edge_index*2;
        //std::cout << "row_i: " << row_i << std::endl;
        Error(row_i,0) = x_err;
        Error(row_i+1,0) = y_err;
        error += x_err*x_err+y_err*y_err;
    }
    //std::cout << "Error: " << Error << std::endl;
}

void ComputeJacobianAndError(const std::unordered_map<unsigned long, std::shared_ptr<Vec6>> &poses,
                  const std::unordered_map<unsigned long, std::shared_ptr<Vec3>> &points,
                  const std::map<std::pair<unsigned long, unsigned long>, Edge> &pose2points,
                  const std::vector<int> &pose_points_num,
                  const Mat33 &K,const SE3 &left_ext,const SE3 &right_ext,
                  Eigen::SparseMatrix<double> &J, Eigen::VectorXd &Error,double &error){
    int N_poses = poses.size();
    int N_points = points.size();
    std::vector<int> dp(N_poses, 0);
    for (int i = 0; i < pose_points_num.size(); i++)  {
        for (int j = i+1; j < pose_points_num.size(); j++) {
            dp[j] += pose_points_num[i] * 2;
        }
    }
    int rows = pose2points.size() * 2;
    J.resize(rows, 6 * N_poses+3* N_points);
    J.setZero();
    Error = Eigen::VectorXd::Zero(rows, 1);
    error = 0;
    //std::cout << "JacobianPose : rows(" << J.rows() <<  ") cols(" << J.cols() << ")" << std::endl;

    SE3 ext;
    double fx = K(0,0);
    double fy = K(1,1);
    // 循环用的Jacobian，放在循环外部初始化
    Eigen::Matrix<double,2,6> J_ij_pose(Eigen::Matrix<double,2,6>::Zero(2, 6));
    Eigen::Matrix<double,2,3> J_ij_point(Eigen::Matrix<double,2,3>::Zero(2, 3));
    int point_index;
    int pose_index;
    for (const auto& obs_pair:pose2points) {
        Edge edge = obs_pair.second;
        //如果在右图，则计算右图的重投影误差，需要乘以右图的外参
        if (edge.is_left ){
            ext = left_ext;
        }else{
            ext = right_ext;
        }
        // P' = R*P + t
        SE3 T = ext*SE3::exp(*(poses.find(obs_pair.first.first)->second));
        Mat33 R_ = T.rotationMatrix();
        Vec3 t_ = T.translation();
        Vec3 P_ = R_ * (*(points.find(obs_pair.first.second)->second)) + t_;

        double x = P_(0, 0), y = P_(1, 0), z = P_(2, 0), z_2 = z * z;

        // Jacobian of pose : のf/のT
        // 只优化位姿，所以只有位姿导数，没有坐标点导数
        J_ij_pose(0, 0) = -1. / z * fx;
        J_ij_pose(0, 1) = 0;
        J_ij_pose(0, 2) = x / z_2 * fx;
        J_ij_pose(0, 3) = x * y / z_2 * fx;
        J_ij_pose(0, 4) = -(1 + (x * x / z_2)) * fx;
        J_ij_pose(0, 5) = y / z * fx;
        J_ij_pose(1, 0) = 0;
        J_ij_pose(1, 1) = -1. / z * fy;
        J_ij_pose(1, 2) = y / z_2 * fy;
        J_ij_pose(1, 3) = (1 + (y * y / z_2)) * fy;
        J_ij_pose(1, 4) = -x * y / z_2 * fy;
        J_ij_pose(1, 5) = -x / z * fy;
        //J.block<2, 6>(row_ij, 6 * i) = J_ij_pose;

        //该位姿观测到的点在landmark_indexs中的索引 全局索引 代表是第几个点
        pose_index = edge.pose_index;
        point_index = edge.point_index;
        int row_ij = dp[pose_index]+edge.edge_index*2;
        //std::cout << "row_ij: " << row_ij << std::endl;
        J.insert(row_ij, 6 * pose_index) = J_ij_pose.coeffRef(0,0);
        J.insert(row_ij, 6 * pose_index+1) = J_ij_pose.coeffRef(0,1);
        J.insert(row_ij, 6 * pose_index+2) = J_ij_pose.coeffRef(0,2);
        J.insert(row_ij, 6 * pose_index+3) = J_ij_pose.coeffRef(0,3);
        J.insert(row_ij, 6 * pose_index+4) = J_ij_pose.coeffRef(0,4);
        J.insert(row_ij, 6 * pose_index+5) = J_ij_pose.coeffRef(0,5);
        J.insert(row_ij+1, 6 * pose_index) = J_ij_pose.coeffRef(1,0);
        J.insert(row_ij+1, 6 * pose_index+1) = J_ij_pose.coeffRef(1,1);
        J.insert(row_ij+1, 6 * pose_index+2) = J_ij_pose.coeffRef(1,2);
        J.insert(row_ij+1, 6 * pose_index+3) = J_ij_pose.coeffRef(1,3);
        J.insert(row_ij+1, 6 * pose_index+4) = J_ij_pose.coeffRef(1,4);
        J.insert(row_ij+1, 6 * pose_index+5) = J_ij_pose.coeffRef(1,5);

        // Jacobian of point : のf/のp 坐标导数
        J_ij_point(0, 0) = -fx / z;
        J_ij_point(0, 1) = 0;
        J_ij_point(0, 2) = fx * x / z_2;
        J_ij_point(1, 0) = 0;
        J_ij_point(1, 1) = -fy / z;
        J_ij_point(1, 2) = fy * y / z_2;
        J_ij_point *= R_;
        //J.block<2, 3>(row_ij, 6*N_poses + 3 * point_index) = J_ij_point;
        int col_ij = 6*N_poses + 3 * point_index;
        J.insert(row_ij,col_ij) = J_ij_point.coeffRef(0, 0);
        J.insert(row_ij,col_ij + 1) = J_ij_point.coeffRef(0, 1);
        J.insert(row_ij,col_ij + 2) = J_ij_point.coeffRef(0, 2);
        J.insert(row_ij+1,col_ij) = J_ij_point.coeffRef(1, 0);
        J.insert(row_ij+1,col_ij + 1) = J_ij_point.coeffRef(1, 1);
        J.insert(row_ij+1,col_ij + 2) = J_ij_point.coeffRef(1, 2);


        Vec3 P_proj = K * P_;
        double x_proj = P_proj(0, 0) / P_proj(2, 0);
        double y_proj = P_proj(1, 0) / P_proj(2, 0);

        double x_err = edge.u-x_proj ;
        double y_err = edge.v-y_proj ;
        //std::cout << "row_i: " << row_i << std::endl;
        Error(row_ij,0) = x_err;
        Error(row_ij+1,0) = y_err;
        error += x_err*x_err+y_err*y_err;
    }
    //std::cout << "Error: " << Error << std::endl;
}

void ComputeUpdateError(const std::unordered_map<unsigned long, std::shared_ptr<Vec6>> &poses,
                             const std::unordered_map<unsigned long, std::shared_ptr<Vec3>> &points,
                             const std::map<std::pair<unsigned long, unsigned long>, Edge> &pose2points,
                             const Mat33 &K,const SE3 &left_ext,const SE3 &right_ext,
                             double &error){
    error = 0;
    SE3 ext;
    for (const auto& obs_pair:pose2points) {
        Edge edge = obs_pair.second;
        //如果在右图，则计算右图的重投影误差，需要乘以右图的外参
        if (edge.is_left ){
            ext = left_ext;
        }else{
            ext = right_ext;
        }
        // P' = R*P + t
        SE3 T = ext*SE3::exp(*(poses.find(obs_pair.first.first)->second));
        Mat33 R_ = T.rotationMatrix();
        Vec3 t_ = T.translation();
        Vec3 P_ = R_ * (*(points.find(obs_pair.first.second)->second)) + t_;

        Vec3 P_proj = K * P_;
        double x_proj = P_proj(0, 0) / P_proj(2, 0);
        double y_proj = P_proj(1, 0) / P_proj(2, 0);
        double x_err = edge.u-x_proj ;
        double y_err = edge.v-y_proj ;

        error += x_err*x_err+y_err*y_err;
    }
}

void ComputeDeltaX(Eigen::SparseMatrix<double> &H, const Eigen::VectorXd &Error,  Eigen::VectorXd &dx
                 ,const int NC, const int NP,const int &i,double &lambda, const Eigen::VectorXd &g) {
    dx.resize(NC*6+NP*3,1);

    for (int j = 0; j < H.cols(); ++j) {
        H.coeffRef(j,j) += lambda;
    }

    /*Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(H);
    if(solver.info()!=Eigen::Success) {
        std::cout << "decomposition failed" << std::endl;
        return;
    }
    dx = solver.solve(g);*/

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> LDLTsolver;
    LDLTsolver.compute(H.bottomRightCorner(NP*3,NP*3));
    Eigen::SparseMatrix<double> I(NP*3,NP*3);
    I.setIdentity();
    Eigen::SparseMatrix<double> C_inverse = LDLTsolver.solve(I);



    //Eigen::VectorXd ans;
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> QRsolver;
    QRsolver.compute(H.topLeftCorner(NC*6,NC*6) - H.topRightCorner(NC*6,NP*3) * C_inverse * H.bottomLeftCorner(NP*3,NC*6));
    dx.topRows(NC*6) = QRsolver.solve(g.topRows(NC*6) - H.topRightCorner(NC*6,NP*3) * C_inverse * g.bottomRows(NP*3));
    dx.bottomRows(NP*3) = C_inverse*(g.bottomRows(NP*3) - H.bottomLeftCorner(NP*3,NC*6)*dx.topRows(NC*6));


    //std::cout << "ans norm is " << dx.norm() << std::endl;
    /*if(solver.info()!=Eigen::Success) {
        std::cout << "solving failed" << std::endl;
        return;
    }*/
}

void UpdateDx(const std::unordered_map<unsigned long, std::shared_ptr<Vec6>> &poses,
               const std::unordered_map<unsigned long, int> &poses_index,
               const std::unordered_map<unsigned long, std::shared_ptr<Vec3>> &points,
               const std::unordered_map<unsigned long, int> &points_index,
                const Eigen::VectorXd &dx){
    // 待优化的位姿节点数量
    int N_poses = poses.size();
    for (const auto& pose_pair:poses) {
        //更新位姿
        int pose_index = poses_index.find(pose_pair.first)->second;
        Vec6 delta_pose = dx.block<6, 1>(6 * pose_index, 0);
        *(pose_pair.second) = (SE3::exp(*(pose_pair.second))*SE3::exp(delta_pose)).log();
    }
    for (const auto& point_pair:points) {
        //更新点
        int point_index = points_index.find(point_pair.first)->second;
        Vec3 delta_point = dx.block<3, 1>(6 * N_poses + 3 * point_index, 0);
        *(point_pair.second) += delta_point;
    }
}
}  // namespace slam

#endif  // SLAM_EIGEN_BA_H
