#include "backend.h"
#include "algorithm.h"
#include "feature.h"
#include "g2o_types.h"
#include "ceres_types.h"
#include "map.h"
#include "mappoint.h"
#include "eigen_BA.h"
#include "problem.h"
#include "vertex_pose.h"
#include "vertex_point.h"
#include "edge_reprojection.h"

namespace slam {

Backend::Backend() {
    backend_running_.store(true);
    backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

void Backend::UpdateMap() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    map_update_.notify_one();
}

void Backend::Stop() {
    backend_running_.store(false);
    map_update_.notify_one();
    backend_thread_.join();
}

void Backend::BackendLoop() {
    while (backend_running_.load()) {
        std::unique_lock<std::mutex> lock(data_mutex_);
        map_update_.wait(lock);

        /// 后端仅优化激活的Frames和Landmarks
        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
        //g2o优化
        Optimize(active_kfs, active_landmarks);
        //手写优化
        //Optimize2(active_kfs, active_landmarks);
        //高翔贺一家版本手写优化
        //Optimize4(active_kfs, active_landmarks);
        //ceres优化
        //Optimize3(active_kfs, active_landmarks);
    }
}

void Backend::Optimize2(Map::KeyframesType &keyframes,Map::LandmarksType &landmarks) {
    //todo 优化发散 不知道哪里出问题
    std::unordered_map<unsigned long, std::shared_ptr<Vec6>> poses;
    std::unordered_map<unsigned long, int> poses_index;
    int pose_index = 0;
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;
        std::shared_ptr<Vec6> pose = std::make_shared<Vec6>();
        *pose << kf->Pose().log();
        //添加位姿待优化变量
        poses.insert({kf->keyframe_id_,pose});
        poses_index.insert({kf->keyframe_id_,pose_index});
        pose_index++;
    }
    // K 和左右外参
    Mat33 K = cam_left_->K();
    SE3 left_ext = cam_left_->pose();
    SE3 right_ext = cam_right_->pose();
    SE3 camera_ext;
    std::map<std::pair<unsigned long, unsigned long>, Edge> pose2points;
    std::unordered_map<unsigned long, std::shared_ptr<Vec3>> points;
    std::unordered_map<unsigned long, int> points_index;
    //每个位姿观测的路标点数
    std::vector<int> pose_points_num(poses.size(), 0);

    //每个点的索引
    int point_index = 0;
    for (auto &landmark : landmarks) {
        if (landmark.second->is_outlier_) continue;
        unsigned long landmark_id = landmark.second->id_;
        std::shared_ptr<Vec3> point = std::make_shared<Vec3>();
        *point << landmark.second->Pos();
        //添加路标点待优化变量
        points.insert({landmark_id,point});
        points_index.insert({landmark_id, point_index});
        auto observations = landmark.second->GetObs();
        for (auto &obs : observations) {
            if (obs.lock() == nullptr) continue;
            auto feat = obs.lock();
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;
            auto frame = feat->frame_.lock();
            //添加位姿对路标点的观测 只添加一次
            if (pose2points.find({frame->keyframe_id_, landmark_id}) == pose2points.end()){
                int edge_index = pose_points_num[poses_index[frame->keyframe_id_]]++;
                Edge edge(poses_index[frame->keyframe_id_], points_index[landmark_id], frame->keyframe_id_, landmark_id,
                          feat->is_on_left_image_,  feat->position_.pt.x,feat->position_.pt.y,edge_index);
                pose2points.insert({std::make_pair(frame->keyframe_id_, landmark_id), edge});
            }
        }
        point_index++;
    }
    int N = 10;
    double cur_error, pre_error, lambda=0,rho,ni=2;
    Eigen::SparseMatrix<double> J,H;
    Eigen::VectorXd Error,g;
    Eigen::VectorXd dx;

    //计算雅克比和重投影误差
    ComputeJacobianAndError(poses, points, pose2points, pose_points_num, K, left_ext, right_ext, J, Error, pre_error);
    H = J.transpose() * J;
    g = -J.transpose() * Error;
    //计算初始lambda
    for (int j = 0; j < H.cols(); ++j) {
        lambda = std::max(lambda, fabs(H.coeffRef(j,j)));
    }
    lambda = std::min(5e10, lambda);
    lambda *= 1e-5;

    bool isUpdate = true;
    for (int i = 0; i < N; ++i) {
        //求解增量
        bool oneStepSuccess = false;
        int false_cnt = 0;
        //备份当前状态
        std::unordered_map<unsigned long, std::shared_ptr<Vec6>> poses_bak;
        std::unordered_map<unsigned long, std::shared_ptr<Vec3>> points_bak;
        for(auto &pose : poses){
            //深拷贝位姿
            std::shared_ptr<Vec6> pose_bak = std::make_shared<Vec6>();
            *pose_bak << *pose.second;
            //添加位姿待优化变量
            poses_bak.insert({pose.first,pose_bak});
        }
        for(auto &point : points){
            //深拷贝路标点
            points_bak.insert({point.first,std::make_shared<Vec3>(*point.second)});
        }
        // 不断尝试 Lambda, 直到成功迭代一步 连续失败10次 则放弃
        while (!oneStepSuccess && false_cnt < 10){
            //std::cout << "Iteration " << i << " lambda is " << lambda << std::endl;
            ComputeDeltaX(H, Error, dx, poses.size(), points.size(), i, lambda, g);
            //更新dx
            UpdateDx(poses,poses_index,points, points_index, dx);
            //计算更新后的误差
            ComputeUpdateError(poses, points, pose2points, K, left_ext, right_ext, cur_error);
            double scale = 0.5*dx.transpose()*(lambda*dx+g)+1e-18;
            double tmp_error = pre_error-cur_error;
            rho = 0.5*(tmp_error)/scale;
            //更新lambda
            if(rho>0 && isfinite(tmp_error)){
                //误差减小 缩小lambda
                double alpha = 1. - pow((2 * rho - 1), 3);
                alpha = std::min(alpha, 2. / 3.);
                double scaleFactor = (std::max)(1. / 3., alpha);
                lambda *= scaleFactor;
                ni = 2;
                oneStepSuccess = true;
                ComputeJacobianAndError(poses, points, pose2points, pose_points_num, K, left_ext, right_ext, J, Error, pre_error);
                H = J.transpose() * J;
                g = -J.transpose() * Error;
            }else{
                lambda *= ni;
                ni *= 2;
                oneStepSuccess = false;
                false_cnt++;
                //回滚更新
                poses = poses_bak;
                points = points_bak;
            }
        }

        if (fabs(pre_error - cur_error) < 1e-5) {
            std::cout << "Iteration " << i << " error is " << pre_error << "->" << cur_error <<".lanmda is "<<lambda<< std::endl;
            std::cout << "dx norm is " << dx.norm() << ".failure count is "<<false_cnt<< std::endl;
            break;
        }else if(cur_error*0.001>pre_error&&dx.norm()<1e-5){
            isUpdate = false;
        }
    }
    if(isUpdate){
        //更新位姿和路标点
        for (auto &pose : poses) {
            keyframes.at(pose.first)->SetPose(SE3::exp(*(pose.second)));
        }
        for (auto &point : points) {
            landmarks.at(point.first)->SetPos(*(point.second));
        }
    }
}

void Backend::Optimize4(Map::KeyframesType &keyframes,Map::LandmarksType &landmarks) {
    backend::Problem problem;
    int pose_dim = 0;
    std::unordered_map<unsigned long,std::shared_ptr<backend::VertexPose>> vertexCamPoses;
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;
        std::shared_ptr<backend::VertexPose> vertex_pose(new backend::VertexPose());
        vertex_pose->SetParameters(kf->Pose().log());
        vertexCamPoses.insert({kf->keyframe_id_, vertex_pose});
        problem.AddVertex(vertex_pose);
        pose_dim += 6;
    }
    // K 和左右外参
    Mat33 K = cam_left_->K();
    SE3 left_ext = cam_left_->pose();
    SE3 right_ext = cam_right_->pose();
    SE3 camera_ext;
    std::unordered_map<unsigned long,std::shared_ptr<backend::VertexPoint>> vertexPoints;
    for (auto &landmark : landmarks) {
        if (landmark.second->is_outlier_) continue;
        unsigned long landmark_id = landmark.second->id_;
        std::shared_ptr<backend::VertexPoint> vertex_point(new backend::VertexPoint());
        vertex_point->SetParameters(landmark.second->Pos());
        problem.AddVertex(vertex_point);
        vertexPoints.insert({landmark_id, vertex_point});
        auto observations = landmark.second->GetObs();
        for (auto &obs : observations) {
            if (obs.lock() == nullptr) continue;
            auto feat = obs.lock();
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;
            auto frame = feat->frame_.lock();
            if (feat->is_on_left_image_) {
                camera_ext = left_ext;
            } else {
                camera_ext = right_ext;
            }
            std::shared_ptr<backend::EdgeReprojectionXYZ> edge(new backend::EdgeReprojectionXYZ(toVec2(feat->position_.pt),K,camera_ext));
            std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
            edge_vertex.push_back(vertexCamPoses[frame->keyframe_id_]);
            edge_vertex.push_back(vertex_point);
            edge->SetVertex(edge_vertex);
            //edge->SetInformation(Mat22::Identity());
            //edge->SetLossFunction(nullptr);
            problem.AddEdge(edge);
        }
    }
    //todo 先验

    problem.Solve(10);

    // 更新位姿和路标点
    for (auto &pose : vertexCamPoses) {
        keyframes.at(pose.first)->SetPose(SE3::exp(pose.second->Parameters()));
    }
    for (auto &point : vertexPoints) {
        landmarks.at(point.first)->SetPos(point.second->Parameters());
    }
}

void Backend::Optimize3(Map::KeyframesType &keyframes,Map::LandmarksType &landmarks) {
    // pose 顶点，使用Keyframe id
    /*ceres::LocalParameterization* local_parameterization =
            new ceres::ProductParameterization(
                    new ceres::EigenQuaternionParameterization(),
                    new ceres::IdentityParameterization(3));*/
    ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
    ceres::Problem problem;
    std::map<unsigned long, std::shared_ptr<Vec4>> quaternions;
    std::map<unsigned long, std::shared_ptr<Vec3>> translations;
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;
        Eigen::Quaterniond q = kf->Pose().unit_quaternion();
        Vec3 t = kf->Pose().translation();
        std::shared_ptr<Vec4> quaternion =  std::make_shared<Vec4>();
        std::shared_ptr<Vec3> translation =  std::make_shared<Vec3>();
        *quaternion<<q.w(),q.x(),q.y(),q.z();
        *translation<<t[0],t[1],t[2];
        quaternions.insert({kf->keyframe_id_, quaternion});
        translations.insert({kf->keyframe_id_, translation});
        problem.AddParameterBlock(quaternions[kf->keyframe_id_]->data(), 4, local_parameterization);
        problem.AddParameterBlock(translations[kf->keyframe_id_]->data(), 3);
    }
    // 路标顶点，使用路标id索引
    std::map<unsigned long, std::shared_ptr<Vec3>> points;

    // K 和左右外参
    Mat33 K = cam_left_->K();
    SE3 left_ext = cam_left_->pose();
    SE3 right_ext = cam_right_->pose();
    SE3 camera_ext;
    for (auto &landmark : landmarks) {
        if (landmark.second->is_outlier_) continue;
        unsigned long landmark_id = landmark.second->id_;
        std::shared_ptr<Vec3> point = std::make_shared<Vec3>();
        *point << landmark.second->Pos();
        points.insert({landmark_id,point});
        auto observations = landmark.second->GetObs();
        for (auto &obs : observations) {
            if (obs.lock() == nullptr) continue;
            auto feat = obs.lock();
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;
            auto frame = feat->frame_.lock();
            // Each Residual block takes a point and a pose as input
            // and outputs a 2 dimensional Residual
            if (feat->is_on_left_image_) {
                camera_ext = left_ext;
            } else {
                camera_ext = right_ext;
            }
            //std::cout<<feat->position_.pt<<std::endl;
            ceres::CostFunction *cost_function = SnavelyReprojectionError::Create(feat->position_.pt.x,feat->position_.pt.y,K,camera_ext);
            // If enabled use Huber's loss function.
            ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
            auto mappoint = feat->map_point_.lock();
            problem.AddResidualBlock(cost_function, loss_function, quaternions[frame->keyframe_id_]->data(),translations[frame->keyframe_id_]->data(), points[landmark_id]->data());
        }
    }
    //problem.SetParameterBlockConstant(quaternions[0]->data());
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.FullReport() << "\n";
    // Set pose and landmark position
    for (auto &pose : quaternions) {
        std::shared_ptr<Vec4> v4 = pose.second;
        Eigen::Quaterniond q(v4->coeffRef(0), v4->coeffRef(1), v4->coeffRef(2), v4->coeffRef(3));
        std::shared_ptr<Vec3> v3 = translations.at(pose.first);
        SE3 se3(q,*v3);
        keyframes.at(pose.first)->SetPose(se3);
    }
    for (auto &point : points) {
        landmarks.at(point.first)->SetPos(*(point.second));
    }
}

void Backend::Optimize(Map::KeyframesType &keyframes,
                       Map::LandmarksType &landmarks) {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // pose 顶点，使用Keyframe id
    std::map<unsigned long, VertexPose *> vertices;
    unsigned long max_kf_id = 0;
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;
        VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
        vertex_pose->setId(kf->keyframe_id_);
        vertex_pose->setEstimate(kf->Pose());
        optimizer.addVertex(vertex_pose);
        if (kf->keyframe_id_ > max_kf_id) {
            max_kf_id = kf->keyframe_id_;
        }

        vertices.insert({kf->keyframe_id_, vertex_pose});
    }

    // 路标顶点，使用路标id索引
    std::map<unsigned long, VertexXYZ *> vertices_landmarks;

    // K 和左右外参
    Mat33 K = cam_left_->K();
    SE3 left_ext = cam_left_->pose();
    SE3 right_ext = cam_right_->pose();

    // edges
    int index = 1;
    double chi2_th = 5.991;  // robust kernel 阈值
    std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

    for (auto &landmark : landmarks) {
        if (landmark.second->is_outlier_) continue;
        unsigned long landmark_id = landmark.second->id_;
        auto observations = landmark.second->GetObs();
        for (auto &obs : observations) {
            if (obs.lock() == nullptr) continue;
            auto feat = obs.lock();
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;

            auto frame = feat->frame_.lock();
            EdgeProjection *edge = nullptr;
            if (feat->is_on_left_image_) {
                edge = new EdgeProjection(K, left_ext);
            } else {
                edge = new EdgeProjection(K, right_ext);
            }

            // 如果landmark还没有被加入优化，则新加一个顶点
            if (vertices_landmarks.find(landmark_id) ==
                vertices_landmarks.end()) {
                VertexXYZ *v = new VertexXYZ;
                v->setEstimate(landmark.second->Pos());
                v->setId(landmark_id + max_kf_id + 1);
                v->setMarginalized(true);
                vertices_landmarks.insert({landmark_id, v});
                optimizer.addVertex(v);
            }

            edge->setId(index);
            edge->setVertex(0, vertices.at(frame->keyframe_id_));    // pose
            edge->setVertex(1, vertices_landmarks.at(landmark_id));  // landmark
            edge->setMeasurement(toVec2(feat->position_.pt));
            edge->setInformation(Mat22::Identity());
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(chi2_th);
            edge->setRobustKernel(rk);
            edges_and_features.insert({edge, feat});

            optimizer.addEdge(edge);

            index++;
        }
    }

    // do optimization and eliminate the outliers
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5) {
        cnt_outlier = 0;
        cnt_inlier = 0;
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_features) {
            if (ef.first->chi2() > chi2_th) {
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        if (inlier_ratio > 0.5) {
            break;
        } else {
            chi2_th *= 2;
            iteration++;
        }
    }

    for (auto &ef : edges_and_features) {
        if (ef.first->chi2() > chi2_th) {
            ef.second->is_outlier_ = true;
            // remove the observation
            ef.second->map_point_.lock()->RemoveObservation(ef.second);
        } else {
            ef.second->is_outlier_ = false;
        }
    }

    LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
              << cnt_inlier;

    // Set pose and lanrmark position
    for (auto &v : vertices) {
        keyframes.at(v.first)->SetPose(v.second->estimate());
    }
    for (auto &v : vertices_landmarks) {
        landmarks.at(v.first)->SetPos(v.second->estimate());
    }
}

}  // namespace slam