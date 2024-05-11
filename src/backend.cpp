#include "backend.h"
#include "algorithm.h"
#include "feature.h"
#include "g2o_types.h"
#include "ceres_types.h"
#include "map.h"
#include "mappoint.h"

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
        //Optimize(active_kfs, active_landmarks);
        //Optimize2(active_kfs, active_landmarks);
        Optimize3(active_kfs, active_landmarks);
    }
}

void Backend::Optimize2(Map::KeyframesType &keyframes,Map::LandmarksType &landmarks) {
    //todo
}

void Backend::Optimize3(Map::KeyframesType &keyframes,Map::LandmarksType &landmarks) {
    // pose 顶点，使用Keyframe id
    std::map<unsigned long, std::shared_ptr<Vec7>> poses;
    unsigned long max_kf_id = 0;
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;
        Eigen::Quaterniond q = kf->Pose().unit_quaternion();
        Vec3 t = kf->Pose().translation();
        std::shared_ptr<Vec7> pose =  std::make_shared<Vec7>();
        *pose<<q.x(),q.y(),q.z(),q.w(),t[0],t[1],t[2];
        poses.insert({kf->keyframe_id_, pose});
    }

    // 路标顶点，使用路标id索引
    std::map<unsigned long, std::shared_ptr<Vec3>> points;

    // K 和左右外参
    Mat33 K = cam_left_->K();
    SE3 left_ext = cam_left_->pose();
    SE3 right_ext = cam_right_->pose();
    SE3 camera_ext;
    ceres::LocalParameterization* local_parameterization =
            new ceres::ProductParameterization(
                    new ceres::EigenQuaternionParameterization(),
                    new ceres::IdentityParameterization(3));
    ceres::Problem problem;
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

            ceres::CostFunction *cost_function;
            // Each Residual block takes a point and a pose as input
            // and outputs a 2 dimensional Residual
            if (feat->is_on_left_image_) {
                camera_ext = left_ext;
            } else {
                camera_ext = right_ext;
            }
            cost_function = SnavelyReprojectionError::Create(feat->position_,K,camera_ext);
            // If enabled use Huber's loss function.
            ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
            auto mappoint = feat->map_point_.lock();
            problem.AddResidualBlock(cost_function, loss_function, frame->Pose().log().data(), mappoint->Pos().data());
            problem.AddParameterBlock(poses[frame->keyframe_id_]->data(),7,local_parameterization);
        }
    }
    problem.SetParameterBlockConstant(poses[0]->data());
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.FullReport() << "\n";

    // Set pose and lanrmark position
    for (auto &pose : poses) {
        std::shared_ptr<Vec7> v7 = pose.second;
        Eigen::Quaterniond q(v7->coeffRef(0),v7->coeffRef(1),v7->coeffRef(2),v7->coeffRef(3));
        Vec3 t(v7->coeffRef(4),v7->coeffRef(5),v7->coeffRef(6));
        SE3 se3(q,t);
        keyframes.at(pose.first)->SetPose(se3);
    }
    /*for (auto &v : vertices_landmarks) {
        landmarks.at(v.first)->SetPos(v.second->estimate());
    }*/
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