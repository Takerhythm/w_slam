//
// Created by gaoxiang on 19-5-2.
//

#include <memory>
#include <opencv2/opencv.hpp>

#include <utility>
#include "algorithm.h"
#include "backend.h"
#include "config.h"
#include "feature.h"
#include "frontend.h"
#include "g2o_types.h"
#include "map.h"
#include "viewer.h"

namespace slam {

Frontend::Frontend() {
    gftt_ = cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
    num_features_tracking_bad_ = Config::Get<int>("num_features_tracking_bad");

    std::cout << "num_features_tracking_bad: " << num_features_tracking_bad_ << std::endl;
}

Frame::Ptr Frontend::createFrame(const Mat& left, const Mat& right){
    //resize
    cv::Mat image_left_resized, image_right_resized;
    cv::resize(left, image_left_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);
    cv::resize(right, image_right_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);

    auto new_frame = Frame::CreateFrame();
    new_frame->left_img_ = image_left_resized;
    new_frame->right_img_ = image_right_resized;
    current_image_index_++;
    return new_frame;
}

Sophus::SE3d Frontend::AddFrame(slam::Frame::Ptr frame) {
    current_frame_ = std::move(frame);

    //todo 联合IMU初始化 帧数和窗口大小一致时 初始化IMU
    //1.第一帧双目初始化
    //2.连续跟踪 同时预积分
    //3.到达窗口大小 IMU初始化
    //4.超过窗口大小 边缘化 滑动窗口
    //预积分
    //preintegration()

    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }

    last_frame_ = current_frame_;
    return current_frame_->Pose();
}

bool Frontend::Track() {
    if (last_frame_) {
        //恒速运动
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
    }

    TrackLastFrame();
    //todo euroc 跟踪有问题 内点数太少
    tracking_inliers_ = EstimateCurrentPose();

    /*std::vector<cv::Point2d> pts2D;
    std::vector<cv::Point3d>pts3D;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        if (mp) {
            pts2D.emplace_back(current_frame_->features_left_[i]->position_.pt);
            pts3D.emplace_back(cv::Point3d(mp->Pos().x(), mp->Pos().y(), mp->Pos().z()));
        }
    }
    cv::Mat rvec, t;

    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts3D, pts2D, camera_left_->K_Mat(), camera_left_->distortion_coeff_, rvec, t, 1);
    Vec3 so3{rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)};
    SO3 R = SO3::exp(so3);
    Vec3 T_pnp{t.at<double>(0), t.at<double>(1), t.at<double>(2)};
    SE3 T(R, T_pnp);
    current_frame_->SetPose(T);
    tracking_inliers_ = pts2D.size();*/

    if (tracking_inliers_ > num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // lost
        status_ = FrontendStatus::LOST;
    }

    InsertKeyframe();
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
    return true;
}

bool Frontend::InsertKeyframe() {
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // still have enough features, don't insert keyframe
        return false;
    }
    // current frame is a new keyframe
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    //对关键帧的特征点对应的地图点加入对该特征点的观测
    SetObservationsForKeyFrame();
    //对关键帧检测新的特征点并三角化
    DetectFeatures();  // detect new features

    // track in right image
    FindFeaturesInRight();

    if (camera_left_->distortion_coeff_.at<double>(0)!=0){
        undistortPoints();
    }

    // triangulate map points
    TriangulateNewPoints();
    // update backend because we have a new keyframe
    //后端优化 未使用滑动窗口
    backend_->UpdateMap();

    if (viewer_) viewer_->UpdateMap();

    return true;
}
void Frontend::undistortPoints(){
    for (auto &cur_point : current_frame_->features_left_) {
        Vec2 p = Vec2(cur_point->position_.pt.x, cur_point->position_.pt.y);
        Vec2 P;
        liftProjective(p, P,true);
        cur_point->position_.pt.x = P[0];
        cur_point->position_.pt.y = P[1];
    }
    for (auto &cur_point : current_frame_->features_right_) {
        if(!cur_point){
            continue;
        }
        Vec2 p = Vec2(cur_point->position_.pt.x, cur_point->position_.pt.y);
        Vec2 P;
        liftProjective(p, P,false);
        cur_point->position_.pt.x = P[0];
        cur_point->position_.pt.y = P[1];
    }
}


void Frontend::liftProjective(const Eigen::Vector2d& p, Eigen::Vector2d& P,const bool is_left) const{
    double mx_d, my_d, mx_u, my_u;
    double fx, fy, cx, cy;
    double m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23;
    double k1 ,k2 ,p1 ,p2 ;
    if (is_left){
        fx = camera_left_->K()(0, 0);
        fy = camera_left_->K()(1, 1);
        cx = camera_left_->K()(0, 2);
        cy = camera_left_->K()(1, 2);

        k1 = camera_left_->distortion_coeff_.at<double>(0);
        k2 = camera_left_->distortion_coeff_.at<double>(1);
        p1 = camera_left_->distortion_coeff_.at<double>(2);
        p2 = camera_left_->distortion_coeff_.at<double>(3);
    }else{
        fx = camera_right_->K()(0, 0);
        fy = camera_right_->K()(1, 1);
        cx = camera_right_->K()(0, 2);
        cy = camera_right_->K()(1, 2);

        k1 = camera_right_->distortion_coeff_.at<double>(0);
        k2 = camera_right_->distortion_coeff_.at<double>(1);
        p1 = camera_right_->distortion_coeff_.at<double>(2);
        p2 = camera_right_->distortion_coeff_.at<double>(3);
    }
    //1/fx
    m_inv_K11 = 1.0 / fx;
    //-cx/fx
    m_inv_K13 = -cx / fx;
    //1/fy
    m_inv_K22 = 1.0 / fy;
    //-cy/fy
    m_inv_K23 = -cy / fy;
    // Lift points to normalised plane
    //归一化平面坐标
    mx_d = m_inv_K11 * p(0) + m_inv_K13;
    my_d = m_inv_K22 * p(1) + m_inv_K23;

    double mx2_d, my2_d, mxy_d, rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;
    mx2_d = mx_d*mx_d;
    my2_d = my_d*my_d;
    mxy_d = mx_d*my_d;
    rho2_d = mx2_d+my2_d;
    rho4_d = rho2_d*rho2_d;
    radDist_d = k1*rho2_d+k2*rho4_d;
    /*Dx_d = mx_d*radDist_d + p2*(rho2_d+2*mx2_d) + 2*p1*mxy_d;
    Dy_d = my_d*radDist_d + p1*(rho2_d+2*my2_d) + 2*p2*mxy_d;
    inv_denom_d = 1/(1+4*k1*rho2_d+6*k2*rho4_d+8*p1*my_d+8*p2*mx_d);

    mx_u = mx_d - inv_denom_d*Dx_d;
    my_u = my_d - inv_denom_d*Dy_d;*/

    mx_u = mx_d*(1+radDist_d)+2*p1*mxy_d+p2*(rho2_d+2*mx2_d);
    my_u = my_d*(1+radDist_d)+2*p2*mxy_d+p1*(rho2_d+2*my2_d);

    mx_u = mx_u * fx + cx;
    my_u = my_u * fy + cy;
    /*// Recursive distortion model
    int n = 8;
    Eigen::Vector2d d_u;
    distortion(Eigen::Vector2d(mx_d, my_d), d_u,is_left);
    // Approximate value
    mx_u = mx_d - d_u(0);
    my_u = my_d - d_u(1);

    for (int i = 1; i < n; ++i)
    {
        distortion(Eigen::Vector2d(mx_u, my_u), d_u,is_left);
        mx_u = mx_d - d_u(0);
        my_u = my_d - d_u(1);
    }*/
    // Obtain a projective ray
    P << mx_u, my_u, 1.0;
}


void Frontend::distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u,bool is_left) const{
    double k1 ,k2 ,p1 ,p2 ;
    if(is_left){
        k1 = camera_left_->distortion_coeff_.at<double>(0);
        k2 = camera_left_->distortion_coeff_.at<double>(1);
        p1 = camera_left_->distortion_coeff_.at<double>(2);
        p2 = camera_left_->distortion_coeff_.at<double>(3);
    }else{
        k1 = camera_right_->distortion_coeff_.at<double>(0);
        k2 = camera_right_->distortion_coeff_.at<double>(1);
        p1 = camera_right_->distortion_coeff_.at<double>(2);
        p2 = camera_right_->distortion_coeff_.at<double>(3);
    }
    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
            p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) mp->AddObservation(feat);
    }
}

int Frontend::TriangulateNewPoints() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    SE3 current_pose_Twc = current_frame_->Pose().inverse();
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();
                pworld = current_pose_Twc * pworld;
                new_map_point->SetPos(pworld);
                new_map_point->AddObservation(
                    current_frame_->features_left_[i]);
                new_map_point->AddObservation(
                    current_frame_->features_right_[i]);

                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

int Frontend::EstimateCurrentPose() {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    auto *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());
    optimizer.addVertex(vertex_pose);

    // K_left
    Mat33 K_left = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            auto *edge = new EdgeProjectionPoseOnly(mp->pos_, K_left);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            if (features[i]->is_outlier_) {
                e->computeError();
            }
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
    // Set pose and outlier
    current_frame_->SetPose(vertex_pose->estimate());

    //LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();
            feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }
    return features.size() - cnt_outlier;
}

int Frontend::TrackLastFrame() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_last, kps_current;
    for (auto &kp : last_frame_->features_left_) {
        if (kp->map_point_.lock()) {
            // use project point
            auto mp = kp->map_point_.lock();
            auto px =
                camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(cv::Point2f(px[0], px[1]));
        } else {
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_, current_frame_->left_img_, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));
            feature->map_point_ = last_frame_->features_left_[i]->map_point_;
            current_frame_->features_left_.push_back(feature);
            num_good_pts++;
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::StereoInit() {
    DetectFeatures();
    int num_coor_features = FindFeaturesInRight();
    if (num_coor_features < num_features_init_) {
        return false;
    }
    if (camera_left_->distortion_coeff_.at<double>(0) != 0 ){
        undistortPoints();
    }

    BuildInitMap();
    status_ = FrontendStatus::TRACKING_GOOD;
    if (viewer_) {
        viewer_->AddCurrentFrame(current_frame_);
        viewer_->UpdateMap();
    }
    return true;
}

int Frontend::DetectFeatures() {
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    for (auto &feat : current_frame_->features_left_) {
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    gftt_->detect(current_frame_->left_img_, keypoints, mask);
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        current_frame_->features_left_.push_back(
            std::make_shared<Feature>(current_frame_, kp));
        cnt_detected++;
    }
    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

int Frontend::FindFeaturesInRight() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto &kp : current_frame_->features_left_) {
        kps_left.push_back(kp->position_.pt);
        auto mp = kp->map_point_.lock();
        if (mp) {
            // use projected points as initial guess
            auto px =
                camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else {
            // use same pixel in left iamge
            kps_right.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        current_frame_->left_img_, current_frame_->right_img_, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_right[i], 7);
            Feature::Ptr feat(new Feature(current_frame_, kp));
            feat->is_on_left_image_ = false;
            current_frame_->features_right_.push_back(feat);
            num_good_pts++;
        } else {
            current_frame_->features_right_.push_back(nullptr);
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

bool Frontend::BuildInitMap() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_right_[i] == nullptr) continue;
        // create map point from triangulation
        std::vector<Vec3> points{
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
        Vec3 pworld = Vec3::Zero();
        if (triangulation(poses, points, pworld) && pworld[2] > 0) {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);
            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    backend_->UpdateMap();

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Reset() {
    //todo 重置系统状态
    LOG(INFO) << "Reset is not implemented. ";
    return true;
}

}  // namespace slam