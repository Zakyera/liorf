#include "utility.h"
#include "liorf/cloud_info.h"
#include "liorf/save_map.h"
// <!-- liorf_yjz_lucky_boy -->
#include <sensor_msgs/NavSatFix.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
// zy Step 1_c
// Intuition: keep CBS dependency optional at compile time.
#ifdef LIORF_USE_CBS
#include <cbs/bpsam/bpsam.h>
#endif
// zy Step 5_a
// Adds fixed-width timestamp/source types for external belief bookkeeping.
#include <cstdint>
#include <string>
// zy Step 6_a
// Adds numeric helpers for nearest-timestamp matching and bounded comparisons.
#include <cstdlib>
#include <limits>
// zy Step 12_a
// Adds source-tag normalization helpers for robust external-prior routing.
#include <algorithm>
#include <cctype>
// zy Step 13_a
// Enables eigenvalue-based covariance conditioning for robust external prior ingestion.
#include <Eigen/Eigenvalues>



#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>

#include "Scancontext.h"

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose
// zy Step 2_a
// Runtime optimizer selector used by the mapping backend.
inline bool UseCbsOptimizer(const bool use_cbs_optimizer_flag
#ifdef LIORF_USE_CBS
                            , const std::shared_ptr<cbs::BPSAM>& cbs_optimizer
#endif
                            ) {
#ifdef LIORF_USE_CBS
    return use_cbs_optimizer_flag && static_cast<bool>(cbs_optimizer);
#else
    (void)use_cbs_optimizer_flag;
    return false;
#endif
}

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;

enum class SCInputType 
{ 
    SINGLE_SCAN_FULL, 
    SINGLE_SCAN_FEAT, 
    MULTI_SCAN_FEAT 
}; 

class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;
    // zy Step 2_b
    // Store the active estimate independently from legacy isamCurrentEstimate for unified access.
    Values optimizerCurrentEstimate;

    // zy Step 1_d
    // runtime switch lets us compare legacy ISAM2 and CBS on the same node.
    bool use_cbs_optimizer_ = false;

#ifdef LIORF_USE_CBS
    // zy Step 1_e
    // initialize CBS object now; full update-path substitution comes in next steps.
    std::shared_ptr<cbs::BPSAM> cbs_optimizer_;
#endif

    // zy Step 5_b
    // Holds the latest LIORF pose belief that can be shared with an external fusion bridge.
    struct ExternalPoseBelief
    {
        int64_t timestamp_kf_nsec_ = -1;
        size_t pose_index_ = 0;
        gtsam::Pose3 W_Pose_L_ = gtsam::Pose3();
        Eigen::Matrix<double, 6, 6> covariance_ =
            Eigen::Matrix<double, 6, 6>::Identity();
        std::string source_ = "liorf";
    };

    // zy Step 5_c
    // Buffers incoming external pose priors until they are matched/injected in optimize flow.
    struct ExternalPosePrior
    {
        int64_t timestamp_kf_nsec_ = -1;
        gtsam::Pose3 W_Pose_L_ = gtsam::Pose3();
        Eigen::Matrix<double, 6, 6> covariance_ =
            Eigen::Matrix<double, 6, 6>::Identity();
        std::string source_ = "unknown";
        uint64_t source_seq_ = 0;
    };

    // zy Step 5_d
    // Keeps exchange-related state bounded and thread-safe for async bridge I/O.
    mutable std::mutex external_pose_priors_queue_mutex_;
    std::deque<ExternalPosePrior> external_pose_priors_queue_;
    size_t max_external_pose_priors_queue_size_ = 5000;

    mutable std::mutex timestamp_to_pose_idx_map_mutex_;
    std::map<int64_t, size_t> timestamp_to_pose_idx_map_;
    size_t max_timestamp_to_pose_idx_map_size_ = 20000;
    int64_t external_prior_timestamp_tolerance_ns_ = 2000000;  // 2 ms
    // zy Step 10_a
    // Bounds external-prior freshness and per-cycle ingestion to keep optimization stable under bursty inputs.
    double max_external_prior_age_sec_ = 2.0;
    double max_external_prior_future_lead_sec_ = 0.05;
    size_t max_external_priors_per_optimize_ = 200;
    // zy Step 13_b
    // Sets minimum and maximum confidence bounds for external pose-prior covariance.
    double external_prior_min_variance_ = 1e-6;
    double external_prior_max_variance_ = 1e2;


    mutable std::mutex latest_external_pose_belief_mutex_;
    ExternalPoseBelief latest_external_pose_belief_;
    bool has_latest_external_pose_belief_ = false;


    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Publisher pubSLAMInfo;
    ros::Publisher pubGpsOdom;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;

    // zy Step 7_a
    // ROS bridge endpoints for exchanging external pose beliefs and priors.
    ros::Publisher pubExternalPoseBelief_;
    ros::Subscriber subExternalPosePrior_;

    // zy Step 7_b
    // Topic and source settings are configurable so bridge wiring does not require recompiling LIORF.
    std::string external_pose_belief_topic_ = "liorf/cbs/external_pose_belief";
    std::string external_pose_prior_topic_ = "liorf/cbs/external_pose_prior";
    std::string external_prior_default_source_ = "kimera";
    // zy Step 9_a
    // Controls whether external exchange uses IMU/body frame (true) or lidar frame (false).
    bool external_exchange_in_body_frame_ = false;
    // zy Step 12_b
    // Tracks last seen sequence per source to drop replayed/out-of-order external priors.
    mutable std::mutex external_source_seq_mutex_;
    std::map<std::string, uint64_t> last_external_source_seq_by_source_;



    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;
    liorf::cloud_info cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterLocalMapSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    // vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    vector<gtsam::SharedNoiseModel> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    GeographicLib::LocalCartesian gps_trans_;

    // scancontext loop closure
    SCManager scManager;

    mapOptimization()
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);
        // zy Step 1_f
        // runtime ROS param controls whether this node should attempt CBS mode.
        nh.param<bool>("liorf/use_cbs_optimizer", use_cbs_optimizer_, false);

        // zy Step 7_c
        // Loads bridge-facing topics and default source tag for incoming external priors.
        nh.param<std::string>("liorf/external_pose_belief_topic",
                              external_pose_belief_topic_,
                              "liorf/cbs/external_pose_belief");
        nh.param<std::string>("liorf/external_pose_prior_topic",
                              external_pose_prior_topic_,
                              "liorf/cbs/external_pose_prior");
        nh.param<std::string>("liorf/external_prior_default_source",
                              external_prior_default_source_,
                              "kimera");
        // zy Step 10_b
        // Makes prior-aging/budget limits configurable from launch without code edits.
        nh.param<double>("liorf/max_external_prior_age_sec",
                         max_external_prior_age_sec_,
                         2.0);
        nh.param<double>("liorf/max_external_prior_future_lead_sec",
                         max_external_prior_future_lead_sec_,
                         0.05);

        int max_external_priors_per_optimize_tmp =
            static_cast<int>(max_external_priors_per_optimize_);
        nh.param<int>("liorf/max_external_priors_per_optimize",
                      max_external_priors_per_optimize_tmp,
                      200);
        max_external_priors_per_optimize_ =
            static_cast<size_t>(std::max(1, max_external_priors_per_optimize_tmp));
        
        // zy Step 9_b
        // Keeps old behavior by default, while allowing body-frame exchange when wiring with Kimera.
        nh.param<bool>("liorf/external_exchange_in_body_frame",
                       external_exchange_in_body_frame_,
                       false);



#ifdef LIORF_USE_CBS
        if (use_cbs_optimizer_) {
            cbs::BPSAM::Params cbs_params;
            cbs_params.sam_params_ = parameters;
            cbs_params.enable_gkcm = false;
            cbs_params.robot_id = static_cast<cbs::AgentId>('b');  // LIORF agent id.
            cbs_optimizer_ = std::make_shared<cbs::BPSAM>(cbs_params);
            ROS_INFO_STREAM("LIORF CBS BPSAM initialized. use_cbs_optimizer=true");
        }
#else
        if (use_cbs_optimizer_) {
            ROS_WARN_STREAM("liorf/use_cbs_optimizer=true but LIORF was built without LIORF_USE_CBS. Falling back to legacy ISAM2.");
            use_cbs_optimizer_ = false;
        }
#endif


        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/trajectory", 1);
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/map_global", 1);
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("liorf/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("liorf/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("liorf/mapping/path", 1);

        subCloud = nh.subscribe<liorf::cloud_info>("liorf/deskew/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subGPS   = nh.subscribe<sensor_msgs::NavSatFix> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        subLoop  = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());

        srvSaveMap  = nh.advertiseService("liorf/save_map", &mapOptimization::saveMapService, this);

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/liorf/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/map_local", 1);
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/cloud_registered_raw", 1);

        pubSLAMInfo           = nh.advertise<liorf::cloud_info>("liorf/mapping/slam_info", 1);
        pubGpsOdom            = nh.advertise<nav_msgs::Odometry> ("liorf/mapping/gps_odom", 1);

        // zy Step 7_d
        // Publishes LIORF beliefs and receives external priors over ROS for cross-estimator fusion.
        pubExternalPoseBelief_ =
            nh.advertise<nav_msgs::Odometry>(external_pose_belief_topic_, 10);
        subExternalPosePrior_ = nh.subscribe<nav_msgs::Odometry>(
            external_pose_prior_topic_,
            200,
            &mapOptimization::externalPosePriorHandler,
            this,
            ros::TransportHints().tcpNoDelay());


        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterLocalMapSurf.setLeafSize(surroundingKeyframeMapLeafSize, surroundingKeyframeMapLeafSize, surroundingKeyframeMapLeafSize);
        downSizeFilterICP.setLeafSize(loopClosureICPSurfLeafSize, loopClosureICPSurfLeafSize, loopClosureICPSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
    }
    // zy Step 2_c
    // Returns whether CBS is currently the active optimization heart.
    bool usingCbs() const
    {
#ifdef LIORF_USE_CBS
        return UseCbsOptimizer(use_cbs_optimizer_, cbs_optimizer_);
#else
        return false;
#endif
    }

    // zy Step 2_d
    // Returns the latest estimate from whichever optimizer is active.
    Values computeActiveEstimate() const
    {
        if (usingCbs()) {
#ifdef LIORF_USE_CBS
            return cbs_optimizer_->calculateEstimate();
#endif
        }
        return isam->calculateEstimate();
    }

    // zy Step 2_e
    // Unified marginal covariance query for the last pose key.
    gtsam::Matrix activePoseMarginalCovariance(const gtsam::Key& key) const
    {
        if (usingCbs()) {
#ifdef LIORF_USE_CBS
            return cbs_optimizer_->marginalCovariance(key);
#endif
        }
        return isam->marginalCovariance(key);
    }

    // zy Step 2_f
    // Unified value-existence check used before querying estimate/covariance.
    bool activeValueExists(const gtsam::Key& key) const
    {
        if (usingCbs()) {
#ifdef LIORF_USE_CBS
            return cbs_optimizer_->valueExists(key);
#endif
        }
        return isamCurrentEstimate.exists(key);
    }
    
    // zy Step 4_a
    // Aligns LIORF pose keys with Kimera/CBS (`x(index)`) so exchanged beliefs hit the same variable IDs.
    gtsam::Key poseKeyFromIndex(const size_t idx) const
    {
        return X(static_cast<uint64_t>(idx));
    }

    // zy Step 5_e
    // Converts ROS timestamp to integer nanoseconds so cross-system matching uses one time domain.
    int64_t toTimestampNsec(const ros::Time& stamp) const
    {
        return static_cast<int64_t>(stamp.sec) * 1000000000LL +
               static_cast<int64_t>(stamp.nsec);
    }

    // zy Step 12_c
    // Normalizes incoming source tags so routing logic is case-insensitive and consistent.
    std::string normalizeExternalSourceTag(const std::string& raw_source) const
    {
        std::string source =
            raw_source.empty() ? external_prior_default_source_ : raw_source;
        std::transform(source.begin(), source.end(), source.begin(),
                    [](unsigned char c) {
                        return static_cast<char>(std::tolower(c));
                    });
        return source;
    }

    // zy Step 12_d
    // Identifies tags that refer to LIORF itself to prevent local self-feedback loops.
    bool isSelfExternalSourceTag(const std::string& source) const
    {
        return source == "liorf" || source == "liosam" || source == "self";
    }

    // zy Step 12_e
    // Accepts only strictly newer sequence numbers per source to suppress replay duplicates.
    bool shouldAcceptExternalSourceSeq(const std::string& source,
                                    const uint64_t source_seq)
    {
        if (source_seq == 0) {
            return true;
        }

        std::lock_guard<std::mutex> lock(external_source_seq_mutex_);
        auto it = last_external_source_seq_by_source_.find(source);
        if (it != last_external_source_seq_by_source_.end() &&
            source_seq <= it->second) {
            return false;
        }

        last_external_source_seq_by_source_[source] = source_seq;
        return true;
    }


    // zy Step 5_f
    // Tracks local keyframe timestamp -> pose index for future external prior matching.
    void rememberPoseIndexForTimestamp(const ros::Time& stamp, const size_t pose_idx)
    {
        const int64_t ts_nsec = toTimestampNsec(stamp);
        std::lock_guard<std::mutex> lock(timestamp_to_pose_idx_map_mutex_);
        timestamp_to_pose_idx_map_[ts_nsec] = pose_idx;

        while (timestamp_to_pose_idx_map_.size() > max_timestamp_to_pose_idx_map_size_) {
            timestamp_to_pose_idx_map_.erase(timestamp_to_pose_idx_map_.begin());
        }
    }

    // zy Step 5_g
    // Caches the latest LIORF belief snapshot for external bridge publication.
    void updateLatestExternalPoseBelief(
        const ros::Time& stamp,
        const size_t pose_idx,
        const gtsam::Pose3& W_Pose_L,
        const Eigen::Matrix<double, 6, 6>& covariance)
    {
        ExternalPoseBelief belief;
        belief.timestamp_kf_nsec_ = toTimestampNsec(stamp);
        belief.pose_index_ = pose_idx;
        belief.W_Pose_L_ = W_Pose_L;
        belief.covariance_ = covariance;
        belief.source_ = "liorf";

        std::lock_guard<std::mutex> lock(latest_external_pose_belief_mutex_);
        latest_external_pose_belief_ = belief;
        has_latest_external_pose_belief_ = true;
    }

    // zy Step 5_h
    // Exposes the latest LIORF belief to bridge code without touching optimizer internals.
    bool getLatestExternalPoseBelief(ExternalPoseBelief* belief) const
    {
        if (!belief) {
            return false;
        }
        std::lock_guard<std::mutex> lock(latest_external_pose_belief_mutex_);
        if (!has_latest_external_pose_belief_) {
            return false;
        }
        *belief = latest_external_pose_belief_;
        return true;
    }

    // zy Step 9_c
    // Uses LIORF's existing translation-only lidar<->body extrinsic convention for exchange conversion.
    gtsam::Pose3 lidarToBodyExtrinsic() const
    {
        return gtsam::Pose3(
            gtsam::Rot3(1, 0, 0, 0),
            gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));
    }

    // zy Step 9_d
    // Converts incoming exchange-frame pose into LIORF internal lidar frame.
    gtsam::Pose3 exchangePoseToLidarPose(const gtsam::Pose3& W_Pose_exchange) const
    {
        if (!external_exchange_in_body_frame_) {
            return W_Pose_exchange;
        }
        const gtsam::Pose3 B_Pose_L = lidarToBodyExtrinsic().inverse();
        return W_Pose_exchange.compose(B_Pose_L);
    }

    // zy Step 9_e
    // Converts internal lidar-frame pose into configured exchange frame before publication.
    gtsam::Pose3 lidarPoseToExchangePose(const gtsam::Pose3& W_Pose_L) const
    {
        if (!external_exchange_in_body_frame_) {
            return W_Pose_L;
        }
        const gtsam::Pose3 L_Pose_B = lidarToBodyExtrinsic();
        return W_Pose_L.compose(L_Pose_B);
    }

    // zy Step 11_a
    // Converts exchange-frame pose covariance into LIORF lidar-frame covariance using adjoint mapping.
    Eigen::Matrix<double, 6, 6> exchangeCovarianceToLidarCovariance(
        const Eigen::Matrix<double, 6, 6>& covariance_exchange) const
    {
        if (!external_exchange_in_body_frame_) {
            return covariance_exchange;
        }
        const gtsam::Pose3 B_Pose_L = lidarToBodyExtrinsic().inverse();
        const gtsam::Matrix66 adj = B_Pose_L.AdjointMap();
        return adj * covariance_exchange * adj.transpose();
    }

    // zy Step 11_b
    // Converts LIORF lidar-frame pose covariance into configured exchange-frame covariance.
    Eigen::Matrix<double, 6, 6> lidarCovarianceToExchangeCovariance(
        const Eigen::Matrix<double, 6, 6>& covariance_lidar) const
    {
        if (!external_exchange_in_body_frame_) {
            return covariance_lidar;
        }
        const gtsam::Pose3 L_Pose_B = lidarToBodyExtrinsic();
        const gtsam::Matrix66 adj = L_Pose_B.AdjointMap();
        return adj * covariance_lidar * adj.transpose();
    }

    // zy Step 6_b
    // Finds the closest local keyframe index to an external timestamp within configured tolerance.
    bool findNearestPoseIndexForTimestamp(
        const int64_t timestamp_kf_nsec,
        size_t* pose_idx,
        int64_t* matched_timestamp_kf_nsec = nullptr,
        const int64_t tolerance_nsec = -1) const
    {
        if (!pose_idx) {
            return false;
        }

        const int64_t effective_tolerance =
            (tolerance_nsec >= 0) ? tolerance_nsec : external_prior_timestamp_tolerance_ns_;

        std::lock_guard<std::mutex> lock(timestamp_to_pose_idx_map_mutex_);
        if (timestamp_to_pose_idx_map_.empty()) {
            return false;
        }

        auto lower = timestamp_to_pose_idx_map_.lower_bound(timestamp_kf_nsec);
        auto best = timestamp_to_pose_idx_map_.end();
        int64_t best_abs_dt = std::numeric_limits<int64_t>::max();

        const auto consider =
            [&](const std::map<int64_t, size_t>::const_iterator& it) {
                if (it == timestamp_to_pose_idx_map_.end()) {
                    return;
                }
                const int64_t abs_dt = std::llabs(it->first - timestamp_kf_nsec);
                if (abs_dt < best_abs_dt) {
                    best_abs_dt = abs_dt;
                    best = it;
                }
            };

        consider(lower);
        if (lower != timestamp_to_pose_idx_map_.begin()) {
            consider(std::prev(lower));
        }

        if (best == timestamp_to_pose_idx_map_.end() ||
            best_abs_dt > effective_tolerance) {
            return false;
        }

        *pose_idx = best->second;
        if (matched_timestamp_kf_nsec) {
            *matched_timestamp_kf_nsec = best->first;
        }
        return true;
    }

    // zy Step 6_c
    // Queues an external pose prior so it can be injected during the next graph update.
    bool enqueueExternalPosePriorFromCovariance(
        const int64_t timestamp_kf_nsec,
        const gtsam::Pose3& W_Pose_L,
        const Eigen::Matrix<double, 6, 6>& covariance,
        const std::string& source = "unknown",
        const uint64_t source_seq = 0)
    {
        if (!covariance.allFinite()) {
            ROS_WARN_STREAM("Dropped external prior: covariance has non-finite values.");
            return false;
        }

        const Eigen::Matrix<double, 6, 6> sym_cov =
            0.5 * (covariance + covariance.transpose());
        if ((sym_cov.diagonal().array() <= 0.0).any()) {
            ROS_WARN_STREAM("Dropped external prior: covariance diagonal must be positive.");
            return false;
        }

        ExternalPosePrior prior;
        prior.timestamp_kf_nsec_ = timestamp_kf_nsec;
        prior.W_Pose_L_ = W_Pose_L;
        prior.covariance_ = sym_cov;
        prior.source_ = source;
        prior.source_seq_ = source_seq;

        std::lock_guard<std::mutex> lock(external_pose_priors_queue_mutex_);
        external_pose_priors_queue_.push_back(prior);
        while (external_pose_priors_queue_.size() > max_external_pose_priors_queue_size_) {
            external_pose_priors_queue_.pop_front();
        }
        return true;
    }

    // zy Step 6_d
    // Returns a LIORF belief at a requested timestamp by querying active optimizer state at the matched keyframe.
    bool getExternalPoseBeliefAtTimestamp(
        const int64_t timestamp_kf_nsec,
        ExternalPoseBelief* belief,
        const int64_t tolerance_nsec = 2000000) const
    {
        if (!belief) {
            return false;
        }

        size_t pose_idx = 0;
        int64_t matched_ts_nsec = -1;
        if (!findNearestPoseIndexForTimestamp(
                timestamp_kf_nsec, &pose_idx, &matched_ts_nsec, tolerance_nsec)) {
            return false;
        }

        const gtsam::Key pose_key = poseKeyFromIndex(pose_idx);
        if (!activeValueExists(pose_key)) {
            return false;
        }

        const gtsam::Values estimate = computeActiveEstimate();
        if (!estimate.exists(pose_key)) {
            return false;
        }

        const gtsam::Matrix cov = activePoseMarginalCovariance(pose_key);
        if (cov.rows() < 6 || cov.cols() < 6) {
            return false;
        }

        ExternalPoseBelief out;
        out.timestamp_kf_nsec_ = matched_ts_nsec;
        out.pose_index_ = pose_idx;
        out.W_Pose_L_ = estimate.at<gtsam::Pose3>(pose_key);
        out.covariance_ = cov.block<6, 6>(0, 0);
        out.source_ = "liorf";

        *belief = out;
        return true;
    }

    // zy Step 7_e
    // Converts inbound ROS prior messages into LIORF external-prior queue entries.
    void externalPosePriorHandler(const nav_msgs::OdometryConstPtr& msg)
    {
        if (!msg) {
            return;
        }

        const auto& q = msg->pose.pose.orientation;
        const double q_norm =
            std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        if (q_norm < 1e-9) {
            ROS_WARN_STREAM("Dropped external prior: invalid zero-norm quaternion.");
            return;
        }

        Eigen::Matrix<double, 6, 6> covariance =
            Eigen::Matrix<double, 6, 6>::Zero();
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                covariance(r, c) = msg->pose.covariance[r * 6 + c];
            }
        }

        const gtsam::Pose3 W_Pose_exchange(
            gtsam::Rot3::Quaternion(q.w, q.x, q.y, q.z),
            gtsam::Point3(msg->pose.pose.position.x,
                          msg->pose.pose.position.y,
                          msg->pose.pose.position.z));

        // zy Step 9_f
        // Converts external message pose to LIORF lidar frame before queueing.
        const gtsam::Pose3 W_Pose_L =
            exchangePoseToLidarPose(W_Pose_exchange);
        // zy Step 11_c
        // Keeps covariance in the same frame as the converted pose before queueing.
        const Eigen::Matrix<double, 6, 6> covariance_lidar =
            exchangeCovarianceToLidarCovariance(covariance);


        const int64_t ts_nsec = toTimestampNsec(msg->header.stamp);
        // zy Step 12_f
        // Normalizes source tags and drops self/duplicate priors before they enter the queue.
        const std::string source = normalizeExternalSourceTag(msg->child_frame_id);
        const uint64_t source_seq = static_cast<uint64_t>(msg->header.seq);

        if (isSelfExternalSourceTag(source)) {
            ROS_WARN_STREAM_THROTTLE(
                2.0,
                "Dropped external prior from self source tag: " << source);
            return;
        }

        if (!shouldAcceptExternalSourceSeq(source, source_seq)) {
            ROS_WARN_STREAM_THROTTLE(
                2.0,
                "Dropped replay/out-of-order external prior. source=" << source
                << ", seq=" << source_seq);
            return;
        }

        const bool queued = enqueueExternalPosePriorFromCovariance(
            ts_nsec, W_Pose_L, covariance_lidar, source, source_seq);


        // if (!queued) {
        //     ROS_WARN_STREAM("Failed to queue external prior. ts_nsec="
        //                     << ts_nsec << ", source=" << source
        //                     << ", seq=" << source_seq); (zy cancelled it)

        // zy Step 22_b
        // Avoids duplicate warning spam because ingress helper already logs exact rejection reasons.
        if (!queued) {
            ROS_DEBUG_STREAM_THROTTLE(
                2.0,
                "External prior was not queued. ts_nsec=" << ts_nsec
                << ", source=" << normalizeExternalSourceTag(msg->child_frame_id)
                << ", seq=" << source_seq);
}

        }
    }

    // zy Step 7_f
    // Publishes latest LIORF belief as ROS odometry so bridge nodes can forward it to Kimera.
    void publishLatestExternalPoseBelief()
    {
        if (pubExternalPoseBelief_.getNumSubscribers() == 0) {
            return;
        }

        ExternalPoseBelief belief;
        if (!getLatestExternalPoseBelief(&belief)) {
            return;
        }
        if (belief.timestamp_kf_nsec_ < 0) {
            return;
        }

        nav_msgs::Odometry msg;
        msg.header.stamp.fromNSec(
            static_cast<uint64_t>(belief.timestamp_kf_nsec_));
        msg.header.frame_id = odometryFrame;
        msg.child_frame_id = belief.source_;

        // zy Step 9_g
        // Publishes in configured exchange frame while keeping LIORF internals in lidar frame.
        const gtsam::Pose3 W_Pose_exchange =
            lidarPoseToExchangePose(belief.W_Pose_L_);

        msg.pose.pose.position.x = W_Pose_exchange.translation().x();
        msg.pose.pose.position.y = W_Pose_exchange.translation().y();
        msg.pose.pose.position.z = W_Pose_exchange.translation().z();

        const Eigen::Quaterniond q_belief(W_Pose_exchange.rotation().matrix());
        msg.pose.pose.orientation.x = q_belief.x();
        msg.pose.pose.orientation.y = q_belief.y();
        msg.pose.pose.orientation.z = q_belief.z();
        msg.pose.pose.orientation.w = q_belief.w();

        // zy Step 11_d
        // Publishes covariance in the same exchange frame as the outgoing pose.
        const Eigen::Matrix<double, 6, 6> covariance_exchange =
            lidarCovarianceToExchangeCovariance(belief.covariance_);

        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                msg.pose.covariance[r * 6 + c] = covariance_exchange(r, c);
            }
        }

        pubExternalPoseBelief_.publish(msg);
    }

#ifdef LIORF_USE_CBS
    // zy Step 8_a
    // Maps source tags to stable CBS agent IDs so beliefs keep correct sender ownership.
    bool mapSourceToCbsAgent(const std::string& source,
                             cbs::AgentId* agent_id) const
    {
        if (!agent_id) {
            return false;
        }

        if (source == "kimera" || source == "kimera_vio" || source == "vio") {
            *agent_id = static_cast<cbs::AgentId>('a');
            return true;
        }

        if (source == "liorf" || source == "liosam" || source == "self") {
            *agent_id = static_cast<cbs::AgentId>('b');
            return true;
        }

        return false;
    }
#endif

    // zy Step 10_c
    // Applies time-window and per-cycle budget guards so external fusion remains real-time and stable.
    void injectQueuedExternalPosePriors()
    {
        std::deque<ExternalPosePrior> incoming_priors;
        {
            std::lock_guard<std::mutex> lock(external_pose_priors_queue_mutex_);
            if (external_pose_priors_queue_.empty()) {
                return;
            }
            incoming_priors.swap(external_pose_priors_queue_);
        }

        const int64_t current_ts_nsec = toTimestampNsec(timeLaserInfoStamp);
        const int64_t max_prior_age_ns = static_cast<int64_t>(
            std::max(0.0, max_external_prior_age_sec_) * 1e9);
        const int64_t max_future_lead_ns = static_cast<int64_t>(
            std::max(0.0, max_external_prior_future_lead_sec_) * 1e9);

        int64_t newest_local_ts_nsec = -1;
        {
            std::lock_guard<std::mutex> lock(timestamp_to_pose_idx_map_mutex_);
            if (!timestamp_to_pose_idx_map_.empty()) {
                newest_local_ts_nsec = timestamp_to_pose_idx_map_.rbegin()->first;
            }
        }

        size_t injected = 0;
        size_t deferred = 0;
        size_t dropped_old = 0;
        size_t dropped_bad_noise = 0;
        size_t dropped_self_source = 0;
        size_t dropped_unknown_source = 0;
        size_t deferred_budget = 0;

#ifdef LIORF_USE_CBS
        std::map<gtsam::Key, std::vector<std::pair<cbs::AgentId, gbp::Gaussian>>>
            cbs_incoming_beliefs;
        size_t num_external_beliefs_staged = 0;
        size_t num_external_beliefs_rejected = 0;
        constexpr cbs::AgentId kLiorfAgentId = static_cast<cbs::AgentId>('b');
#endif

        std::deque<ExternalPosePrior> deferred_priors;

        for (const auto& prior : incoming_priors) {
            if (max_prior_age_ns > 0 &&
                prior.timestamp_kf_nsec_ + max_prior_age_ns < current_ts_nsec) {
                ++dropped_old;
                continue;
            }

            if (max_future_lead_ns > 0 &&
                prior.timestamp_kf_nsec_ > current_ts_nsec + max_future_lead_ns) {
                deferred_priors.push_back(prior);
                ++deferred;
                continue;
            }

            if (!prior.covariance_.allFinite() ||
                (prior.covariance_.diagonal().array() <= 0.0).any()) {
                ++dropped_bad_noise;
                continue;
            }

            if (injected >= max_external_priors_per_optimize_) {
                deferred_priors.push_back(prior);
                ++deferred_budget;
                continue;
            }

            size_t matched_pose_idx = 0;
            int64_t matched_ts_nsec = -1;
            if (!findNearestPoseIndexForTimestamp(
                    prior.timestamp_kf_nsec_, &matched_pose_idx, &matched_ts_nsec)) {
                const bool could_match_future =
                    (newest_local_ts_nsec >= 0) &&
                    (prior.timestamp_kf_nsec_ >
                     newest_local_ts_nsec + external_prior_timestamp_tolerance_ns_);
                if (could_match_future) {
                    deferred_priors.push_back(prior);
                    ++deferred;
                } else {
                    ++dropped_old;
                }
                continue;
            }

            const gtsam::Key pose_key = poseKeyFromIndex(matched_pose_idx);

#ifdef LIORF_USE_CBS
            if (usingCbs()) {
                if (!cbs_optimizer_) {
                    ++dropped_bad_noise;
                    continue;
                }

                cbs::AgentId sender_id = static_cast<cbs::AgentId>('a');
                if (!mapSourceToCbsAgent(prior.source_, &sender_id)) {
                    ++dropped_unknown_source;
                    continue;
                }

                if (sender_id == kLiorfAgentId) {
                    ++dropped_self_source;
                    continue;
                }

                const gtsam::Vector6 mu =
                    gtsam::traits<gtsam::Pose3>::Logmap(prior.W_Pose_L_);
                const gtsam::Matrix66 cov = prior.covariance_;
                gbp::Gaussian belief(pose_key, mu, cov, 1);

                cbs_incoming_beliefs[pose_key].emplace_back(sender_id, belief);
                ++num_external_beliefs_staged;
                ++injected;
                continue;
            }
#endif

                if (isSelfExternalSourceTag(prior.source_)) {
                ++dropped_self_source;
                continue;
            }

            const gtsam::SharedNoiseModel prior_noise =
                gtsam::noiseModel::Gaussian::Covariance(prior.covariance_);
            gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(
                pose_key, prior.W_Pose_L_, prior_noise));
            ++injected;
        }

        size_t queue_size_now = 0;
        {
            std::lock_guard<std::mutex> lock(external_pose_priors_queue_mutex_);
            for (const auto& prior : deferred_priors) {
                external_pose_priors_queue_.push_back(prior);
                while (external_pose_priors_queue_.size() >
                       max_external_pose_priors_queue_size_) {
                    external_pose_priors_queue_.pop_front();
                }
            }
            queue_size_now = external_pose_priors_queue_.size();
        }

#ifdef LIORF_USE_CBS
        if (usingCbs() && !cbs_incoming_beliefs.empty()) {
            num_external_beliefs_rejected =
                cbs_optimizer_->addBeliefs(cbs_incoming_beliefs);

            ROS_INFO_STREAM("CBS addBeliefs: staged=" << num_external_beliefs_staged
                            << ", rejected=" << num_external_beliefs_rejected
                            << ", unknown_source=" << dropped_unknown_source
                            << ", self_source=" << dropped_self_source);
        }
#endif

        if (injected > 0) {
            aLoopIsClosed = true;
        }

        ROS_INFO_STREAM_COND(
            (injected + deferred + dropped_old + dropped_bad_noise +
             dropped_self_source + dropped_unknown_source + deferred_budget) > 0,
            "External prior stats: injected=" << injected
            << ", deferred=" << deferred
            << ", dropped_old=" << dropped_old
            << ", dropped_bad_noise=" << dropped_bad_noise
            << ", dropped_self_source=" << dropped_self_source
            << ", dropped_unknown_source=" << dropped_unknown_source
            << ", deferred_budget=" << deferred_budget
            << ", queue_size_now=" << queue_size_now
            << ", per_optimize_budget=" << max_external_priors_per_optimize_);
    }


    // zy Step 3_a
    // Routes factor-graph updates through whichever optimizer is active and preserves loop-closure extra iterations.
    void updateActiveOptimizer(const gtsam::NonlinearFactorGraph& factors,
                               const gtsam::Values& values,
                               const bool run_extra_updates)
    {
        if (usingCbs()) {
#ifdef LIORF_USE_CBS
            cbs::BPSAM::UpdateParams update_params;
            cbs_optimizer_->update(factors, values, update_params);

            if (run_extra_updates) {
                for (int i = 0; i < 5; ++i) {
                    cbs_optimizer_->update(gtsam::NonlinearFactorGraph(),
                                           gtsam::Values(),
                                           update_params);
                }
            }
            return;
#endif
        }

        isam->update(factors, values);
        isam->update();

        if (run_extra_updates) {
            for (int i = 0; i < 5; ++i) {
                isam->update();
            }
        }
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }

    void laserCloudInfoHandler(const liorf::cloud_infoConstPtr& msgIn)
    {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        // extract info and feature cloud
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudSurfLast);

        // TODO
        // ......
        // remapping
        // ......
        // END

        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;

            updateInitialGuess();

            extractSurroundingKeyFrames();

            downsampleCurrentScan();

            scan2MapOptimization();

            saveKeyFramesAndFactor();

            correctPoses();

            publishOdometry();

            publishFrames();
        }
    }

    void gpsHandler(const sensor_msgs::NavSatFixConstPtr& gpsMsg)
    {
        if (gpsMsg->status.status != 0)
            return;

        Eigen::Vector3d trans_local_;
        static bool first_gps = false;
        if (!first_gps) {
            first_gps = true;
            gps_trans_.Reset(gpsMsg->latitude, gpsMsg->longitude, gpsMsg->altitude);
        }

        gps_trans_.Forward(gpsMsg->latitude, gpsMsg->longitude, gpsMsg->altitude, trans_local_[0], trans_local_[1], trans_local_[2]);

        nav_msgs::Odometry gps_odom;
        gps_odom.header = gpsMsg->header;
        gps_odom.header.frame_id = "map";
        gps_odom.pose.pose.position.x = trans_local_[0];
        gps_odom.pose.pose.position.y = trans_local_[1];
        gps_odom.pose.pose.position.z = trans_local_[2];
        gps_odom.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, 0.0);
        pubGpsOdom.publish(gps_odom);
        gpsQueue.push_back(gps_odom);
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    













    bool saveMapService(liorf::save_mapRequest& req, liorf::save_mapResponse& res)
    {
      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      if(req.destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
      else saveMapDirectory = std::getenv("HOME") + req.destination;
      cout << "Save destination: " << saveMapDirectory << endl;
      // create directory and remove old files;
      int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
      unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
      // save key frame transformations
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // extract global point cloud map

      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {

        // save surf cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      }

      // save global point cloud map
      *globalMapCloud += *globalSurfCloud;

      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      res.success = ret == 0;

      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }

    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        liorf::save_mapRequest  req;
        liorf::save_mapResponse res;

        if(!saveMapService(req, res)){
            cout << "Fail to save map" << endl;
        }
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (common_lib_->pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }












    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            rate.sleep();
            performRSLoopClosure();
            performSCLoopClosure();
            visualizeLoopClosure();
        }
    }

    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    void performRSLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0, -1);
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum, -1);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    // copy from sc-lio-sam
    void performSCLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        // first: nn index, second: yaw diff 
        auto detectResult = scManager.detectLoopClosureID(); 
        int loopKeyCur    = copy_cloudKeyPoses3D->size() - 1;;
        int loopKeyPre    = detectResult.first;
        float yawDiffRad  = detectResult.second; // not use for v1 (because pcl icp withi initial somthing wrong...)
        if( loopKeyPre == -1)
            return;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return;

        // std::cout << "SC loop found! between " << loopKeyCur << " and " << loopKeyPre << "." << std::endl; // giseop

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            int base_key = 0;
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0, base_key); // giseop 
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum, base_key); // giseop 

            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();

        // // transform from world origin to wrong pose
        // Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // // transform from world origin to corrected pose
        // Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        // pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        // gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);

        // gtsam::Vector Vector6(6);
        // float noiseScore = icp.getFitnessScore();
        // Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        // noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // giseop 
        pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

        // giseop, robust kernel for a SC loop
        float robustNoiseScore = 0.5; // constant is ok...
        gtsam::Vector robustNoiseVector6(6); 
        robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
        noiseModel::Base::shared_ptr robustConstraintNoise; 
        robustConstraintNoise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure, but with a good front-end loop detector, Cauchy is empirically enough.
            gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6)
        ); // - checked it works. but with robust kernel, map modification may be delayed (i.e,. requires more true-positive loop factors)

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(robustConstraintNoise);
        mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum, const int& loop_index)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;

            int select_loop_index = (loop_index != -1) ? loop_index : key + i;
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[select_loop_index]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;
        
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }

    void updateInitialGuess()
    {
        // save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;
        // initialization
        if (cloudKeyPoses3D->points.empty())
        {
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;

            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }

        // use imu pre-integration estimation for pose guess
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odomAvailable == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            if (lastImuPreTransAvailable == false)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                lastImuPreTransformation = transBack;

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                return;
            }
        }

        // use imu incremental estimation for pose guess (only rotation)
        if (cloudInfo.imuAvailable == true && imuType)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        laserCloudSurfFromMap->clear(); 
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            if (common_lib_->pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp;
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding surf key frames (or map)
        downSizeFilterLocalMapSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterLocalMapSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();    
        // } else {
        //     extractNearby();
        // }

        extractNearby();
    }

    void downsampleCurrentScan()
    {
        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    void combineOptimizationCoeffs()
    {
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeMapped[2]);
        float crx = cos(transformTobeMapped[2]);
        float sry = sin(transformTobeMapped[1]);
        float cry = cos(transformTobeMapped[1]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].x;
            pointOri.y = laserCloudOri->points[i].y;
            pointOri.z = laserCloudOri->points[i].z;
            // lidar -> camera
            coeff.x = coeffSel->points[i].x;
            coeff.y = coeffSel->points[i].y;
            coeff.z = coeffSel->points[i].z;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
/*             float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
             */

            float arx = (-srx * cry * pointOri.x - (srx * sry * srz + crx * crz) * pointOri.y + (crx * srz - srx * sry * crz) * pointOri.z) * coeff.x
                      + (crx * cry * pointOri.x - (srx * crz - crx * sry * srz) * pointOri.y + (crx * sry * crz + srx * srz) * pointOri.z) * coeff.y;

            float ary = (-crx * sry * pointOri.x + crx * cry * srz * pointOri.y + crx * cry * crz * pointOri.z) * coeff.x
                      + (-srx * sry * pointOri.x + srx * sry * srz * pointOri.y + srx * cry * crz * pointOri.z) * coeff.y
                      + (-cry * pointOri.x - sry * srz * pointOri.y - sry * crz * pointOri.z) * coeff.z;

            float arz = ((crx * sry * crz + srx * srz) * pointOri.y + (srx * crz - crx * sry * srz) * pointOri.z) * coeff.x
                      + ((-crx * srz + srx * sry * crz) * pointOri.y + (-srx * sry * srz - crx * crz) * pointOri.z) * coeff.y
                      + (cry * crz * pointOri.y - cry * srz * pointOri.z) * coeff.z;
              
            // camera -> lidar
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arx;
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }

    void scan2MapOptimization()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (laserCloudSurfLastDSNum > 30)
        {
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true)
                    break;              
            }

            transformUpdate();
        } else {
            ROS_WARN("Not enough features! Only %d planar features available.", laserCloudSurfLastDSNum);
        }
    }

    void transformUpdate()
    {
        if (cloudInfo.imuAvailable == true && imuType)
        {
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    // void addOdomFactor()
    // {
    //     if (cloudKeyPoses3D->points.empty())
    //     {
    //         noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
    //         gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
    //         initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    //     }else{
    //         noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    //         gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
    //         gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
    //         gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
    //         initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    //     }
    // } (zy cancelled it)

    // zy Step 4_b
    // Builds odometry constraints with shared symbolic pose keys instead of raw integer keys.
    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise =
                noiseModel::Diagonal::Variances(
                    (Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished());

            const gtsam::Key first_key = poseKeyFromIndex(0);
            gtSAMgraph.add(PriorFactor<Pose3>(
                first_key, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(first_key, trans2gtsamPose(transformTobeMapped));
        }
        else
        {
            noiseModel::Diagonal::shared_ptr odometryNoise =
                noiseModel::Diagonal::Variances(
                    (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);

            const gtsam::Key prev_key =
                poseKeyFromIndex(cloudKeyPoses3D->size() - 1);
            const gtsam::Key curr_key =
                poseKeyFromIndex(cloudKeyPoses3D->size());

            gtSAMgraph.add(BetweenFactor<Pose3>(
                prev_key, curr_key, poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(curr_key, poseTo);
        }
    }


    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (common_lib_->pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (common_lib_->pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                // gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                // gtSAMgraph.add(gps_factor); (zy cancelled it)

                // zy Step 4_c
                // Anchors GPS to the same shared pose-key namespace used by odometry and CBS beliefs.
                const gtsam::Key curr_key = poseKeyFromIndex(cloudKeyPoses3D->size());
                gtsam::GPSFactor gps_factor(
                    curr_key, gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);


                aLoopIsClosed = true;
                break;
            }
        }
    }

    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            // gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            auto noiseBetween = loopNoiseQueue[i];
            // gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween)); (zy cancelled it)
            // zy Step 4_d
            // Converts loop-closure indices into shared symbolic pose keys.
            const gtsam::Key key_from =
                poseKeyFromIndex(static_cast<size_t>(indexFrom));
            const gtsam::Key key_to =
                poseKeyFromIndex(static_cast<size_t>(indexTo));
            gtSAMgraph.add(BetweenFactor<Pose3>(
                key_from, key_to, poseBetween, noiseBetween));

        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void saveKeyFramesAndFactor()
    {
        if (saveFrame() == false)
            return;

        // odom factor
        addOdomFactor();

        // gps factor
        addGPSFactor();

        // loop factor
        addLoopFactor();
        // zy Step 6_f
        // Injects matched external beliefs as pose priors before running the active optimizer update.
        injectQueuedExternalPosePriors();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // zy Step 3_b
        // Keeps one update path for both ISAM2 and CBS so mapOptimization behavior stays consistent.
        updateActiveOptimizer(gtSAMgraph, initialEstimate, aLoopIsClosed);

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        // isamCurrentEstimate = isam->calculateEstimate();
        // latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1); (zy cancelled it)
        
        // zy Step 4_e
        // Reads the newest optimized pose via shared symbolic key (`x(index)`), not raw key count.
        optimizerCurrentEstimate = computeActiveEstimate();
        isamCurrentEstimate = optimizerCurrentEstimate; // keep legacy variable updated for unchanged code paths.
        const size_t latest_pose_idx = cloudKeyPoses3D->size();
        const gtsam::Key latest_pose_key = poseKeyFromIndex(latest_pose_idx);
        latestEstimate = optimizerCurrentEstimate.at<Pose3>(latest_pose_key);


        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);
        // zy Step 5_i
        // Saves timestamp->index alignment right when a new keyframe pose is committed.
        const size_t new_pose_idx = cloudKeyPoses3D->size() - 1;
        rememberPoseIndexForTimestamp(timeLaserInfoStamp, new_pose_idx);

        // zy Step 7_g
        // Emits one fresh LIORF belief per committed keyframe for external fusion consumers.
        publishLatestExternalPoseBelief();

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        // poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1); (zy cancelled it)
        // zy Step 4_f
        // Queries covariance of the latest saved pose using the shared symbolic key.
        const gtsam::Key last_saved_pose_key =
            poseKeyFromIndex(cloudKeyPoses3D->size() - 1);
        poseCovariance = activePoseMarginalCovariance(last_saved_pose_key);
        // zy Step 5_j
        // Updates the outgoing LIORF belief snapshot (pose + covariance) for external fusion.
        updateLatestExternalPoseBelief(
            timeLaserInfoStamp,
            cloudKeyPoses3D->size() - 1,
            latestEstimate,
            poseCovariance.block<6, 6>(0, 0));




        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // The following code is copy from sc-lio-sam
        // Scan Context loop detector - giseop
        // - SINGLE_SCAN_FULL: using downsampled original point cloud (/full_cloud_projected + downsampling)
        // - SINGLE_SCAN_FEAT: using surface feature as an input point cloud for scan context (2020.04.01: checked it works.)
        // - MULTI_SCAN_FEAT: using NearKeyframes (because a MulRan scan does not have beyond region, so to solve this issue ... )
        const SCInputType sc_input_type = SCInputType::SINGLE_SCAN_FULL; // change this 

        if( sc_input_type == SCInputType::SINGLE_SCAN_FULL )
        {
            pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *thisRawCloudKeyFrame);

            scManager.makeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame);
        }  
        else if (sc_input_type == SCInputType::SINGLE_SCAN_FEAT)
        { 
            scManager.makeAndSaveScancontextAndKeys(*thisSurfKeyFrame); 
        }
        else if (sc_input_type == SCInputType::MULTI_SCAN_FEAT)
        { 
            pcl::PointCloud<PointType>::Ptr multiKeyFrameFeatureCloud(new pcl::PointCloud<PointType>());
            loopFindNearKeyframes(multiKeyFrameFeatureCloud, cloudKeyPoses6D->size() - 1, historyKeyframeSearchNum, -1);
            scManager.makeAndSaveScancontextAndKeys(*multiKeyFrameFeatureCloud); 
        }

        // save path for visualization
        updatePath(thisPose6D);
    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // clear path
            globalPath.poses.clear();
            // update key poses
            // int numPoses = isamCurrentEstimate.size(); (zy cancelled it)
            // zy Step 2_i
            // Use the active estimate container as the single source for pose correction after loop/CBS updates.
            int numPoses = optimizerCurrentEstimate.size();

            // for (int i = 0; i < numPoses; ++i)
            // {   
            //     // zy Step 2_j i replaced below all the isamCurrentEstimate with optimizerCurrentEstimate to make sure the corrected poses are from the currently active optimizer.
            //     cloudKeyPoses3D->points[i].x = optimizerCurrentEstimate.at<Pose3>(i).translation().x();
            //     cloudKeyPoses3D->points[i].y = optimizerCurrentEstimate.at<Pose3>(i).translation().y();
            //     cloudKeyPoses3D->points[i].z = optimizerCurrentEstimate.at<Pose3>(i).translation().z();

            //     cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            //     cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            //     cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            //     cloudKeyPoses6D->points[i].roll  = optimizerCurrentEstimate.at<Pose3>(i).rotation().roll();
            //     cloudKeyPoses6D->points[i].pitch = optimizerCurrentEstimate.at<Pose3>(i).rotation().pitch();
            //     cloudKeyPoses6D->points[i].yaw   = optimizerCurrentEstimate.at<Pose3>(i).rotation().yaw();

            //     updatePath(cloudKeyPoses6D->points[i]);
            // } (zy cancelled it)

            for (int i = 0; i < numPoses; ++i)
            {
                // zy Step 4_g
                // Applies loop-corrected poses by symbolic key so map backfill matches optimizer key IDs.
                const gtsam::Key pose_key =
                    poseKeyFromIndex(static_cast<size_t>(i));
                const gtsam::Pose3 pose_i =
                    optimizerCurrentEstimate.at<Pose3>(pose_key);

                cloudKeyPoses3D->points[i].x = pose_i.translation().x();
                cloudKeyPoses3D->points[i].y = pose_i.translation().y();
                cloudKeyPoses3D->points[i].z = pose_i.translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = pose_i.rotation().roll();
                cloudKeyPoses6D->points[i].pitch = pose_i.rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = pose_i.rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }


            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);
        
        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imuAvailable == true && imuType)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
        // publish SLAM infomation for 3rd-party usage
        static int lastSLAMInfoPubSize = -1;
        if (pubSLAMInfo.getNumSubscribers() != 0)
        {
            if (lastSLAMInfoPubSize != cloudKeyPoses6D->size())
            {
                liorf::cloud_info slamInfo;
                slamInfo.header.stamp = timeLaserInfoStamp;
                pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
                *cloudOut += *laserCloudSurfLastDS;
                slamInfo.key_frame_cloud = publishCloud(ros::Publisher(), cloudOut, timeLaserInfoStamp, lidarFrame);
                slamInfo.key_frame_poses = publishCloud(ros::Publisher(), cloudKeyPoses6D, timeLaserInfoStamp, odometryFrame);
                pcl::PointCloud<PointType>::Ptr localMapOut(new pcl::PointCloud<PointType>());
                *localMapOut += *laserCloudSurfFromMapDS;
                slamInfo.key_frame_map = publishCloud(ros::Publisher(), localMapOut, timeLaserInfoStamp, odometryFrame);
                pubSLAMInfo.publish(slamInfo);
                lastSLAMInfoPubSize = cloudKeyPoses6D->size();
            }
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "liorf");

    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
    
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
