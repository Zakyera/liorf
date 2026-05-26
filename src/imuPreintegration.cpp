#include "utility.h"

#include "liorf/pose_odom_belief_array.h"

#include <cbs/bpsam/incremental_fixed_lag_bpsam_smoother.h>
#include <cbs/key.h>
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
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
#include <rosgraph_msgs/Clock.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <deque>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

namespace {

struct ImuCbsStampedOdomBelief
{
    cbs::AgentId sourceAgent = 0;
    size_t fromPoseIndex = 0;
    size_t toPoseIndex = 0;
    double fromStampSec = -1.0;
    double toStampSec = -1.0;
    std::array<double, 6> relativeMu{};
    std::array<double, 36> covariance{};
    double relaxFactor = 0.0;
    double receivedWallTimeSec = 0.0;
};

cbs::AgentId resolveCbsAgentId(const std::string& id, cbs::AgentId fallback)
{
    if (id.size() == 1u) {
        return static_cast<cbs::AgentId>(id[0]);
    }
    try {
        const int numeric = std::stoi(id);
        if (numeric >= 0 && numeric <= std::numeric_limits<uint8_t>::max()) {
            return static_cast<cbs::AgentId>(numeric);
        }
    } catch (...) {
    }
    return fallback;
}

std::string normalizeCbsOdomSenderMode(std::string mode)
{
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::replace(mode.begin(), mode.end(), '-', '_');
    if (mode == "adjacent" || mode == "adjacent_only") {
        return "adjacent_window";
    }
    if (mode == "latest" || mode == "latest_only") {
        return "latest_edge";
    }
    return mode;
}

bool isValidCbsOdomSenderMode(const std::string& mode)
{
    return mode == "adjacent_window" ||
           mode == "latest_edge" ||
           mode == "new_edge_once";
}

std::string normalizeImuCorrectionFactorMode(std::string mode)
{
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::replace(mode.begin(), mode.end(), '-', '_');
    if (mode == "prior" || mode == "absolute" || mode == "absolute_pose") {
        return "absolute_prior";
    }
    if (mode == "relative" || mode == "between") {
        return "relative_between";
    }
    return mode;
}

bool isValidImuCorrectionFactorMode(const std::string& mode)
{
    return mode == "absolute_prior" || mode == "relative_between";
}

gtsam::Vector6 vector6FromArray(const std::array<double, 6>& values)
{
    gtsam::Vector6 vector;
    for (size_t i = 0; i < values.size(); ++i) {
        vector(static_cast<Eigen::Index>(i)) = values[i];
    }
    return vector;
}

void vector6ToArray(const gtsam::Vector6& vector, std::array<double, 6>* values)
{
    if (!values) {
        return;
    }
    for (size_t i = 0; i < values->size(); ++i) {
        (*values)[i] = vector(static_cast<Eigen::Index>(i));
    }
}

gtsam::Matrix6 matrix6FromArray(const std::array<double, 36>& values)
{
    gtsam::Matrix6 matrix;
    for (size_t r = 0; r < 6u; ++r) {
        for (size_t c = 0; c < 6u; ++c) {
            matrix(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
                values[r * 6u + c];
        }
    }
    return matrix;
}

void matrix6ToArray(const gtsam::Matrix& matrix, std::array<double, 36>* values)
{
    if (!values) {
        return;
    }
    values->fill(std::numeric_limits<double>::quiet_NaN());
    if (matrix.rows() != 6 || matrix.cols() != 6) {
        return;
    }
    for (size_t r = 0; r < 6u; ++r) {
        for (size_t c = 0; c < 6u; ++c) {
            (*values)[r * 6u + c] =
                matrix(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
        }
    }
}

bool finiteArray(const std::array<double, 6>& values)
{
    for (const double value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

bool finiteArray(const std::array<double, 36>& values)
{
    for (const double value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

std::string cbsPoseToken(cbs::AgentId agent, size_t index)
{
    std::ostringstream ss;
    ss << static_cast<char>(agent) << index;
    return ss.str();
}

} // namespace

class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    ros::Subscriber subImuOdometry;
    ros::Subscriber subLaserOdometry;

    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuPath;

    Eigen::Affine3f lidarOdomAffine;
    Eigen::Affine3f imuOdomAffineFront;
    Eigen::Affine3f imuOdomAffineBack;

    tf::TransformListener tfListener;
    tf::StampedTransform lidar2Baselink;

    double lidarOdomTime = -1;
    deque<nav_msgs::Odometry> imuOdomQueue;

    TransformFusion()
    {
        if(lidarFrame != baselinkFrame)
        {
            try
            {
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
            }
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s",ex.what());
            }
        }

        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("liorf/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental",   2000, &TransformFusion::imuOdometryHandler,   this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry   = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
        pubImuPath       = nh.advertise<nav_msgs::Path>    ("liorf/imu/path", 1);
    }

    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }

    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        lidarOdomAffine = odom2affine(*odomMsg);

        lidarOdomTime = odomMsg->header.stamp.toSec();
    }

    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        // static tf
        static tf::TransformBroadcaster tfMap2Odom;
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg);

        // get latest odometry (at current IMU stamp)
        if (lidarOdomTime == -1)
            return;
        while (!imuOdomQueue.empty())
        {
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);
        
        // publish latest odometry
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry);

        // publish tf
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        // publish IMU path
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            while(!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }
};

class IMUPreintegration : public ParamServer
{
public:

    std::mutex mtx;

    ros::Subscriber subImu;
    ros::Subscriber subOdometry;
    ros::Subscriber subPoseOdomBeliefsIn;
    ros::Subscriber subCbsBeliefReceiveClock;
    ros::Publisher pubImuOdometry;
    ros::Publisher pubCbsImuOdometry;
    ros::Publisher pubPoseOdomBeliefsOut;

    bool systemInitialized = false;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::Vector6 correctionNoiseSigmas;
    gtsam::Vector6 correctionNoise2Sigmas;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;


    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;
    gtsam::Pose3 previousImuCorrectionPose;
    bool previousImuCorrectionPoseValid = false;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer;
    std::unique_ptr<cbs::IncrementalFixedLagBpsamSmoother> cbsSmoother;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1;

    bool cbsBeliefBridgeEnable = true;
    bool cbsImuBackendEnable = false;
    bool cbsBeliefRejectFirstMessage = true;
    bool cbsEnableSoftReset = true;
    bool cbsUseRawPreviousBeliefGate = false;
    bool cbsUseTemporaryCbsLinearFactors = true;
    bool cbsTemporaryLinearAlreadyAppliedGateEnable = true;
    double cbsTemporaryLinearAlreadyAppliedMetricThreshold = 0.01;
    double cbsTemporaryLinearAlreadyAppliedDmuThreshold = 1e-3;
    double cbsTemporaryLinearAlreadyAppliedCovRelThreshold = 1e-3;
    double cbsDReset = 0.1;
    double cbsK2LOdomFactorCovarianceScale = 1.0;
    double cbsBeliefReceiveStartDelaySec = 0.0;
    bool cbsBeliefReceiveGateReferenceSet = false;
    double cbsBeliefReceiveGateReferenceStampSec = 0.0;
    double cbsBeliefTimestampToleranceSec = 0.12;
    int cbsBeliefExchangeWindowSize = 30;
    double cbsOdomUnmatchedRetryMaxAgeSec = 5.0;
    size_t cbsOdomUnmatchedRetryMaxBeliefs = 500u;
    std::string cbsBackendMode = "map_optimization";
    std::string cbsOdomSenderMode = "adjacent_window";
    std::string cbsOdomBeliefInTopic = "liorf/cbs/odom_belief_in";
    std::string cbsOdomBeliefOutTopic = "liorf/cbs/odom_belief_out";
    std::string cbsImuOdometryTopic = "liorf/cbs/imu_preintegration/odometry";
    bool cbsImuOdometryPublishEnable = true;
    std::string imuCorrectionFactorMode = "absolute_prior";
    double imuRelativeCorrectionNoiseScale = 1.0;
    cbs::AgentId selfAgentId = static_cast<cbs::AgentId>('l');
    cbs::AgentId kimeraAgentId = static_cast<cbs::AgentId>('k');
    bool cbsLastOutgoingOdomPairValid = false;
    gtsam::Key cbsLastOutgoingOdomFromKey = 0u;
    gtsam::Key cbsLastOutgoingOdomToKey = 0u;

    std::mutex mtxCbsBeliefs;
    std::deque<ImuCbsStampedOdomBelief> incomingCbsOdomBeliefs;
    std::vector<double> cbsPoseTimestampsSec;
    std::atomic<size_t> cbsIncomingReceivedTotal{0u};
    std::atomic<size_t> cbsIncomingMatchedTotal{0u};
    std::atomic<size_t> cbsIncomingDroppedTotal{0u};
    std::atomic<size_t> cbsIncomingAddedTotal{0u};
    std::atomic<size_t> cbsIncomingRejectedTotal{0u};
    std::atomic<size_t> cbsOutgoingPublishedTotal{0u};
    
    // T_bl: tramsform points from lidar frame to imu frame 
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    // T_lb: tramsform points from imu frame to lidar frame
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

    IMUPreintegration()
    {
        configureCbsBackend();

        subImu      = nh.subscribe<sensor_msgs::Imu>  (imuTopic,                   2000, &IMUPreintegration::imuHandler,      this, ros::TransportHints().tcpNoDelay());
        subOdometry = nh.subscribe<nav_msgs::Odometry>("liorf/mapping/odometry_incremental", 5,    &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry = nh.advertise<nav_msgs::Odometry> (odomTopic+"_incremental", 2000);
        if (cbsImuBackendEnable && cbsImuOdometryPublishEnable) {
            pubCbsImuOdometry =
                nh.advertise<nav_msgs::Odometry>(cbsImuOdometryTopic, 50);
        }

        if (cbsImuBackendEnable && cbsBeliefBridgeEnable) {
            subPoseOdomBeliefsIn =
                nh.subscribe<liorf::pose_odom_belief_array>(
                    cbsOdomBeliefInTopic,
                    50,
                    &IMUPreintegration::poseOdomBeliefInHandler,
                    this,
                    ros::TransportHints().tcpNoDelay());
            pubPoseOdomBeliefsOut =
                nh.advertise<liorf::pose_odom_belief_array>(
                    cbsOdomBeliefOutTopic, 50);
            if (cbsBeliefReceiveStartDelaySec > 0.0) {
                subCbsBeliefReceiveClock =
                    nh.subscribe<rosgraph_msgs::Clock>(
                        "/clock",
                        10,
                        &IMUPreintegration::cbsBeliefReceiveClockHandler,
                        this,
                        ros::TransportHints().tcpNoDelay());
            }
        }

        std::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(imuIntegrationSigma, 2); // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << imuInitialRollPitchSigma,
             imuInitialRollPitchSigma, imuInitialYawSigma,
             imuInitialPositionSigma, imuInitialPositionSigma,
             imuInitialPositionSigma)
                .finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, imuInitialVelocitySigma); // m/s
        priorBiasNoise  = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << imuInitialAccBiasSigma,
             imuInitialAccBiasSigma, imuInitialAccBiasSigma,
             imuInitialGyrBiasSigma, imuInitialGyrBiasSigma,
             imuInitialGyrBiasSigma)
                .finished());
        correctionNoiseSigmas =
            (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished(); // rad,rad,rad,m, m, m
        correctionNoise2Sigmas =
            (gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished(); // rad,rad,rad,m, m, m
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas(correctionNoiseSigmas);
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas(correctionNoise2Sigmas);
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }

    void configureCbsBackend()
    {
        nh.param<std::string>("liorf/cbsBackendMode",
                              cbsBackendMode,
                              "map_optimization");
        std::transform(cbsBackendMode.begin(),
                       cbsBackendMode.end(),
                       cbsBackendMode.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (cbsBackendMode != "map_optimization" &&
            cbsBackendMode != "imu_preintegration" &&
            cbsBackendMode != "off") {
            ROS_WARN_STREAM("Unknown LiORF CBS backend mode '"
                            << cbsBackendMode
                            << "'; falling back to map_optimization");
            cbsBackendMode = "map_optimization";
        }
        cbsImuBackendEnable = cbsBackendMode == "imu_preintegration";

        nh.param<bool>("liorf/cbsBeliefBridgeEnable", cbsBeliefBridgeEnable, true);
        if (cbsBackendMode == "off") {
            cbsBeliefBridgeEnable = false;
        }
        nh.param<bool>("liorf/cbsBeliefRejectFirstMessage",
                       cbsBeliefRejectFirstMessage,
                       true);
        nh.param<bool>("liorf/cbsEnableSoftReset", cbsEnableSoftReset, true);
        nh.param<bool>("liorf/cbsUseRawPreviousBeliefGate",
                       cbsUseRawPreviousBeliefGate,
                       false);
        nh.param<bool>("liorf/cbsUseTemporaryCbsLinearFactors",
                       cbsUseTemporaryCbsLinearFactors,
                       true);
        nh.param<bool>("liorf/cbsTemporaryLinearAlreadyAppliedGateEnable",
                       cbsTemporaryLinearAlreadyAppliedGateEnable,
                       true);
        nh.param<double>("liorf/cbsTemporaryLinearAlreadyAppliedMetricThreshold",
                         cbsTemporaryLinearAlreadyAppliedMetricThreshold,
                         0.01);
        nh.param<double>("liorf/cbsTemporaryLinearAlreadyAppliedDmuThreshold",
                         cbsTemporaryLinearAlreadyAppliedDmuThreshold,
                         1e-3);
        nh.param<double>("liorf/cbsTemporaryLinearAlreadyAppliedCovRelThreshold",
                         cbsTemporaryLinearAlreadyAppliedCovRelThreshold,
                         1e-3);
        nh.param<double>("liorf/cbsDReset", cbsDReset, 0.1);
        nh.param<double>("liorf/cbsK2LOdomFactorCovarianceScale",
                         cbsK2LOdomFactorCovarianceScale,
                         1.0);
        nh.param<double>("liorf/cbsBeliefReceiveStartDelaySec",
                         cbsBeliefReceiveStartDelaySec,
                         0.0);
        nh.param<double>("liorf/cbsBeliefTimestampToleranceSec",
                         cbsBeliefTimestampToleranceSec,
                         0.12);
        nh.param<int>("liorf/cbsBeliefExchangeWindowSize",
                      cbsBeliefExchangeWindowSize,
                      30);
        nh.param<std::string>("liorf/cbsOdomSenderMode",
                              cbsOdomSenderMode,
                              "adjacent_window");
        nh.param<double>("liorf/cbsOdomUnmatchedRetryMaxAgeSec",
                         cbsOdomUnmatchedRetryMaxAgeSec,
                         5.0);
        int cbsOdomUnmatchedRetryMaxBeliefsParam = 500;
        nh.param<int>("liorf/cbsOdomUnmatchedRetryMaxBeliefs",
                      cbsOdomUnmatchedRetryMaxBeliefsParam,
                      500);
        nh.param<std::string>("liorf/cbsOdomBeliefInTopic",
                              cbsOdomBeliefInTopic,
                              "liorf/cbs/odom_belief_in");
        nh.param<std::string>("liorf/cbsOdomBeliefOutTopic",
                              cbsOdomBeliefOutTopic,
                              "liorf/cbs/odom_belief_out");
        nh.param<std::string>("liorf/cbsImuOdometryTopic",
                              cbsImuOdometryTopic,
                              "liorf/cbs/imu_preintegration/odometry");
        nh.param<bool>("liorf/cbsImuOdometryPublishEnable",
                       cbsImuOdometryPublishEnable,
                       true);
        nh.param<std::string>("liorf/imuCorrectionFactorMode",
                              imuCorrectionFactorMode,
                              "absolute_prior");
        nh.param<double>("liorf/imuRelativeCorrectionNoiseScale",
                         imuRelativeCorrectionNoiseScale,
                         1.0);

        std::string cbsAgentId = "l";
        nh.param<std::string>("liorf/cbsAgentId", cbsAgentId, "l");
        selfAgentId = resolveCbsAgentId(cbsAgentId, static_cast<cbs::AgentId>('l'));

        if (!std::isfinite(cbsBeliefTimestampToleranceSec) ||
            cbsBeliefTimestampToleranceSec <= 0.0) {
            cbsBeliefTimestampToleranceSec = 0.12;
        }
        if (!std::isfinite(cbsBeliefReceiveStartDelaySec) ||
            cbsBeliefReceiveStartDelaySec < 0.0) {
            cbsBeliefReceiveStartDelaySec = 0.0;
        }
        cbsBeliefExchangeWindowSize = std::max(1, cbsBeliefExchangeWindowSize);
        cbsOdomSenderMode = normalizeCbsOdomSenderMode(cbsOdomSenderMode);
        if (!isValidCbsOdomSenderMode(cbsOdomSenderMode)) {
            ROS_WARN_STREAM("Invalid liorf/cbsOdomSenderMode='"
                            << cbsOdomSenderMode
                            << "', falling back to adjacent_window.");
            cbsOdomSenderMode = "adjacent_window";
        }
        if (!std::isfinite(cbsOdomUnmatchedRetryMaxAgeSec) ||
            cbsOdomUnmatchedRetryMaxAgeSec < 0.0) {
            cbsOdomUnmatchedRetryMaxAgeSec = 5.0;
        }
        cbsOdomUnmatchedRetryMaxBeliefs =
            static_cast<size_t>(std::max(0, cbsOdomUnmatchedRetryMaxBeliefsParam));
        if (!std::isfinite(cbsK2LOdomFactorCovarianceScale) ||
            cbsK2LOdomFactorCovarianceScale <= 0.0) {
            cbsK2LOdomFactorCovarianceScale = 1.0;
        }
        imuCorrectionFactorMode =
            normalizeImuCorrectionFactorMode(imuCorrectionFactorMode);
        if (!isValidImuCorrectionFactorMode(imuCorrectionFactorMode)) {
            ROS_WARN_STREAM("Invalid liorf/imuCorrectionFactorMode='"
                            << imuCorrectionFactorMode
                            << "', falling back to absolute_prior.");
            imuCorrectionFactorMode = "absolute_prior";
        }
        if (!std::isfinite(imuRelativeCorrectionNoiseScale) ||
            imuRelativeCorrectionNoiseScale <= 0.0) {
            imuRelativeCorrectionNoiseScale = 1.0;
        }

        ROS_INFO_STREAM("LiORF IMU CBS backend mode: " << cbsBackendMode
                        << " bridge="
                        << (cbsBeliefBridgeEnable ? "enabled" : "disabled")
                        << " imu_backend="
                        << (cbsImuBackendEnable ? "enabled" : "disabled"));
        ROS_INFO_STREAM("LiORF IMU CBS outgoing odometry mode: "
                        << cbsOdomSenderMode
                        << " window=" << cbsBeliefExchangeWindowSize
                        << " retry_max_age=" << cbsOdomUnmatchedRetryMaxAgeSec
                        << " retry_max_beliefs="
                        << cbsOdomUnmatchedRetryMaxBeliefs);
        ROS_INFO_STREAM("LiORF IMU CBS odometry covariance topic: "
                        << cbsImuOdometryTopic
                        << " publish="
                        << (cbsImuOdometryPublishEnable ? "enabled" : "disabled"));
        ROS_INFO_STREAM("LiORF IMU map correction factor mode: "
                        << imuCorrectionFactorMode
                        << " relative_noise_scale="
                        << imuRelativeCorrectionNoiseScale);
    }

    cbs::BPSAM::Params makeCbsParams() const
    {
        cbs::BPSAM::Params params;
        params.robot_id = selfAgentId;
        params.sam_params_.relinearizeThreshold = 0.1;
        params.sam_params_.relinearizeSkip = 1;
        gtsam::ISAM2GaussNewtonParams gaussNewtonParams;
        params.sam_params_.optimizationParams = gaussNewtonParams;
        params.reject_first_message = cbsBeliefRejectFirstMessage;
        params.gbp_update_params.enable_soft_reset = cbsEnableSoftReset;
        params.gbp_update_params.d_reset = cbsDReset;
        params.use_raw_previous_belief_gate = cbsUseRawPreviousBeliefGate;
        params.use_temporary_cbs_linear_factors = cbsUseTemporaryCbsLinearFactors;
        params.temporary_linear_already_applied_gate_enable =
            cbsTemporaryLinearAlreadyAppliedGateEnable;
        params.temporary_linear_already_applied_metric_threshold =
            cbsTemporaryLinearAlreadyAppliedMetricThreshold;
        params.temporary_linear_already_applied_dmu_threshold =
            cbsTemporaryLinearAlreadyAppliedDmuThreshold;
        params.temporary_linear_already_applied_cov_rel_threshold =
            cbsTemporaryLinearAlreadyAppliedCovRelThreshold;
        params.external_factor_covariance_scale_by_source[kimeraAgentId] =
            cbsK2LOdomFactorCovarianceScale;
        return params;
    }

    gtsam::FixedLagSmoother::KeyTimestampMap makeCbsStateTimestamps(int stateIndex) const
    {
        gtsam::FixedLagSmoother::KeyTimestampMap timestamps;
        const double timestamp = static_cast<double>(stateIndex);
        timestamps[X(stateIndex)] = timestamp;
        timestamps[V(stateIndex)] = timestamp;
        timestamps[B(stateIndex)] = timestamp;
        return timestamps;
    }

    void recordCbsPoseTimestamp(size_t index, double stampSec)
    {
        if (!cbsImuBackendEnable || !std::isfinite(stampSec)) {
            return;
        }
        if (cbsPoseTimestampsSec.size() <= index) {
            cbsPoseTimestampsSec.resize(index + 1u, -1.0);
        }
        cbsPoseTimestampsSec[index] = stampSec;
    }

    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        resetPreviousImuCorrectionPose();

        if (cbsImuBackendEnable) {
            cbsSmoother = std::make_unique<cbs::IncrementalFixedLagBpsamSmoother>(
                static_cast<double>(cbsBeliefExchangeWindowSize),
                makeCbsParams());
            cbsPoseTimestampsSec.clear();
            {
                std::lock_guard<std::mutex> lock(mtxCbsBeliefs);
                incomingCbsOdomBeliefs.clear();
            }
            cbsBeliefReceiveGateReferenceSet = false;
        } else {
            cbsSmoother.reset();
            optimizer = gtsam::ISAM2(optParameters);
        }

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void updateBackend(
        const gtsam::FixedLagSmoother::KeyTimestampMap& timestamps =
            gtsam::FixedLagSmoother::KeyTimestampMap())
    {
        if (cbsImuBackendEnable) {
            cbsSmoother->update(graphFactors, graphValues, timestamps);
        } else {
            optimizer.update(graphFactors, graphValues);
        }
    }

    void updateBackendEmpty()
    {
        if (cbsImuBackendEnable) {
            cbsSmoother->update();
        } else {
            optimizer.update();
        }
    }

    gtsam::Values calculateBackendEstimate() const
    {
        if (cbsImuBackendEnable) {
            return cbsSmoother->calculateEstimate();
        }
        return optimizer.calculateEstimate();
    }

    gtsam::Matrix marginalCovarianceForKey(gtsam::Key key) const
    {
        if (cbsImuBackendEnable) {
            return cbsSmoother->marginalCovariance(key);
        }
        return optimizer.marginalCovariance(key);
    }

    gtsam::noiseModel::Diagonal::shared_ptr mapCorrectionNoise(
        bool degenerate,
        bool scaleForRelativeCorrection) const
    {
        if (!scaleForRelativeCorrection ||
            std::abs(imuRelativeCorrectionNoiseScale - 1.0) <
                std::numeric_limits<double>::epsilon()) {
            return degenerate ? correctionNoise2 : correctionNoise;
        }

        const gtsam::Vector6& baseSigmas =
            degenerate ? correctionNoise2Sigmas : correctionNoiseSigmas;
        return gtsam::noiseModel::Diagonal::Sigmas(
            imuRelativeCorrectionNoiseScale * baseSigmas);
    }

    void addMapCorrectionFactor(int stateIndex,
                                const gtsam::Pose3& currentCorrectionPose,
                                bool degenerate)
    {
        if (imuCorrectionFactorMode == "relative_between" && stateIndex > 0 &&
            previousImuCorrectionPoseValid) {
            const gtsam::Pose3 relativeCorrection =
                previousImuCorrectionPose.between(currentCorrectionPose);
            graphFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(
                X(stateIndex - 1),
                X(stateIndex),
                relativeCorrection,
                mapCorrectionNoise(degenerate, true)));
            return;
        }

        if (imuCorrectionFactorMode == "relative_between") {
            ROS_WARN_STREAM_THROTTLE(
                1.0,
                "LiORF IMU relative correction mode has no previous correction "
                "pose; adding one absolute correction prior to keep the graph "
                "anchored after reset.");
        }
        graphFactors.add(gtsam::PriorFactor<gtsam::Pose3>(
            X(stateIndex),
            currentCorrectionPose,
            mapCorrectionNoise(degenerate, false)));
    }

    void resetPreviousImuCorrectionPose()
    {
        previousImuCorrectionPoseValid = false;
    }

    void setPreviousImuCorrectionPose(const gtsam::Pose3& correctionPose)
    {
        previousImuCorrectionPose = correctionPose;
        previousImuCorrectionPoseValid = true;
    }

    void writePoseCovarianceToOdometry(const gtsam::Matrix& covariance,
                                       nav_msgs::Odometry* odometry) const
    {
        if (!odometry) {
            return;
        }
        std::fill(odometry->pose.covariance.begin(),
                  odometry->pose.covariance.end(),
                  0.0);
        if (covariance.rows() < 6 || covariance.cols() < 6) {
            return;
        }
        static const int remapping[6] = {3, 4, 5, 0, 1, 2};
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                odometry->pose.covariance[remapping[i] * 6 + remapping[j]] =
                    covariance(i, j);
            }
        }
    }

    void publishCbsImuOdometry(const ros::Time& stamp,
                               int poseIndex,
                               const gtsam::Pose3& imuPose,
                               const gtsam::Vector3& velocity)
    {
        if (!cbsImuBackendEnable || !cbsImuOdometryPublishEnable ||
            !pubCbsImuOdometry || pubCbsImuOdometry.getNumSubscribers() == 0) {
            return;
        }

        nav_msgs::Odometry odometry;
        odometry.header.stamp = stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "lidar_link";

        const gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);
        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();

        odometry.twist.twist.linear.x = velocity.x();
        odometry.twist.twist.linear.y = velocity.y();
        odometry.twist.twist.linear.z = velocity.z();

        try {
            const gtsam::Matrix poseCovarianceImu =
                marginalCovarianceForKey(X(poseIndex));
            if (poseCovarianceImu.rows() >= 6 && poseCovarianceImu.cols() >= 6) {
                const gtsam::Matrix6 poseCovarianceImu6 =
                    poseCovarianceImu.block<6, 6>(0, 0);
                const gtsam::Matrix6 imuLocalToLidarLocal =
                    imu2Lidar.inverse().AdjointMap();
                const gtsam::Matrix6 poseCovarianceLidar =
                    imuLocalToLidarLocal * poseCovarianceImu6 *
                    imuLocalToLidarLocal.transpose();
                writePoseCovarianceToOdometry(poseCovarianceLidar, &odometry);
            }
        } catch (const std::exception& e) {
            ROS_WARN_STREAM_THROTTLE(
                1.0,
                "LiORF IMU CBS failed to publish marginal covariance for X"
                    << poseIndex << ": " << e.what());
        }

        pubCbsImuOdometry.publish(odometry);
    }

    double cbsGateStamp(const ImuCbsStampedOdomBelief& belief) const
    {
        if (std::isfinite(belief.toStampSec) && belief.toStampSec > 0.0) {
            return belief.toStampSec;
        }
        return belief.fromStampSec;
    }

    void initializeCbsReceiveGateFromMessage(const liorf::pose_odom_belief_array& msg)
    {
        if (cbsBeliefReceiveStartDelaySec <= 0.0 ||
            cbsBeliefReceiveGateReferenceSet) {
            return;
        }
        const double stampSec = msg.header.stamp.toSec();
        if (!std::isfinite(stampSec) || stampSec <= 0.0) {
            return;
        }
        cbsBeliefReceiveGateReferenceStampSec = stampSec;
        cbsBeliefReceiveGateReferenceSet = true;
        ROS_INFO_STREAM(std::fixed << std::setprecision(9)
                        << "LiORF IMU CBS receive gate reference stamp="
                        << cbsBeliefReceiveGateReferenceStampSec
                        << ", accepting belief odometry after "
                        << (cbsBeliefReceiveGateReferenceStampSec +
                            cbsBeliefReceiveStartDelaySec));
    }

    bool isBeforeCbsReceiveGate(const ImuCbsStampedOdomBelief& belief) const
    {
        if (cbsBeliefReceiveStartDelaySec <= 0.0 ||
            !cbsBeliefReceiveGateReferenceSet) {
            return false;
        }
        const double stampSec = cbsGateStamp(belief);
        if (!std::isfinite(stampSec) || stampSec <= 0.0) {
            return true;
        }
        return stampSec <
               cbsBeliefReceiveGateReferenceStampSec +
                   cbsBeliefReceiveStartDelaySec;
    }

    void cbsBeliefReceiveClockHandler(const rosgraph_msgs::ClockConstPtr& msg)
    {
        if (!msg || cbsBeliefReceiveStartDelaySec <= 0.0 ||
            cbsBeliefReceiveGateReferenceSet) {
            return;
        }
        const double stampSec = msg->clock.toSec();
        if (!std::isfinite(stampSec) || stampSec <= 0.0) {
            return;
        }
        cbsBeliefReceiveGateReferenceStampSec = stampSec;
        cbsBeliefReceiveGateReferenceSet = true;
        subCbsBeliefReceiveClock.shutdown();
        ROS_INFO_STREAM(std::fixed << std::setprecision(9)
                        << "LiORF IMU CBS receive gate reference stamp="
                        << cbsBeliefReceiveGateReferenceStampSec
                        << " from /clock, accepting belief odometry after "
                        << (cbsBeliefReceiveGateReferenceStampSec +
                            cbsBeliefReceiveStartDelaySec));
    }

    void poseOdomBeliefInHandler(const liorf::pose_odom_belief_arrayConstPtr& msg)
    {
        if (!msg || !cbsImuBackendEnable || !cbsBeliefBridgeEnable) {
            return;
        }

        initializeCbsReceiveGateFromMessage(*msg);
        size_t droppedByGate = 0u;
        size_t enqueued = 0u;
        std::deque<ImuCbsStampedOdomBelief> newBeliefs;

        for (const auto& beliefMsg : msg->beliefs) {
            ImuCbsStampedOdomBelief belief;
            belief.sourceAgent = static_cast<cbs::AgentId>(beliefMsg.source_agent);
            if (belief.sourceAgent == selfAgentId) {
                continue;
            }
            belief.fromPoseIndex = static_cast<size_t>(beliefMsg.from_pose_index);
            belief.toPoseIndex = static_cast<size_t>(beliefMsg.to_pose_index);
            belief.fromStampSec = beliefMsg.from_stamp_sec;
            belief.toStampSec = beliefMsg.to_stamp_sec > 0.0
                                    ? beliefMsg.to_stamp_sec
                                    : beliefMsg.header.stamp.toSec();
            belief.relaxFactor = beliefMsg.relax_factor;
            belief.receivedWallTimeSec = ros::WallTime::now().toSec();
            for (size_t i = 0; i < belief.relativeMu.size(); ++i) {
                belief.relativeMu[i] = beliefMsg.relative_mu[i];
            }
            for (size_t i = 0; i < belief.covariance.size(); ++i) {
                belief.covariance[i] = beliefMsg.covariance[i];
            }
            if (!finiteArray(belief.relativeMu) || !finiteArray(belief.covariance) ||
                isBeforeCbsReceiveGate(belief)) {
                ++droppedByGate;
                continue;
            }
            newBeliefs.push_back(belief);
            ++enqueued;
        }

        if (!newBeliefs.empty()) {
            std::lock_guard<std::mutex> lock(mtxCbsBeliefs);
            for (const auto& belief : newBeliefs) {
                incomingCbsOdomBeliefs.push_back(belief);
            }
        }
        cbsIncomingReceivedTotal.fetch_add(enqueued, std::memory_order_relaxed);
        cbsIncomingDroppedTotal.fetch_add(droppedByGate, std::memory_order_relaxed);
    }

    bool findClosestCbsPoseIndexByTimestamp(double stampSec,
                                            size_t* index,
                                            double* absDt) const
    {
        if (!std::isfinite(stampSec) || stampSec <= 0.0 ||
            cbsPoseTimestampsSec.empty()) {
            return false;
        }

        double bestDt = std::numeric_limits<double>::infinity();
        size_t bestIndex = 0u;
        for (size_t i = 0; i < cbsPoseTimestampsSec.size(); ++i) {
            const double candidate = cbsPoseTimestampsSec[i];
            if (!std::isfinite(candidate) || candidate <= 0.0) {
                continue;
            }
            const double dt = std::abs(candidate - stampSec);
            if (dt < bestDt) {
                bestDt = dt;
                bestIndex = i;
            }
        }

        if (!std::isfinite(bestDt) || bestDt > cbsBeliefTimestampToleranceSec) {
            return false;
        }
        if (index) {
            *index = bestIndex;
        }
        if (absDt) {
            *absDt = bestDt;
        }
        return true;
    }

    bool shouldRetryCbsBelief(const ImuCbsStampedOdomBelief& belief) const
    {
        if (cbsOdomUnmatchedRetryMaxBeliefs == 0u ||
            cbsPoseTimestampsSec.empty()) {
            return false;
        }
        double latestStamp = -1.0;
        for (auto it = cbsPoseTimestampsSec.rbegin();
             it != cbsPoseTimestampsSec.rend();
             ++it) {
            if (std::isfinite(*it) && *it > 0.0) {
                latestStamp = *it;
                break;
            }
        }
        if (latestStamp <= 0.0) {
            return true;
        }
        const double stampSec = cbsGateStamp(belief);
        if (!std::isfinite(stampSec) || stampSec <= 0.0) {
            return false;
        }
        return latestStamp - stampSec <= cbsOdomUnmatchedRetryMaxAgeSec;
    }

    size_t consumeIncomingCbsOdomBeliefs()
    {
        if (!cbsImuBackendEnable || !cbsSmoother) {
            return 0u;
        }

        std::deque<ImuCbsStampedOdomBelief> pending;
        {
            std::lock_guard<std::mutex> lock(mtxCbsBeliefs);
            pending.swap(incomingCbsOdomBeliefs);
        }
        if (pending.empty()) {
            return 0u;
        }

        size_t matched = 0u;
        size_t dropped = 0u;
        size_t retried = 0u;
        size_t accepted = 0u;
        size_t rejected = 0u;
        std::vector<ImuCbsStampedOdomBelief> retryBeliefs;

        for (const auto& incoming : pending) {
            size_t fromIndex = 0u;
            size_t toIndex = 0u;
            double fromAbsDt = std::numeric_limits<double>::quiet_NaN();
            double toAbsDt = std::numeric_limits<double>::quiet_NaN();
            const bool fromMatched = findClosestCbsPoseIndexByTimestamp(
                incoming.fromStampSec, &fromIndex, &fromAbsDt);
            const bool toMatched = findClosestCbsPoseIndexByTimestamp(
                incoming.toStampSec, &toIndex, &toAbsDt);
            if (!fromMatched || !toMatched || fromIndex == toIndex) {
                if (shouldRetryCbsBelief(incoming)) {
                    retryBeliefs.push_back(incoming);
                    ++retried;
                } else {
                    ++dropped;
                }
                continue;
            }
            ++matched;

            cbs::BPSAM::CbsOdometryBelief odomBelief;
            odomBelief.source_agent = incoming.sourceAgent;
            odomBelief.from_pose_key = X(static_cast<int>(fromIndex));
            odomBelief.to_pose_key = X(static_cast<int>(toIndex));
            odomBelief.sender_from_pose_key =
                cbs::toPoseKey(incoming.sourceAgent, incoming.fromPoseIndex);
            odomBelief.sender_to_pose_key =
                cbs::toPoseKey(incoming.sourceAgent, incoming.toPoseIndex);
            odomBelief.measured_from_to =
                gtsam::Pose3::Expmap(vector6FromArray(incoming.relativeMu));
            odomBelief.covariance = matrix6FromArray(incoming.covariance);
            odomBelief.relax_factor = incoming.relaxFactor;

            std::vector<cbs::BPSAM::CbsOdometryBelief> singleBelief;
            singleBelief.push_back(std::move(odomBelief));
            const auto addResult =
                cbsSmoother->addOdometryBeliefsDetailed(std::move(singleBelief));
            accepted += addResult.accepted;
            accepted += addResult.accepted_but_skipped_already_applied;
            rejected += addResult.rejected();
            for (const auto& detail : addResult.details) {
                ROS_INFO_STREAM(
                    "LIORF_IMU_CBS_ODOM_ADD_ROW,"
                    << cbsPoseToken(incoming.sourceAgent, incoming.fromPoseIndex)
                    << "->"
                    << cbsPoseToken(incoming.sourceAgent, incoming.toPoseIndex)
                    << ","
                    << cbsPoseToken(selfAgentId, fromIndex)
                    << "->"
                    << cbsPoseToken(selfAgentId, toIndex)
                    << ","
                    << fromAbsDt << ","
                    << toAbsDt << ","
                    << static_cast<int>(detail.status) << ","
                    << detail.covariance_trace << ","
                    << detail.message);
            }
        }

        if (!retryBeliefs.empty()) {
            std::lock_guard<std::mutex> lock(mtxCbsBeliefs);
            for (const auto& belief : retryBeliefs) {
                incomingCbsOdomBeliefs.push_back(belief);
            }
            while (incomingCbsOdomBeliefs.size() >
                   cbsOdomUnmatchedRetryMaxBeliefs) {
                incomingCbsOdomBeliefs.pop_front();
                ++dropped;
            }
        }

        cbsIncomingMatchedTotal.fetch_add(matched, std::memory_order_relaxed);
        cbsIncomingDroppedTotal.fetch_add(dropped, std::memory_order_relaxed);
        cbsIncomingAddedTotal.fetch_add(accepted, std::memory_order_relaxed);
        cbsIncomingRejectedTotal.fetch_add(rejected, std::memory_order_relaxed);
        ROS_INFO_STREAM_THROTTLE(
            1.0,
            "LiORF IMU CBS incoming odometry flow: dequeued="
                << pending.size() << " matched=" << matched
                << " accepted=" << accepted << " rejected=" << rejected
                << " dropped=" << dropped << " retried=" << retried);
        return accepted;
    }

    void refreshOutgoingCbsBeliefs(const ros::Time& stamp)
    {
        if (!cbsImuBackendEnable || !cbsBeliefBridgeEnable || !cbsSmoother ||
            pubPoseOdomBeliefsOut.getNumSubscribers() == 0 ||
            cbsPoseTimestampsSec.size() < 2u) {
            return;
        }

        gtsam::KeySet requestKeys;
        const size_t lastIndex = cbsPoseTimestampsSec.size() - 1u;
        const size_t firstIndex = lastIndex > static_cast<size_t>(cbsBeliefExchangeWindowSize)
                                      ? lastIndex - static_cast<size_t>(cbsBeliefExchangeWindowSize)
                                      : 0u;
        std::vector<gtsam::Key> latestCandidateKeys;
        for (size_t i = firstIndex; i <= lastIndex; ++i) {
            if (std::isfinite(cbsPoseTimestampsSec[i]) &&
                cbsPoseTimestampsSec[i] > 0.0) {
                const gtsam::Key poseKey = X(static_cast<int>(i));
                if (cbsOdomSenderMode == "adjacent_window") {
                    requestKeys.insert(poseKey);
                } else if (cbsSmoother->valueExists(poseKey)) {
                    latestCandidateKeys.push_back(poseKey);
                }
            }
        }
        bool latestPairValid = false;
        gtsam::Key latestFromKey = 0u;
        gtsam::Key latestToKey = 0u;
        if (cbsOdomSenderMode == "latest_edge" ||
            cbsOdomSenderMode == "new_edge_once") {
            if (latestCandidateKeys.size() < 2u) {
                return;
            }
            latestFromKey = latestCandidateKeys[latestCandidateKeys.size() - 2u];
            latestToKey = latestCandidateKeys.back();
            latestPairValid = true;
            if (cbsOdomSenderMode == "new_edge_once" &&
                cbsLastOutgoingOdomPairValid &&
                latestFromKey == cbsLastOutgoingOdomFromKey &&
                latestToKey == cbsLastOutgoingOdomToKey) {
                return;
            }
            requestKeys.insert(latestFromKey);
            requestKeys.insert(latestToKey);
        }
        if (requestKeys.size() < 2u) {
            return;
        }

        std::vector<cbs::BPSAM::CbsOdometryBelief> outgoing;
        try {
            cbsSmoother->setMarginalizationGraph(cbs::BPSAM::MarginalizationType::LOCAL);
            outgoing = cbsSmoother->getOdometryBeliefs(requestKeys, kimeraAgentId);
        } catch (const std::exception& e) {
            ROS_WARN_STREAM_THROTTLE(
                1.0,
                "LiORF IMU CBS failed to create outgoing odometry beliefs: "
                    << e.what());
            return;
        }

        liorf::pose_odom_belief_array msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = odometryFrame;
        msg.beliefs.reserve(outgoing.size());
        for (const auto& odom : outgoing) {
            const size_t fromIndex = gtsam::Symbol(odom.from_pose_key).index();
            const size_t toIndex = gtsam::Symbol(odom.to_pose_key).index();
            if (fromIndex >= cbsPoseTimestampsSec.size() ||
                toIndex >= cbsPoseTimestampsSec.size()) {
                continue;
            }
            if (cbsPoseTimestampsSec[fromIndex] <= 0.0 ||
                cbsPoseTimestampsSec[toIndex] <= 0.0) {
                continue;
            }

            liorf::pose_odom_belief beliefMsg;
            beliefMsg.header = msg.header;
            beliefMsg.source_agent = static_cast<uint8_t>(selfAgentId);
            beliefMsg.from_pose_index = static_cast<uint32_t>(fromIndex);
            beliefMsg.to_pose_index = static_cast<uint32_t>(toIndex);
            beliefMsg.from_stamp_sec = cbsPoseTimestampsSec[fromIndex];
            beliefMsg.to_stamp_sec = cbsPoseTimestampsSec[toIndex];
            beliefMsg.relax_factor = odom.relax_factor;
            std::array<double, 6> relativeMu;
            std::array<double, 36> covariance;
            vector6ToArray(gtsam::Pose3::Logmap(odom.measured_from_to),
                           &relativeMu);
            matrix6ToArray(odom.covariance, &covariance);
            for (size_t i = 0; i < relativeMu.size(); ++i) {
                beliefMsg.relative_mu[i] = relativeMu[i];
            }
            for (size_t i = 0; i < covariance.size(); ++i) {
                beliefMsg.covariance[i] = covariance[i];
            }
            msg.beliefs.push_back(beliefMsg);
        }
        if (latestPairValid && !msg.beliefs.empty()) {
            cbsLastOutgoingOdomPairValid = true;
            cbsLastOutgoingOdomFromKey = latestFromKey;
            cbsLastOutgoingOdomToKey = latestToKey;
        }

        pubPoseOdomBeliefsOut.publish(msg);
        cbsOutgoingPublishedTotal.fetch_add(msg.beliefs.size(),
                                            std::memory_order_relaxed);
        ROS_INFO_STREAM_THROTTLE(
            1.0,
            "LiORF IMU CBS outgoing odometry flow: published="
                << msg.beliefs.size()
                << " total="
                << cbsOutgoingPublishedTotal.load(std::memory_order_relaxed));
    }

    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
        resetPreviousImuCorrectionPose();
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        double currentCorrectionTime = ROS_TIME(odomMsg);

        // make sure we have imu data to integrate
        if (imuQueOpt.empty())
            return;

        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));


        // 0. initialize system
        if (systemInitialized == false)
        {
            resetOptimization();

            // pop old IMU message
            while (!imuQueOpt.empty())
            {
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            prevPose_ = lidarPose.compose(lidar2Imu);
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            graphFactors.add(priorPose);
            // initial velocity
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // initial bias
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            prevState_ = gtsam::NavState(prevPose_, prevVel_);
            recordCbsPoseTimestamp(0u, currentCorrectionTime);
            updateBackend(cbsImuBackendEnable
                              ? makeCbsStateTimestamps(0)
                              : gtsam::FixedLagSmoother::KeyTimestampMap());
            setPreviousImuCorrectionPose(prevPose_);
            publishCbsImuOdometry(odomMsg->header.stamp, 0, prevPose_, prevVel_);
            graphFactors.resize(0);
            graphValues.clear();

            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            key = 1;
            systemInitialized = true;
            return;
        }


        // reset graph for speed
        if (!cbsImuBackendEnable && key == 100)
        {
            // get updated noise before reset
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(marginalCovarianceForKey(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(marginalCovarianceForKey(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(marginalCovarianceForKey(B(key-1)));
            // reset graph
            resetOptimization();
            // add pose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            updateBackend();
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }


        // 1. integrate imu data and optimize
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            if (imuTime < currentCorrectionTime - delta_t)
            {
                double dt = (lastImuT_opt < 0) ? (1.0 / imuRate) : (imuTime - lastImuT_opt);
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }
        // add imu factor to graph
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // add imu bias between factor
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        addMapCorrectionFactor(key, curPose, degenerate);
        // insert predicted values
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize
        recordCbsPoseTimestamp(static_cast<size_t>(key), currentCorrectionTime);
        updateBackend(cbsImuBackendEnable
                          ? makeCbsStateTimestamps(key)
                          : gtsam::FixedLagSmoother::KeyTimestampMap());
        updateBackendEmpty();
        graphFactors.resize(0);
        graphValues.clear();
        if (consumeIncomingCbsOdomBeliefs() > 0u) {
            updateBackendEmpty();
        }
        // Overwrite the beginning of the preintegration for the next step.
        gtsam::Values result = calculateBackendEstimate();
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
        setPreviousImuCorrectionPose(curPose);
        publishCbsImuOdometry(odomMsg->header.stamp, key, prevPose_, prevVel_);
        // Reset the optimization preintegration object.
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        refreshOutgoingCbsBeliefs(odomMsg->header.stamp);
        // check optimization
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        double lastImuQT = -1;
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // repropogate
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / imuRate) :(imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
    }

    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);

        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        if (doneFirstOpt == false)
            return;

        double imuTime = ROS_TIME(&thisImu);
        double dt = (lastImuT_imu < 0) ? (1.0 / imuRate) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry.publish(odometry);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "roboat_loam");
    
    IMUPreintegration ImuP;

    TransformFusion TF;

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");
    
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    
    return 0;
}
