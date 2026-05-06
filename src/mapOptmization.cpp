#include "utility.h"
#include "liorf/cloud_info.h"
#include "liorf/pose_odom_belief.h"
#include "liorf/pose_odom_belief_array.h"
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
#include <aria_viz/visualizer_rerun.h>
#include <cbs/bpsam/bpsam.h>
#include <cbs/key.h>
#include <cbs/utils/gtsam_compat.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <atomic>
#include <unordered_map>
#include <unordered_set>

#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>

#include "Scancontext.h"

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

namespace {
std::string makeRerunRecordingId(const std::string& prefix)
{
    std::time_t now = std::time(nullptr);
    std::tm local_time{};
    localtime_r(&now, &local_time);

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &local_time);
    return prefix + "_" + buffer;
}

std::string getDockerGatewayIp()
{
    std::ifstream route_file("/proc/net/route");
    std::string line;
    std::getline(route_file, line);
    while (std::getline(route_file, line)) {
        std::istringstream iss(line);
        std::string iface;
        std::string destination;
        std::string gateway;
        unsigned int flags = 0u;
        if (!(iss >> iface >> destination >> gateway >> std::hex >> flags)) {
            continue;
        }
        if (destination != "00000000" || gateway.size() != 8u) {
            continue;
        }

        const unsigned long raw_gateway = std::stoul(gateway, nullptr, 16);
        std::ostringstream ip;
        ip << (raw_gateway & 0xfful) << "."
           << ((raw_gateway >> 8u) & 0xfful) << "."
           << ((raw_gateway >> 16u) & 0xfful) << "."
           << ((raw_gateway >> 24u) & 0xfful);
        return ip.str();
    }
    return "";
}

std::string defaultRerunHost()
{
    const char* env_host = std::getenv("CBSMS_RERUN_HOST");
    if (env_host != nullptr && std::string(env_host).empty() == false) {
        return env_host;
    }

    const std::string gateway_ip = getDockerGatewayIp();
    if (!gateway_ip.empty()) {
        return "rerun+http://" + gateway_ip + ":9876/proxy";
    }

    return "rerun+http://host.docker.internal:9876/proxy";
}

double elapsedMs(const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

void atomicAddRelaxed(std::atomic<double>* target, double value)
{
    if (!target || !std::isfinite(value) || value < 0.0) {
        return;
    }

    double current = target->load(std::memory_order_relaxed);
    while (!target->compare_exchange_weak(current,
                                          current + value,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed)) {
    }
}

template <typename PointCloudT>
std::vector<gtsam::Point3> toRerunPoints(const PointCloudT& cloud, size_t max_points)
{
    std::vector<gtsam::Point3> points;
    if (cloud.points.empty() || max_points == 0u) {
        return points;
    }

    const size_t stride = std::max<size_t>(
        1u, (cloud.points.size() + max_points - 1u) / max_points);
    points.reserve(std::min(cloud.points.size(), max_points));

    for (size_t i = 0u; i < cloud.points.size(); i += stride) {
        const auto& point = cloud.points[i];
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
            points.emplace_back(point.x, point.y, point.z);
        }
    }
    return points;
}
} // namespace

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
    std::unique_ptr<cbs::BPSAM> bpsam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance = Eigen::MatrixXd::Identity(6, 6);
    cbs::AgentId selfAgentId = static_cast<cbs::AgentId>('a');
    std::vector<Key> localPoseKeys;
    std::unordered_map<Key, size_t> localPoseKeyToIndex;
    std::vector<double> localPoseTimestampsSec;

    struct StampedOdomBelief
    {
        cbs::AgentId sourceAgent;
        size_t fromPoseIndex;
        size_t toPoseIndex;
        double fromStampSec;
        double toStampSec;
        uint64_t senderTimestampNs;
        std::string senderFrameId;
        std::array<double, 6> relativeMu;
        std::array<double, 36> covariance;
        double relaxFactor;
    };

    struct RerunExternalOdomEdge
    {
        cbs::AgentId sourceAgent;
        Key fromPoseKey;
        Key toPoseKey;
    };

    std::mutex mtxBeliefExchange;
    std::mutex mtxRerunCbsVisualization;
    std::deque<StampedOdomBelief> incomingStampedOdomBeliefs;
    std::vector<StampedOdomBelief> outgoingStampedOdomBeliefs;
    std::vector<RerunExternalOdomEdge> rerunExternalOdomEdges;
    size_t beliefExchangeWindowSize = 30;
    size_t cbsBeliefMaxRootSize = 60;
    double beliefTimestampToleranceSec = 0.05;
    bool cbsBeliefRejectFirstMessage = true;
    bool cbsEnableSoftReset = true;
    bool cbsUseRawPreviousBeliefGate = false;
    bool cbsUseTemporaryCbsLinearFactors = true;
    bool cbsTemporaryLinearAlreadyAppliedGateEnable = true;
    double cbsTemporaryLinearAlreadyAppliedMetricThreshold = 0.01;
    double cbsTemporaryLinearAlreadyAppliedDmuThreshold = 1e-3;
    double cbsTemporaryLinearAlreadyAppliedCovRelThreshold = 1e-3;
    bool cbsBeliefBridgeEnable = true;
    std::string cbsOdomBeliefInTopic = "liorf/cbs/odom_belief_in";
    std::string cbsOdomBeliefOutTopic = "liorf/cbs/odom_belief_out";
    std::atomic<size_t> cbsBeliefsIncomingReceivedTotal{0u};
    std::atomic<size_t> cbsBeliefsIncomingDequeuedTotal{0u};
    std::atomic<size_t> cbsBeliefsIncomingMatchedByIndexTotal{0u};
    std::atomic<size_t> cbsBeliefsIncomingMatchedByTimestampTotal{0u};
    std::atomic<size_t> cbsBeliefsIncomingDroppedNoLocalTimestampTotal{0u};
    std::atomic<size_t> cbsBeliefsIncomingDroppedInvalidStampTotal{0u};
    std::atomic<size_t> cbsBeliefsIncomingDroppedTimestampMismatchTotal{0u};
    std::atomic<size_t> cbsBeliefsIncomingAddedToBpsamTotal{0u};
    std::atomic<size_t> cbsBeliefsIncomingRejectedByBpsamTotal{0u};
    std::atomic<size_t> cbsBeliefsIncomingRejectedFirstMessageTotal{0u};
    std::atomic<size_t> cbsBeliefsIncomingRejectedUpdateStatusTotal{0u};
    std::atomic<size_t> cbsBeliefsIncomingRejectedShapeTotal{0u};
    std::atomic<size_t> cbsBeliefsIncomingRejectedExceptionTotal{0u};
    std::atomic<size_t> cbsBeliefsOutgoingPreparedTotal{0u};
    std::atomic<size_t> cbsBeliefsOutgoingPublishedTotal{0u};
    std::atomic<size_t> cbsBeliefsIncomingReceivedPerRerunFrame{0u};
    std::atomic<size_t> cbsBeliefsIncomingAddedToBpsamPerRerunFrame{0u};
    std::atomic<size_t> cbsBeliefsIncomingRejectedFirstMessagePerRerunFrame{0u};
    std::atomic<size_t> cbsBeliefsIncomingRejectedUpdateStatusPerRerunFrame{0u};
    std::atomic<size_t> cbsBeliefsIncomingRejectedShapePerRerunFrame{0u};
    std::atomic<size_t> cbsBeliefsIncomingRejectedExceptionPerRerunFrame{0u};
    std::atomic<size_t> cbsBeliefsOutgoingPublishedPerRerunFrame{0u};
    std::atomic<double> bpsamOptimizationTimeMsPerRerunFrame{0.0};
    std::atomic<double> cbsBeliefGenerationTimeMsPerRerunFrame{0.0};
    std::atomic<size_t> cbsMarginalizationGraphFactorCountPerRerunFrame{0u};
    std::atomic<size_t> cbsBpsamRootSizePerRerunFrame{0u};
    std::atomic<size_t> cbsBpsamActivePriorSlotsPerRerunFrame{0u};
    std::atomic<size_t> cbsMergeK2LSampleCounter{0u};
    enum class L2KOutgoingCovMode {
        kAsIsLocalAnchored = 0,
        kInitAnchorReplaced = 1,
        kQueryAnchorOnly = 2,
    };
    bool cbsL2KCovAuditEnable = false;
    size_t cbsL2KCovAuditMaxSamples = 20u;
    double cbsL2KCovAuditReplacementTransVariance = 1.0;
    double cbsL2KCovAuditQueryAnchorVariance = 1e6;
    L2KOutgoingCovMode cbsL2KOutgoingCovMode = L2KOutgoingCovMode::kAsIsLocalAnchored;
    std::string cbsL2KOutgoingCovModeLabel = "A_as_is_local_anchored";
    std::string cbsL2KOutgoingCovSourcePath = "LiORF::bpsam_getBeliefs_local_marginalization";
    std::string cbsL2KOutgoingCovAnchorMode = "local_anchored_perm_init_prior";
    double cbsL2KOutgoingCovarianceScale = 1.0;
    std::unordered_set<size_t> cbsL2KCovAuditLoggedPoseIndices;
    std::atomic<size_t> cbsL2KCovAuditSampleCounter{0u};
    bool cbsExternalExchangeInBodyFrame = true;
    bool cbsConjugateBodyFrameConversion = true;
    gtsam::Pose3 cbsLidarPoseBody = gtsam::Pose3();
    gtsam::Pose3 cbsBodyPoseLidar = gtsam::Pose3();
    gtsam::Matrix6 cbsAdjointLidarPoseBody = gtsam::Matrix6::Identity();
    gtsam::Matrix6 cbsAdjointBodyPoseLidar = gtsam::Matrix6::Identity();
    bool rerunVisualizerEnable = false;
    std::string rerunRecordingId;
    std::string rerunHost = "auto";
    float rerunLocalMapLeafSize = 1.0f;
    size_t rerunLocalMapMaxPoints = 10000u;
    bool rerunFactorGraphEnable = true;
    std::unique_ptr<aria::viz::VisualizerRerun> rerunVisualizer;

    enum class BeliefMatchFailureReason {
        kNone = 0,
        kNoLocalTimestamp = 1,
        kInvalidStamp = 2,
        kTimestampMismatch = 3,
    };

    cbs::AgentId resolveAgentId(const std::string& id) const
    {
        if (!id.empty()) {
            bool numeric = std::all_of(id.begin(), id.end(), [](unsigned char c) { return std::isdigit(c); });
            if (numeric) {
                try {
                    const int numericId = std::stoi(id);
                    return static_cast<cbs::AgentId>('a' + ((numericId % 26 + 26) % 26));
                } catch (const std::exception&) {
                    ROS_WARN_STREAM("Failed to parse numeric cbsAgentId '" << id << "', falling back to hashed id.");
                }
            }
            if (id.size() == 1) {
                return static_cast<cbs::AgentId>(id.front());
            }
        }

        const size_t hashValue = std::hash<std::string>{}(id);
        return static_cast<cbs::AgentId>('a' + (hashValue % 26));
    }

    Key poseKeyFromLocalIndex(size_t localIndex) const
    {
        return cbs::toPoseKey(selfAgentId, localIndex);
    }

    Key ensurePoseKeyForLocalIndex(size_t localIndex)
    {
        while (localPoseKeys.size() <= localIndex) {
            const size_t newIndex = localPoseKeys.size();
            const Key key = poseKeyFromLocalIndex(newIndex);
            localPoseKeys.push_back(key);
            localPoseKeyToIndex[key] = newIndex;
            localPoseTimestampsSec.push_back(-1.0);
        }
        return localPoseKeys[localIndex];
    }

    bool getPoseKeyForLocalIndex(size_t localIndex, Key* key) const
    {
        if (localIndex >= localPoseKeys.size()) {
            return false;
        }
        *key = localPoseKeys[localIndex];
        return true;
    }

    void setPoseTimestamp(size_t localIndex, double stampSec)
    {
        if (localIndex >= localPoseTimestampsSec.size()) {
            localPoseTimestampsSec.resize(localIndex + 1, -1.0);
        }
        localPoseTimestampsSec[localIndex] = stampSec;
    }

    size_t activeBeliefWindowStartIndex() const
    {
        return localPoseKeys.size() > beliefExchangeWindowSize ? localPoseKeys.size() - beliefExchangeWindowSize : 0u;
    }

    bool isLocalIndexInBeliefWindow(size_t localIndex) const
    {
        return localIndex >= activeBeliefWindowStartIndex() && localIndex < localPoseKeys.size();
    }

    KeySet activeBeliefWindowKeys() const
    {
        KeySet keys;
        const size_t startIndex = activeBeliefWindowStartIndex();
        for (size_t i = startIndex; i < localPoseKeys.size(); ++i) {
            keys.insert(localPoseKeys[i]);
        }
        return keys;
    }

    bool findClosestLocalIndexByTimestamp(double stampSec,
                                          size_t* localIndex,
                                          BeliefMatchFailureReason* reason = nullptr) const
    {
        if (reason) {
            *reason = BeliefMatchFailureReason::kNone;
        }

        if (localPoseTimestampsSec.empty()) {
            if (reason) {
                *reason = BeliefMatchFailureReason::kNoLocalTimestamp;
            }
            return false;
        }

        if (!std::isfinite(stampSec) || stampSec <= 0.0) {
            if (reason) {
                *reason = BeliefMatchFailureReason::kInvalidStamp;
            }
            return false;
        }

        size_t bestIndex = 0;
        double bestAbsDt = std::numeric_limits<double>::max();
        bool hasCandidate = false;
        for (size_t i = 0; i < localPoseTimestampsSec.size(); ++i) {
            if (localPoseTimestampsSec[i] < 0.0) {
                continue;
            }
            hasCandidate = true;
            const double dt = std::abs(localPoseTimestampsSec[i] - stampSec);
            if (dt < bestAbsDt) {
                bestAbsDt = dt;
                bestIndex = i;
            }
        }

        if (!hasCandidate) {
            if (reason) {
                *reason = BeliefMatchFailureReason::kNoLocalTimestamp;
            }
            return false;
        }

        if (bestAbsDt > beliefTimestampToleranceSec) {
            if (reason) {
                *reason = BeliefMatchFailureReason::kTimestampMismatch;
            }
            return false;
        }
        *localIndex = bestIndex;
        return true;
    }

    static gtsam::Matrix6 beliefArrayToMatrix6(const std::array<double, 36>& covarianceArray)
    {
        gtsam::Matrix6 covariance = gtsam::Matrix6::Zero();
        for (size_t r = 0; r < 6; ++r) {
            for (size_t c = 0; c < 6; ++c) {
                covariance(r, c) = covarianceArray[r * 6 + c];
            }
        }
        return covariance;
    }

    static void beliefMatrix6ToArray(const gtsam::Matrix6& covariance,
                                     std::array<double, 36>* covarianceArray)
    {
        if (!covarianceArray) {
            return;
        }
        for (size_t r = 0; r < 6; ++r) {
            for (size_t c = 0; c < 6; ++c) {
                (*covarianceArray)[r * 6 + c] = covariance(r, c);
            }
        }
    }

    static gtsam::Vector6 beliefArrayToVector6(const std::array<double, 6>& muArray)
    {
        gtsam::Vector6 mu = gtsam::Vector6::Zero();
        for (size_t i = 0; i < 6; ++i) {
            mu(i) = muArray[i];
        }
        return mu;
    }

    static void beliefVector6ToArray(const gtsam::Vector6& mu,
                                     std::array<double, 6>* muArray)
    {
        if (!muArray) {
            return;
        }
        for (size_t i = 0; i < 6; ++i) {
            (*muArray)[i] = mu(i);
        }
    }

    static double beliefTraceFromArray(const std::array<double, 36>& covarianceArray)
    {
        double trace = 0.0;
        for (size_t i = 0; i < 6; ++i) {
            trace += covarianceArray[i * 6 + i];
        }
        return trace;
    }

    static double beliefTraceFromMatrix(const Eigen::MatrixXd& covariance)
    {
        return covariance.trace();
    }

    static std::string sanitizeCsvToken(std::string token)
    {
        std::replace(token.begin(), token.end(), ',', '_');
        std::replace(token.begin(), token.end(), ' ', '_');
        std::replace(token.begin(), token.end(), '\n', '_');
        std::replace(token.begin(), token.end(), '\r', '_');
        return token.empty() ? "na" : token;
    }

    static std::string vector6ToToken(const gtsam::Vector6& vector)
    {
        std::ostringstream oss;
        oss << "v[";
        for (size_t i = 0u; i < 6u; ++i) {
            if (i > 0u) {
                oss << ";";
            }
            oss << vector(i);
        }
        oss << "]";
        return oss.str();
    }

    static std::string matrix6ToToken(const gtsam::Matrix6& matrix)
    {
        std::ostringstream oss;
        oss << "m[";
        for (size_t r = 0u; r < 6u; ++r) {
            if (r > 0u) {
                oss << "|";
            }
            for (size_t c = 0u; c < 6u; ++c) {
                if (c > 0u) {
                    oss << ";";
                }
                oss << matrix(r, c);
            }
        }
        oss << "]";
        return oss.str();
    }

    static double minEigenvalueSymmetric(const gtsam::Matrix6& matrix)
    {
        Eigen::SelfAdjointEigenSolver<gtsam::Matrix6> eig(0.5 * (matrix + matrix.transpose()));
        if (eig.info() != Eigen::Success) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return eig.eigenvalues().minCoeff();
    }

    static double poseErrorNorm(const gtsam::Pose3& lhs, const gtsam::Pose3& rhs)
    {
        try {
            return gtsam::Pose3::Logmap(lhs.inverse() * rhs).norm();
        } catch (...) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    static std::string formatPoseKeyToken(cbs::AgentId agent, size_t poseIndex)
    {
        const char agentChar = static_cast<char>(agent);
        return std::string("p:") + agentChar + ":" + std::to_string(poseIndex);
    }

    static std::string normalizeCovModeToken(std::string modeToken)
    {
        std::transform(modeToken.begin(), modeToken.end(), modeToken.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return modeToken;
    }

    static std::string covModeToLabel(const L2KOutgoingCovMode mode)
    {
        switch (mode) {
            case L2KOutgoingCovMode::kAsIsLocalAnchored:
                return "A_as_is_local_anchored";
            case L2KOutgoingCovMode::kInitAnchorReplaced:
                return "B_init_anchor_replaced_with_query_solve";
            case L2KOutgoingCovMode::kQueryAnchorOnly:
                return "C_query_anchor_only_no_perm_init_anchor";
        }
        return "A_as_is_local_anchored";
    }

    static std::pair<std::string, std::string> covModeSemantics(const L2KOutgoingCovMode mode)
    {
        switch (mode) {
            case L2KOutgoingCovMode::kAsIsLocalAnchored:
                return {"LiORF::bpsam_getBeliefs_local_marginalization",
                        "local_anchored_perm_init_prior"};
            case L2KOutgoingCovMode::kInitAnchorReplaced:
                return {"LiORF::local_graph_init_anchor_replaced_query_solve",
                        "local_init_anchor_replaced_query_solve"};
            case L2KOutgoingCovMode::kQueryAnchorOnly:
                return {"LiORF::local_graph_query_anchor_only",
                        "local_query_anchor_only_no_perm_init_anchor"};
        }
        return {"LiORF::bpsam_getBeliefs_local_marginalization",
                "local_anchored_perm_init_prior"};
    }

    static L2KOutgoingCovMode parseCovModeToken(const std::string& modeToken)
    {
        const std::string token = normalizeCovModeToken(modeToken);
        if (token == "a" || token == "as_is" || token == "asis" || token == "baseline" ||
            token == "as_is_local_anchored" || token == "a_as_is_local_anchored") {
            return L2KOutgoingCovMode::kAsIsLocalAnchored;
        }
        if (token == "b" || token == "init_anchor_replaced" ||
            token == "b_init_anchor_replaced_with_query_solve") {
            return L2KOutgoingCovMode::kInitAnchorReplaced;
        }
        if (token == "c" || token == "query_anchor_only" ||
            token == "c_query_anchor_only_no_perm_init_anchor") {
            return L2KOutgoingCovMode::kQueryAnchorOnly;
        }
        return L2KOutgoingCovMode::kAsIsLocalAnchored;
    }

    static double safeHellingerDistance(const gtsam::Vector6& muA,
                                        const gtsam::Matrix6& covA,
                                        const gtsam::Vector6& muB,
                                        const gtsam::Matrix6& covB)
    {
        try {
            const gtsam::Pose3 poseA = gtsam::Pose3::Expmap(muA);
            const gtsam::Pose3 poseB = gtsam::Pose3::Expmap(muB);
            const gtsam::Vector6 delta = gtsam::Pose3::Logmap(poseA.inverse() * poseB);
            return gbp::Hellinger::hellingerDistanceGaussian(delta, covA, covB);
        } catch (...) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    static double safeMahalanobisDistance(const gtsam::Vector6& muA,
                                          const gtsam::Matrix6& covA,
                                          const gtsam::Vector6& muB)
    {
        try {
            const gtsam::Vector6 delta = muB - muA;
            const gtsam::Matrix6 covInv = covA.inverse();
            return static_cast<double>(delta.transpose() * covInv * delta);
        } catch (...) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    static double safeDeltaNorm(const gtsam::Vector6& muA,
                                const gtsam::Vector6& muB)
    {
        return (muB - muA).norm();
    }

    static gtsam::Matrix6 sanitizeBeliefCovariance(const gtsam::Matrix6& covariance)
    {
        gtsam::Matrix6 sym = 0.5 * (covariance + covariance.transpose());
        for (size_t i = 0; i < 6; ++i) {
            if (!std::isfinite(sym(i, i)) || sym(i, i) <= 1e-9) {
                sym(i, i) = 1e-3;
            }
        }
        return sym;
    }

    static Eigen::Matrix3d sanitizeTranslationCovariance(const Eigen::MatrixXd& poseCovariance)
    {
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity() * 1e-3;
        if (poseCovariance.rows() >= 6 && poseCovariance.cols() >= 6) {
            covariance = poseCovariance.block<3, 3>(3, 3);
        }

        covariance = 0.5 * (covariance + covariance.transpose());
        if (!covariance.allFinite()) {
            return Eigen::Matrix3d::Identity() * 1e-3;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(covariance);
        if (eig.info() != Eigen::Success) {
            return Eigen::Matrix3d::Identity() * 1e-3;
        }

        const Eigen::Vector3d eigenvalues =
            eig.eigenvalues().array().max(1e-9).matrix();
        return eig.eigenvectors() * eigenvalues.asDiagonal() * eig.eigenvectors().transpose();
    }

    struct L2KOutgoingCovAuditResult
    {
        double traceAsIs = std::numeric_limits<double>::quiet_NaN();
        double traceInitPriorReplaced = std::numeric_limits<double>::quiet_NaN();
        double traceQueryAnchorOnly = std::numeric_limits<double>::quiet_NaN();
        std::string modeBStatus = "unavailable";
        std::string modeCStatus = "unavailable";
    };

    bool computePoseMarginalCovariance(const gtsam::NonlinearFactorGraph& graph,
                                       const gtsam::Values& values,
                                       const Key poseKey,
                                       gtsam::Matrix6* covarianceOut) const
    {
        if (!covarianceOut || graph.empty() || !values.exists(poseKey)) {
            return false;
        }
        try {
            gtsam::Marginals marginals(graph, values, gtsam::Marginals::Factorization::CHOLESKY);
            const gtsam::Matrix poseCov = marginals.marginalCovariance(poseKey);
            if (poseCov.rows() != 6 || poseCov.cols() != 6 || !poseCov.allFinite()) {
                return false;
            }
            *covarianceOut = sanitizeBeliefCovariance(poseCov);
            return true;
        } catch (...) {
            return false;
        }
    }

    bool computePoseMarginalTrace(const gtsam::NonlinearFactorGraph& graph,
                                  const gtsam::Values& values,
                                  const Key poseKey,
                                  double* traceOut) const
    {
        if (!traceOut) {
            return false;
        }
        gtsam::Matrix6 cov;
        if (!computePoseMarginalCovariance(graph, values, poseKey, &cov)) {
            return false;
        }
        *traceOut = cov.trace();
        return true;
    }

    bool buildLocalGraphForOutgoingCovAudit(const Key queryPoseKey,
                                            gtsam::NonlinearFactorGraph* localGraph,
                                            gtsam::Values* localValues,
                                            Key* firstPoseKeyOut) const
    {
        if (!localGraph || !localValues || !firstPoseKeyOut || !bpsam) {
            return false;
        }
        localGraph->resize(0);
        localValues->clear();
        const gtsam::Values allValues = bpsam->calculateEstimate();
        if (!allValues.exists(queryPoseKey)) {
            return false;
        }

        const Key firstPoseKey = localPoseKeys.empty()
                                     ? static_cast<Key>(cbs::toPoseKey(selfAgentId, 0))
                                     : localPoseKeys.front();
        *firstPoseKeyOut = firstPoseKey;

        const auto& factors = bpsam->getFactorsUnsafe();
        localGraph->reserve(factors.size());
        for (size_t slot = 0u; slot < factors.size(); ++slot) {
            if (!factors.exists(slot)) {
                continue;
            }
            const auto& factor = factors.at(slot);
            if (!factor) {
                continue;
            }
            if (factor->keys().size() == 2 &&
                cbs::isAnchorBeliefFactor(factor)) {
                continue;
            }
            bool hasValues = true;
            for (const auto key : factor->keys()) {
                if (!allValues.exists(key)) {
                    hasValues = false;
                    break;
                }
            }
            if (!hasValues) {
                continue;
            }
            localGraph->push_back(factor);
            for (const auto key : factor->keys()) {
                cbs::insertOrAssign(*localValues, key, allValues.at(key));
            }
        }

        cbs::insertOrAssign(*localValues, queryPoseKey, allValues.at(queryPoseKey));
        if (allValues.exists(firstPoseKey)) {
            cbs::insertOrAssign(*localValues, firstPoseKey, allValues.at(firstPoseKey));
        }
        return !localGraph->empty();
    }

    bool buildModeBGraph(const gtsam::NonlinearFactorGraph& localGraph,
                         const gtsam::Values& localValues,
                         const Key firstPoseKey,
                         gtsam::NonlinearFactorGraph* modeGraph,
                         bool* removedInitPrior) const
    {
        if (!modeGraph) {
            return false;
        }
        modeGraph->resize(0);
        modeGraph->reserve(localGraph.size() + 1u);
        bool removed = false;
        for (size_t i = 0u; i < localGraph.size(); ++i) {
            if (!localGraph.exists(i)) {
                continue;
            }
            const auto& factor = localGraph.at(i);
            if (!factor) {
                continue;
            }
            const auto* posePrior = dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(factor.get());
            if (posePrior && posePrior->key() == firstPoseKey) {
                removed = true;
                continue;
            }
            modeGraph->push_back(factor);
        }
        if (localValues.exists(firstPoseKey)) {
            gtsam::Vector initVars(6);
            initVars << 1e-2, 1e-2, M_PI * M_PI,
                cbsL2KCovAuditReplacementTransVariance,
                cbsL2KCovAuditReplacementTransVariance,
                cbsL2KCovAuditReplacementTransVariance;
            auto replacementNoise = gtsam::noiseModel::Diagonal::Variances(initVars);
            modeGraph->emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                firstPoseKey,
                localValues.at<gtsam::Pose3>(firstPoseKey),
                replacementNoise);
        }
        if (removedInitPrior) {
            *removedInitPrior = removed;
        }
        return !modeGraph->empty();
    }

    bool buildModeCGraph(const gtsam::NonlinearFactorGraph& localGraph,
                         const gtsam::Values& localValues,
                         const Key queryPoseKey,
                         gtsam::NonlinearFactorGraph* modeGraph) const
    {
        if (!modeGraph) {
            return false;
        }
        modeGraph->resize(0);
        modeGraph->reserve(localGraph.size() + 1u);
        for (size_t i = 0u; i < localGraph.size(); ++i) {
            if (!localGraph.exists(i)) {
                continue;
            }
            const auto& factor = localGraph.at(i);
            if (!factor) {
                continue;
            }
            const auto* posePrior = dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(factor.get());
            if (posePrior) {
                continue;
            }
            modeGraph->push_back(factor);
        }
        if (localValues.exists(queryPoseKey)) {
            gtsam::Vector anchorVars = gtsam::Vector::Constant(6, cbsL2KCovAuditQueryAnchorVariance);
            auto queryAnchorNoise = gtsam::noiseModel::Diagonal::Variances(anchorVars);
            modeGraph->emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                queryPoseKey,
                localValues.at<gtsam::Pose3>(queryPoseKey),
                queryAnchorNoise);
        }
        return !modeGraph->empty();
    }

    bool computeL2KOutgoingCovarianceForMode(const Key queryPoseKey,
                                             const gbp::Gaussian& asIsBelief,
                                             const L2KOutgoingCovMode mode,
                                             gtsam::Matrix6* covarianceOut,
                                             std::string* statusOut) const
    {
        if (!covarianceOut) {
            return false;
        }

        if (mode == L2KOutgoingCovMode::kAsIsLocalAnchored) {
            *covarianceOut = sanitizeBeliefCovariance(asIsBelief.Sigma());
            if (statusOut) {
                *statusOut = "ok_as_is_local_anchored";
            }
            return true;
        }

        gtsam::NonlinearFactorGraph localGraph;
        gtsam::Values localValues;
        Key firstPoseKey = queryPoseKey;
        if (!buildLocalGraphForOutgoingCovAudit(queryPoseKey, &localGraph, &localValues, &firstPoseKey)) {
            if (statusOut) {
                *statusOut = "local_graph_unavailable";
            }
            return false;
        }

        gtsam::NonlinearFactorGraph modeGraph;
        if (mode == L2KOutgoingCovMode::kInitAnchorReplaced) {
            bool removedInitPrior = false;
            if (!buildModeBGraph(localGraph, localValues, firstPoseKey, &modeGraph, &removedInitPrior)) {
                if (statusOut) {
                    *statusOut = "mode_b_graph_unavailable";
                }
                return false;
            }
            if (!computePoseMarginalCovariance(modeGraph, localValues, queryPoseKey, covarianceOut)) {
                if (statusOut) {
                    *statusOut = removedInitPrior ? "failed_removed_and_replaced" : "failed_replaced_only";
                }
                return false;
            }
            if (statusOut) {
                *statusOut = removedInitPrior ? "ok_removed_and_replaced" : "ok_replaced_only";
            }
            return true;
        }

        if (mode == L2KOutgoingCovMode::kQueryAnchorOnly) {
            if (!buildModeCGraph(localGraph, localValues, queryPoseKey, &modeGraph)) {
                if (statusOut) {
                    *statusOut = "mode_c_graph_unavailable";
                }
                return false;
            }
            if (!computePoseMarginalCovariance(modeGraph, localValues, queryPoseKey, covarianceOut)) {
                if (statusOut) {
                    *statusOut = "failed_query_anchor_only";
                }
                return false;
            }
            if (statusOut) {
                *statusOut = "ok_query_anchor_only";
            }
            return true;
        }

        if (statusOut) {
            *statusOut = "unknown_mode";
        }
        return false;
    }

    L2KOutgoingCovAuditResult computeL2KOutgoingCovAudit(const Key queryPoseKey,
                                                         const gbp::Gaussian& asIsBelief) const
    {
        L2KOutgoingCovAuditResult result;
        gtsam::Matrix6 covA = sanitizeBeliefCovariance(asIsBelief.Sigma());
        result.traceAsIs = beliefTraceFromMatrix(covA);

        gtsam::Matrix6 covB;
        if (computeL2KOutgoingCovarianceForMode(queryPoseKey,
                                                asIsBelief,
                                                L2KOutgoingCovMode::kInitAnchorReplaced,
                                                &covB,
                                                &result.modeBStatus)) {
            result.traceInitPriorReplaced = covB.trace();
        }

        gtsam::Matrix6 covC;
        if (computeL2KOutgoingCovarianceForMode(queryPoseKey,
                                                asIsBelief,
                                                L2KOutgoingCovMode::kQueryAnchorOnly,
                                                &covC,
                                                &result.modeCStatus)) {
            result.traceQueryAnchorOnly = covC.trace();
        }

        return result;
    }

    static void computeCovarianceLogdetAndLambdaMin(const gtsam::Matrix6& covariance,
                                                    double* logdetOut,
                                                    double* lambdaMinOut)
    {
        if (logdetOut) {
            *logdetOut = std::numeric_limits<double>::quiet_NaN();
        }
        if (lambdaMinOut) {
            *lambdaMinOut = std::numeric_limits<double>::quiet_NaN();
        }
        Eigen::SelfAdjointEigenSolver<gtsam::Matrix6> eigSolver(0.5 * (covariance + covariance.transpose()));
        if (eigSolver.info() != Eigen::Success) {
            return;
        }
        gtsam::Vector6 eigvals = eigSolver.eigenvalues();
        for (size_t i = 0; i < 6; ++i) {
            if (!std::isfinite(eigvals(i)) || eigvals(i) <= 1e-12) {
                eigvals(i) = 1e-12;
            }
        }
        if (lambdaMinOut) {
            *lambdaMinOut = eigvals.minCoeff();
        }
        if (logdetOut) {
            *logdetOut = eigvals.array().log().sum();
        }
    }

    void convertIncomingExchangeOdomBeliefToLidarFrame(StampedOdomBelief* belief) const
    {
        if (!belief || !cbsExternalExchangeInBodyFrame) {
            return;
        }

        const gtsam::Pose3 relativeBody =
            gtsam::Pose3::Expmap(beliefArrayToVector6(belief->relativeMu));
        const gtsam::Pose3 relativeLidar =
            cbsConjugateBodyFrameConversion
                ? cbsLidarPoseBody.compose(relativeBody).compose(cbsBodyPoseLidar)
                : relativeBody.compose(cbsBodyPoseLidar);
        const gtsam::Matrix6 covarianceBody =
            sanitizeBeliefCovariance(beliefArrayToMatrix6(belief->covariance));
        const gtsam::Matrix6& exchangeToLidarAdjoint =
            cbsConjugateBodyFrameConversion
                ? cbsAdjointLidarPoseBody
                : cbsAdjointBodyPoseLidar;
        const gtsam::Matrix6 covarianceLidar =
            sanitizeBeliefCovariance(exchangeToLidarAdjoint * covarianceBody *
                                     exchangeToLidarAdjoint.transpose());
        beliefVector6ToArray(gtsam::Pose3::Logmap(relativeLidar), &belief->relativeMu);
        beliefMatrix6ToArray(covarianceLidar, &belief->covariance);
    }

    void convertOutgoingLidarOdomBeliefToExchangeFrame(StampedOdomBelief* belief) const
    {
        if (!belief || !cbsExternalExchangeInBodyFrame) {
            return;
        }

        const gtsam::Pose3 relativeLidar =
            gtsam::Pose3::Expmap(beliefArrayToVector6(belief->relativeMu));
        const gtsam::Pose3 relativeBody =
            cbsConjugateBodyFrameConversion
                ? cbsBodyPoseLidar.compose(relativeLidar).compose(cbsLidarPoseBody)
                : relativeLidar.compose(cbsLidarPoseBody);
        const gtsam::Matrix6 covarianceLidar =
            sanitizeBeliefCovariance(beliefArrayToMatrix6(belief->covariance));
        const gtsam::Matrix6& lidarToExchangeAdjoint =
            cbsConjugateBodyFrameConversion
                ? cbsAdjointBodyPoseLidar
                : cbsAdjointLidarPoseBody;
        const gtsam::Matrix6 covarianceBody =
            sanitizeBeliefCovariance(lidarToExchangeAdjoint * covarianceLidar *
                                     lidarToExchangeAdjoint.transpose());
        beliefVector6ToArray(gtsam::Pose3::Logmap(relativeBody), &belief->relativeMu);
        beliefMatrix6ToArray(covarianceBody, &belief->covariance);
    }

    void enqueueIncomingOdomBelief(const StampedOdomBelief& belief)
    {
        std::lock_guard<std::mutex> lock(mtxBeliefExchange);
        incomingStampedOdomBeliefs.push_back(belief);
    }

    void poseOdomBeliefInHandler(const liorf::pose_odom_belief_arrayConstPtr& msg)
    {
        for (const auto& beliefMsg : msg->beliefs) {
            const cbs::AgentId sourceAgent = static_cast<cbs::AgentId>(beliefMsg.source_agent);
            if (sourceAgent == selfAgentId) {
                continue;
            }

            StampedOdomBelief belief;
            belief.sourceAgent = sourceAgent;
            belief.fromPoseIndex = static_cast<size_t>(beliefMsg.from_pose_index);
            belief.toPoseIndex = static_cast<size_t>(beliefMsg.to_pose_index);
            belief.fromStampSec = beliefMsg.from_stamp_sec;
            belief.toStampSec = beliefMsg.to_stamp_sec > 0.0 ? beliefMsg.to_stamp_sec : beliefMsg.header.stamp.toSec();
            belief.senderTimestampNs = beliefMsg.header.stamp.toNSec();
            belief.senderFrameId = beliefMsg.header.frame_id;
            belief.relaxFactor = beliefMsg.relax_factor;

            for (size_t i = 0; i < belief.relativeMu.size(); ++i) {
                belief.relativeMu[i] = beliefMsg.relative_mu[i];
            }
            for (size_t i = 0; i < belief.covariance.size(); ++i) {
                belief.covariance[i] = beliefMsg.covariance[i];
            }

            convertIncomingExchangeOdomBeliefToLidarFrame(&belief);
            enqueueIncomingOdomBelief(belief);
            cbsBeliefsIncomingReceivedTotal.fetch_add(1u, std::memory_order_relaxed);
            cbsBeliefsIncomingReceivedPerRerunFrame.fetch_add(1u, std::memory_order_relaxed);
        }
    }

    void consumeIncomingOdomBeliefsIntoBpsam()
    {
        std::deque<StampedOdomBelief> pendingBeliefs;
        {
            std::lock_guard<std::mutex> lock(mtxBeliefExchange);
            if (incomingStampedOdomBeliefs.empty()) {
                return;
            }
            pendingBeliefs.swap(incomingStampedOdomBeliefs);
        }

        const size_t pendingCount = pendingBeliefs.size();
        cbsBeliefsIncomingDequeuedTotal.fetch_add(pendingCount, std::memory_order_relaxed);

        size_t numMatched = 0u;
        size_t numDropped = 0u;
        size_t numAddedToBpsam = 0u;
        size_t numRejectedByBpsam = 0u;
        size_t numRejectedInactiveWindow = 0u;
        size_t numRejectedShape = 0u;
        size_t numRejectedException = 0u;
        std::vector<RerunExternalOdomEdge> acceptedRerunExternalOdomEdges;

        for (const auto& incoming : pendingBeliefs) {
            size_t fromLocalIndex = 0u;
            size_t toLocalIndex = 0u;
            BeliefMatchFailureReason fromReason = BeliefMatchFailureReason::kNone;
            BeliefMatchFailureReason toReason = BeliefMatchFailureReason::kNone;
            const bool fromMatched =
                findClosestLocalIndexByTimestamp(incoming.fromStampSec, &fromLocalIndex, &fromReason);
            const bool toMatched =
                findClosestLocalIndexByTimestamp(incoming.toStampSec, &toLocalIndex, &toReason);
            if (!fromMatched || !toMatched) {
                ++numDropped;
                continue;
            }
            if (!isLocalIndexInBeliefWindow(fromLocalIndex) ||
                !isLocalIndexInBeliefWindow(toLocalIndex) ||
                fromLocalIndex >= toLocalIndex) {
                ++numRejectedByBpsam;
                ++numRejectedInactiveWindow;
                continue;
            }
            ++numMatched;

            cbs::BPSAM::CbsOdometryBelief odomBelief;
            odomBelief.source_agent = incoming.sourceAgent;
            odomBelief.from_pose_key = ensurePoseKeyForLocalIndex(fromLocalIndex);
            odomBelief.to_pose_key = ensurePoseKeyForLocalIndex(toLocalIndex);
            odomBelief.measured_from_to =
                gtsam::Pose3::Expmap(beliefArrayToVector6(incoming.relativeMu));
            odomBelief.covariance =
                sanitizeBeliefCovariance(beliefArrayToMatrix6(incoming.covariance));
            odomBelief.relax_factor = incoming.relaxFactor;

            std::vector<cbs::BPSAM::CbsOdometryBelief> singleBelief;
            singleBelief.push_back(std::move(odomBelief));
            const auto addResult = bpsam->addOdometryBeliefsDetailed(std::move(singleBelief));
            numAddedToBpsam += addResult.accepted;
            numRejectedByBpsam += addResult.rejected();
            numRejectedInactiveWindow += addResult.rejected_inactive_window;
            numRejectedShape += addResult.rejected_shape;
            numRejectedException += addResult.rejected_exception;
            for (const auto& detail : addResult.details) {
                if (detail.status == cbs::BPSAM::AddOdometryBeliefStatus::Accepted) {
                    acceptedRerunExternalOdomEdges.push_back(
                        RerunExternalOdomEdge{
                            detail.source_agent,
                            detail.from_pose_key,
                            detail.to_pose_key});
                }
                ROS_INFO_STREAM(
                    "CBS_BPSAM_ODOM_ADD_ROW_K2L,"
                    << formatPoseKeyToken(incoming.sourceAgent, incoming.fromPoseIndex) << "->"
                    << formatPoseKeyToken(incoming.sourceAgent, incoming.toPoseIndex) << ","
                    << formatPoseKeyToken(selfAgentId, fromLocalIndex) << "->"
                    << formatPoseKeyToken(selfAgentId, toLocalIndex) << ","
                    << static_cast<int>(detail.status) << ","
                    << detail.covariance_trace << ","
                    << sanitizeCsvToken(detail.message));
            }
        }

        {
            std::lock_guard<std::mutex> lock(mtxRerunCbsVisualization);
            rerunExternalOdomEdges = std::move(acceptedRerunExternalOdomEdges);
        }

        cbsBeliefsIncomingMatchedByTimestampTotal.fetch_add(numMatched, std::memory_order_relaxed);
        cbsBeliefsIncomingDroppedTimestampMismatchTotal.fetch_add(numDropped, std::memory_order_relaxed);
        cbsBeliefsIncomingAddedToBpsamTotal.fetch_add(numAddedToBpsam, std::memory_order_relaxed);
        cbsBeliefsIncomingAddedToBpsamPerRerunFrame.fetch_add(numAddedToBpsam, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedByBpsamTotal.fetch_add(numRejectedByBpsam, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedShapeTotal.fetch_add(numRejectedShape, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedExceptionTotal.fetch_add(numRejectedException, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedShapePerRerunFrame.fetch_add(numRejectedShape, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedExceptionPerRerunFrame.fetch_add(numRejectedException, std::memory_order_relaxed);

        ROS_INFO_STREAM_THROTTLE(
            1.0,
            "LiORF CBS incoming odometry flow: dequeued=" << pendingCount
            << " matched=" << numMatched
            << " dropped=" << numDropped
            << " bpsam(added=" << numAddedToBpsam
            << ",rejected=" << numRejectedByBpsam
            << ",inactive_window=" << numRejectedInactiveWindow
            << ",shape=" << numRejectedShape
            << ",exception=" << numRejectedException << ")");
    }

    void publishOutgoingOdomBeliefs()
    {
        if (!cbsBeliefBridgeEnable) {
            return;
        }

        std::vector<StampedOdomBelief> beliefsToPublish;
        {
            std::lock_guard<std::mutex> lock(mtxBeliefExchange);
            beliefsToPublish = outgoingStampedOdomBeliefs;
        }

        liorf::pose_odom_belief_array msg;
        msg.header.stamp = timeLaserInfoStamp;
        msg.header.frame_id = odometryFrame;
        msg.beliefs.reserve(beliefsToPublish.size());
        for (const auto& belief : beliefsToPublish) {
            liorf::pose_odom_belief beliefMsg;
            beliefMsg.header = msg.header;
            beliefMsg.source_agent = static_cast<uint8_t>(belief.sourceAgent);
            beliefMsg.from_pose_index = static_cast<uint32_t>(belief.fromPoseIndex);
            beliefMsg.to_pose_index = static_cast<uint32_t>(belief.toPoseIndex);
            beliefMsg.from_stamp_sec = belief.fromStampSec;
            beliefMsg.to_stamp_sec = belief.toStampSec;
            beliefMsg.relax_factor = belief.relaxFactor;
            for (size_t i = 0; i < belief.relativeMu.size(); ++i) {
                beliefMsg.relative_mu[i] = belief.relativeMu[i];
            }
            for (size_t i = 0; i < belief.covariance.size(); ++i) {
                beliefMsg.covariance[i] = belief.covariance[i];
            }
            msg.beliefs.push_back(beliefMsg);
        }

        pubPoseOdomBeliefsOut.publish(msg);
        cbsBeliefsOutgoingPublishedTotal.fetch_add(msg.beliefs.size(), std::memory_order_relaxed);
        cbsBeliefsOutgoingPublishedPerRerunFrame.fetch_add(msg.beliefs.size(), std::memory_order_relaxed);
        ROS_INFO_STREAM_THROTTLE(
            1.0,
            "LiORF CBS outgoing odometry flow: published=" << msg.beliefs.size()
            << " totals(prepared="
            << cbsBeliefsOutgoingPreparedTotal.load(std::memory_order_relaxed)
            << ",published="
            << cbsBeliefsOutgoingPublishedTotal.load(std::memory_order_relaxed)
            << ")");
    }

    void refreshOutgoingBeliefs()
    {
        if (localPoseKeys.empty()) {
            return;
        }

        KeySet requestKeys = activeBeliefWindowKeys();
        if (requestKeys.empty()) {
            return;
        }

        bpsam->setMarginalizationGraph(cbs::BPSAM::MarginalizationType::LOCAL);
        cbsMarginalizationGraphFactorCountPerRerunFrame.store(
            bpsam->marginalizationGraphFactorCount(),
            std::memory_order_relaxed);
        const auto cbsGetBeliefsStart = std::chrono::steady_clock::now();
        auto outgoing =
            bpsam->getOdometryBeliefs(requestKeys, static_cast<cbs::AgentId>('k'));
        atomicAddRelaxed(&cbsBeliefGenerationTimeMsPerRerunFrame,
                         elapsedMs(cbsGetBeliefsStart));

        std::vector<StampedOdomBelief> outgoingStamped;
        outgoingStamped.reserve(outgoing.size());
        for (const auto& odom : outgoing) {
            auto fromIt = localPoseKeyToIndex.find(odom.from_pose_key);
            auto toIt = localPoseKeyToIndex.find(odom.to_pose_key);
            if (fromIt == localPoseKeyToIndex.end() ||
                toIt == localPoseKeyToIndex.end()) {
                continue;
            }
            const size_t fromIndex = fromIt->second;
            const size_t toIndex = toIt->second;
            if (fromIndex >= localPoseTimestampsSec.size() ||
                toIndex >= localPoseTimestampsSec.size() ||
                localPoseTimestampsSec[fromIndex] < 0.0 ||
                localPoseTimestampsSec[toIndex] < 0.0) {
                continue;
            }

            StampedOdomBelief stampedBelief;
            stampedBelief.sourceAgent = selfAgentId;
            stampedBelief.fromPoseIndex = fromIndex;
            stampedBelief.toPoseIndex = toIndex;
            stampedBelief.fromStampSec = localPoseTimestampsSec[fromIndex];
            stampedBelief.toStampSec = localPoseTimestampsSec[toIndex];
            stampedBelief.senderTimestampNs =
                static_cast<uint64_t>(std::llround(stampedBelief.toStampSec * 1e9));
            stampedBelief.senderFrameId = odometryFrame;
            stampedBelief.relaxFactor = odom.relax_factor;
            beliefVector6ToArray(gtsam::Pose3::Logmap(odom.measured_from_to),
                                 &stampedBelief.relativeMu);
            beliefMatrix6ToArray(sanitizeBeliefCovariance(odom.covariance),
                                 &stampedBelief.covariance);
            convertOutgoingLidarOdomBeliefToExchangeFrame(&stampedBelief);
            outgoingStamped.push_back(stampedBelief);
        }

        {
            std::lock_guard<std::mutex> lock(mtxBeliefExchange);
            outgoingStampedOdomBeliefs = std::move(outgoingStamped);
        }
        cbsBeliefsOutgoingPreparedTotal.fetch_add(
            outgoingStampedOdomBeliefs.size(), std::memory_order_relaxed);
        publishOutgoingOdomBeliefs();
    }

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
    ros::Publisher pubPoseOdomBeliefsOut;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;
    ros::Subscriber subPoseOdomBeliefsIn;

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
        CHECK(!useGPS,
              "LiORF GPS factors are not supported with the tight initial "
              "pose prior used for CBSMS uncertainty visualization. Disable "
              "/liorf/useGPS or restore a GPS-compatible initial prior.");

        std::string cbsAgentIdStr;
        nh.param<std::string>("liorf/cbsAgentId", cbsAgentIdStr, robot_id);
        selfAgentId = resolveAgentId(cbsAgentIdStr);

        bool cbsEnableBeliefDcs = false;
        double cbsBeliefSimilarityThreshold = 0.01;
        double cbsDReset = 0.1;
        double cbsK2LOdomFactorCovarianceScale = 1.0;
        int cbsBeliefWindow = 30;
        int cbsBeliefMaxRootSizeParam = 60;
        nh.param<bool>("liorf/cbsEnableBeliefDcs", cbsEnableBeliefDcs, false);
        nh.param<double>("liorf/cbsBeliefSimilarityThreshold", cbsBeliefSimilarityThreshold, 0.01);
        nh.param<int>("liorf/cbsBeliefExchangeWindowSize", cbsBeliefWindow, 30);
        nh.param<int>("liorf/cbsBeliefMaxRootSize", cbsBeliefMaxRootSizeParam, 60);
        nh.param<double>("liorf/cbsBeliefTimestampToleranceSec", beliefTimestampToleranceSec, 0.05);
        nh.param<bool>("liorf/cbsBeliefRejectFirstMessage", cbsBeliefRejectFirstMessage, true);
        nh.param<bool>("liorf/cbsEnableSoftReset", cbsEnableSoftReset, true);
        nh.param<bool>("liorf/cbsUseRawPreviousBeliefGate", cbsUseRawPreviousBeliefGate, false);
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
        nh.param<bool>("liorf/cbsBeliefBridgeEnable", cbsBeliefBridgeEnable, true);
        nh.param<std::string>("liorf/cbsOdomBeliefInTopic",
                              cbsOdomBeliefInTopic,
                              "liorf/cbs/odom_belief_in");
        nh.param<std::string>("liorf/cbsOdomBeliefOutTopic",
                              cbsOdomBeliefOutTopic,
                              "liorf/cbs/odom_belief_out");
        nh.param<bool>("liorf/cbsExternalExchangeInBodyFrame", cbsExternalExchangeInBodyFrame, true);
        nh.param<bool>("liorf/cbsConjugateBodyFrameConversion",
                       cbsConjugateBodyFrameConversion,
                       true);
        nh.param<bool>("liorf/cbsL2KCovAuditEnable", cbsL2KCovAuditEnable, false);
        int cbsL2KCovAuditMaxSamplesInt = 20;
        nh.param<int>("liorf/cbsL2KCovAuditMaxSamples", cbsL2KCovAuditMaxSamplesInt, 20);
        nh.param<double>("liorf/cbsL2KCovAuditReplacementTransVariance",
                         cbsL2KCovAuditReplacementTransVariance,
                         1.0);
        nh.param<double>("liorf/cbsL2KCovAuditQueryAnchorVariance",
                         cbsL2KCovAuditQueryAnchorVariance,
                         1e6);
        std::string cbsL2KOutgoingCovModeToken = "A_as_is_local_anchored";
        nh.param<std::string>("liorf/cbsL2KOutgoingCovarianceMode",
                              cbsL2KOutgoingCovModeToken,
                              "A_as_is_local_anchored");
        nh.param<double>("liorf/cbsL2KOutgoingCovarianceScale",
                         cbsL2KOutgoingCovarianceScale,
                         1.0);
        if (!std::isfinite(cbsL2KOutgoingCovarianceScale) ||
            cbsL2KOutgoingCovarianceScale <= 0.0) {
            ROS_WARN_STREAM("Invalid LiORF L2K outgoing covariance scale "
                            << cbsL2KOutgoingCovarianceScale
                            << "; falling back to 1.0");
            cbsL2KOutgoingCovarianceScale = 1.0;
        }
        cbsL2KOutgoingCovMode = parseCovModeToken(cbsL2KOutgoingCovModeToken);
        cbsL2KOutgoingCovModeLabel = covModeToLabel(cbsL2KOutgoingCovMode);
        const auto [covSourcePath, covAnchorMode] = covModeSemantics(cbsL2KOutgoingCovMode);
        cbsL2KOutgoingCovSourcePath = covSourcePath;
        cbsL2KOutgoingCovAnchorMode = covAnchorMode;
        cbsL2KCovAuditMaxSamples = static_cast<size_t>(std::max(1, cbsL2KCovAuditMaxSamplesInt));
        beliefExchangeWindowSize = std::max(1, cbsBeliefWindow);
        cbsBeliefMaxRootSize =
            static_cast<size_t>(std::max(0, cbsBeliefMaxRootSizeParam));

        nh.param<bool>("liorf/rerunVisualizerEnable", rerunVisualizerEnable, false);
        nh.param<std::string>("liorf/rerunRecordingId", rerunRecordingId, "");
        nh.param<std::string>(
            "liorf/rerunHost",
            rerunHost,
            "auto");
        if (rerunHost.empty() || rerunHost == "auto") {
            rerunHost = defaultRerunHost();
        }
        double rerunLocalMapLeafSizeParam = 1.0;
        int rerunLocalMapMaxPointsParam = 10000;
        nh.param<double>("liorf/rerunLocalMapLeafSize",
                         rerunLocalMapLeafSizeParam,
                         1.0);
        nh.param<int>("liorf/rerunLocalMapMaxPoints",
                      rerunLocalMapMaxPointsParam,
                      10000);
        nh.param<bool>("liorf/rerunFactorGraphEnable",
                       rerunFactorGraphEnable,
                       true);
        rerunLocalMapLeafSize = static_cast<float>(
            std::max(0.0, rerunLocalMapLeafSizeParam));
        rerunLocalMapMaxPoints =
            static_cast<size_t>(std::max(1, rerunLocalMapMaxPointsParam));
        if (rerunRecordingId.empty()) {
            ros::param::param<std::string>("/cbsms/rerun_recording_id", rerunRecordingId, "");
        }
        if (rerunRecordingId.empty()) {
            rerunRecordingId = makeRerunRecordingId("liorf");
        }
        if (rerunVisualizerEnable) {
            aria::viz::VisualizerRerun::Params rerunParams("cbsms", rerunRecordingId, rerunHost);
            rerunVisualizer.reset(new aria::viz::VisualizerRerun(rerunParams));
            ROS_INFO_STREAM("LiORF Rerun visualizer enabled. recording_id='"
                            << rerunRecordingId << "', host='" << rerunHost << "'.");
        }

        if (cbsExternalExchangeInBodyFrame) {
            cbsLidarPoseBody = gtsam::Pose3(gtsam::Rot3(extRot), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));
            cbsBodyPoseLidar = cbsLidarPoseBody.inverse();
            cbsAdjointLidarPoseBody = cbsLidarPoseBody.AdjointMap();
            cbsAdjointBodyPoseLidar = cbsBodyPoseLidar.AdjointMap();
            ROS_INFO_STREAM("LiORF CBS exchange semantic: world->body (imu). "
                            << "conjugate_origin="
                            << (cbsConjugateBodyFrameConversion ? "true" : "false")
                            << ". "
                            << "Using lidar->body extrinsic from config. t_l_b=["
                            << extTrans.x() << ", " << extTrans.y() << ", " << extTrans.z() << "]");
        } else {
            cbsLidarPoseBody = gtsam::Pose3();
            cbsBodyPoseLidar = gtsam::Pose3();
            cbsAdjointLidarPoseBody = gtsam::Matrix6::Identity();
            cbsAdjointBodyPoseLidar = gtsam::Matrix6::Identity();
            ROS_INFO("LiORF CBS exchange semantic: world->lidar (legacy pass-through).");
        }

        cbs::BPSAM::Params parameters;
        parameters.robot_id = selfAgentId;
        parameters.sam_params_.relinearizeThreshold = 0.1;
        parameters.sam_params_.relinearizeSkip = 1;
        ISAM2GaussNewtonParams gaussNewtonParams;
        parameters.sam_params_.optimizationParams = gaussNewtonParams;
        parameters.enable_belief_dcs = cbsEnableBeliefDcs;
        parameters.reject_first_message = cbsBeliefRejectFirstMessage;
        parameters.belief_similarity_threshold = cbsBeliefSimilarityThreshold;
        parameters.gbp_update_params.enable_soft_reset = cbsEnableSoftReset;
        parameters.gbp_update_params.d_reset = cbsDReset;
        parameters.use_raw_previous_belief_gate = cbsUseRawPreviousBeliefGate;
        parameters.use_temporary_cbs_linear_factors =
            cbsUseTemporaryCbsLinearFactors;
        parameters.temporary_linear_already_applied_gate_enable =
            cbsTemporaryLinearAlreadyAppliedGateEnable;
        parameters.temporary_linear_already_applied_metric_threshold =
            cbsTemporaryLinearAlreadyAppliedMetricThreshold;
        parameters.temporary_linear_already_applied_dmu_threshold =
            cbsTemporaryLinearAlreadyAppliedDmuThreshold;
        parameters.temporary_linear_already_applied_cov_rel_threshold =
            cbsTemporaryLinearAlreadyAppliedCovRelThreshold;
        if (std::isfinite(cbsK2LOdomFactorCovarianceScale) &&
            cbsK2LOdomFactorCovarianceScale > 0.0) {
            parameters.external_factor_covariance_scale_by_source[static_cast<cbs::AgentId>('k')] =
                cbsK2LOdomFactorCovarianceScale;
        }
        bpsam.reset(new cbs::BPSAM(parameters));

        ROS_INFO_STREAM("LiORF BPSAM backend agent id: '" << static_cast<char>(selfAgentId) << "'");
        ROS_INFO_STREAM("LiORF mapping optimization frequency: "
                        << mappingOptimizationFrequency
                        << " Hz (interval=" << mappingProcessInterval << " s)");
        ROS_INFO_STREAM("LiORF BPSAM reject-first-message gate: "
                        << (cbsBeliefRejectFirstMessage ? "ON" : "OFF"));
        ROS_INFO_STREAM("LiORF BPSAM soft reset: "
                        << (cbsEnableSoftReset ? "ON" : "OFF"));
        ROS_INFO_STREAM("LiORF BPSAM raw-previous receiver gate: "
                        << (cbsUseRawPreviousBeliefGate ? "ON" : "OFF"));
        ROS_INFO_STREAM("LiORF BPSAM temporary CBS linear odometry factors: "
                        << (cbsUseTemporaryCbsLinearFactors ? "ON" : "OFF"));
        ROS_INFO_STREAM("LiORF BPSAM d_reset: " << cbsDReset);
        ROS_INFO_STREAM("LiORF K2L final odometry covariance scale: "
                        << cbsK2LOdomFactorCovarianceScale);
        ROS_INFO_STREAM("LiORF CBS belief window: " << beliefExchangeWindowSize
                        << " poses, max iSAM2 root size="
                        << cbsBeliefMaxRootSize);
        ROS_INFO_STREAM("LiORF L2K covariance decomposition audit: "
                        << (cbsL2KCovAuditEnable ? "ENABLED" : "DISABLED")
                        << " max_samples=" << cbsL2KCovAuditMaxSamples
                        << " replacement_trans_variance=" << cbsL2KCovAuditReplacementTransVariance
                        << " query_anchor_variance=" << cbsL2KCovAuditQueryAnchorVariance);
        ROS_INFO_STREAM("LiORF L2K outgoing covariance mode: "
                        << cbsL2KOutgoingCovModeLabel
                        << " source_path=" << cbsL2KOutgoingCovSourcePath
                        << " anchor_mode=" << cbsL2KOutgoingCovAnchorMode
                        << " scale=" << cbsL2KOutgoingCovarianceScale);

        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/trajectory", 1);
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/map_global", 1);
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("liorf/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("liorf/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("liorf/mapping/path", 1);

        subCloud = nh.subscribe<liorf::cloud_info>("liorf/deskew/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        if (useGPS) {
            subGPS = nh.subscribe<sensor_msgs::NavSatFix>(gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        }
        subLoop  = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());
        if (cbsBeliefBridgeEnable) {
            subPoseOdomBeliefsIn = nh.subscribe<liorf::pose_odom_belief_array>(cbsOdomBeliefInTopic, 50, &mapOptimization::poseOdomBeliefInHandler, this, ros::TransportHints().tcpNoDelay());
            pubPoseOdomBeliefsOut = nh.advertise<liorf::pose_odom_belief_array>(cbsOdomBeliefOutTopic, 50);
        }

        srvSaveMap  = nh.advertiseService("liorf/save_map", &mapOptimization::saveMapService, this);

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/liorf/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/map_local", 1);
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("liorf/mapping/cloud_registered_raw", 1);

        pubSLAMInfo           = nh.advertise<liorf::cloud_info>("liorf/mapping/slam_info", 1);
        pubGpsOdom            = nh.advertise<nav_msgs::Odometry> ("liorf/mapping/gps_odom", 1);

        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterLocalMapSurf.setLeafSize(surroundingKeyframeMapLeafSize, surroundingKeyframeMapLeafSize, surroundingKeyframeMapLeafSize);
        downSizeFilterICP.setLeafSize(loopClosureICPSurfLeafSize, loopClosureICPSurfLeafSize, loopClosureICPSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        localPoseKeys.clear();
        localPoseKeyToIndex.clear();
        localPoseTimestampsSec.clear();
        cbsBeliefsIncomingReceivedTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingDequeuedTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingMatchedByIndexTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingMatchedByTimestampTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingDroppedNoLocalTimestampTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingDroppedInvalidStampTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingDroppedTimestampMismatchTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingAddedToBpsamTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedByBpsamTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedFirstMessageTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedUpdateStatusTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedShapeTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedExceptionTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsOutgoingPreparedTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsOutgoingPublishedTotal.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingReceivedPerRerunFrame.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingAddedToBpsamPerRerunFrame.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedFirstMessagePerRerunFrame.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedUpdateStatusPerRerunFrame.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedShapePerRerunFrame.store(0u, std::memory_order_relaxed);
        cbsBeliefsIncomingRejectedExceptionPerRerunFrame.store(0u, std::memory_order_relaxed);
        cbsBeliefsOutgoingPublishedPerRerunFrame.store(0u, std::memory_order_relaxed);
        bpsamOptimizationTimeMsPerRerunFrame.store(0.0, std::memory_order_relaxed);
        cbsBeliefGenerationTimeMsPerRerunFrame.store(0.0, std::memory_order_relaxed);
        cbsMarginalizationGraphFactorCountPerRerunFrame.store(0u, std::memory_order_relaxed);
        cbsBpsamRootSizePerRerunFrame.store(0u, std::memory_order_relaxed);
        cbsBpsamActivePriorSlotsPerRerunFrame.store(0u, std::memory_order_relaxed);
        cbsL2KCovAuditSampleCounter.store(0u, std::memory_order_relaxed);
        cbsL2KCovAuditLoggedPoseIndices.clear();

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
        if (!useGPS)
            return;

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

    void publishRerunFrame()
    {
        if (!rerunVisualizer || cloudKeyPoses3D->points.empty()) {
            return;
        }

        PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
        rerunVisualizer->setTimeNSec(timeLaserInfoStamp.toNSec());
        rerunVisualizer->drawTf(
            "liorf/lidar_link", pclPointTogtsamPose3(thisPose6D), 0.75f);
        const Eigen::Matrix3d currentPoseCovariance =
            sanitizeTranslationCovariance(poseCovariance);
        rerunVisualizer->drawUncertainty(
            "liorf/current_pose/uncertainty",
            pclPointTogtsamPose3(thisPose6D),
            currentPoseCovariance,
            Eigen::Vector4f(255.f, 160.f, 0.f, 160.f),
            1.25f);
        rerunVisualizer->drawScalar(
            "liorf/current_pose/liorf_uncertainty_frobenius_norm",
            currentPoseCovariance.norm());
        rerunVisualizer->drawScalar(
            "liorf/timing/optimization_ms",
            bpsamOptimizationTimeMsPerRerunFrame.exchange(0.0, std::memory_order_relaxed));
        if (cbsBeliefBridgeEnable) {
            rerunVisualizer->drawScalar(
                "liorf/cbs/beliefs/published_per_update",
                cbsBeliefsOutgoingPublishedPerRerunFrame.exchange(0u, std::memory_order_relaxed));
            rerunVisualizer->drawScalar(
                "liorf/cbs/beliefs/received_per_update",
                cbsBeliefsIncomingReceivedPerRerunFrame.exchange(0u, std::memory_order_relaxed));
            rerunVisualizer->drawScalar(
                "liorf/cbs/beliefs/added_to_factor_graph_per_update",
                cbsBeliefsIncomingAddedToBpsamPerRerunFrame.exchange(0u, std::memory_order_relaxed));
            rerunVisualizer->drawScalar(
                "liorf/cbs/beliefs/rejected_first_message_per_update",
                cbsBeliefsIncomingRejectedFirstMessagePerRerunFrame.exchange(0u, std::memory_order_relaxed));
            rerunVisualizer->drawScalar(
                "liorf/cbs/beliefs/rejected_update_status_per_update",
                cbsBeliefsIncomingRejectedUpdateStatusPerRerunFrame.exchange(0u, std::memory_order_relaxed));
            rerunVisualizer->drawScalar(
                "liorf/cbs/beliefs/rejected_shape_per_update",
                cbsBeliefsIncomingRejectedShapePerRerunFrame.exchange(0u, std::memory_order_relaxed));
            rerunVisualizer->drawScalar(
                "liorf/cbs/beliefs/rejected_exception_per_update",
                cbsBeliefsIncomingRejectedExceptionPerRerunFrame.exchange(0u, std::memory_order_relaxed));
            rerunVisualizer->drawScalar(
                "liorf/cbs/timing/belief_generation_ms",
                cbsBeliefGenerationTimeMsPerRerunFrame.exchange(0.0, std::memory_order_relaxed));
            rerunVisualizer->drawScalar(
                "liorf/cbs/marginalization_graph/factor_count",
                cbsMarginalizationGraphFactorCountPerRerunFrame.exchange(0u, std::memory_order_relaxed));
            rerunVisualizer->drawScalar(
                "liorf/cbs/isam2/root_size",
                cbsBpsamRootSizePerRerunFrame.exchange(0u, std::memory_order_relaxed));
            rerunVisualizer->drawScalar(
                "liorf/cbs/isam2/active_cbs_prior_slots",
                cbsBpsamActivePriorSlotsPerRerunFrame.exchange(0u, std::memory_order_relaxed));
        }

        const auto trajectory = toRerunPoints(*cloudKeyPoses3D, 5000u);
        if (!trajectory.empty()) {
            rerunVisualizer->drawPoints(
                "liorf/trajectory",
                trajectory,
                Eigen::Vector4f(255.f, 160.f, 0.f, 255.f),
                3.f);
        }

        if (rerunFactorGraphEnable && bpsam && isamCurrentEstimate.size() > 0u) {
            const auto& factorGraph = bpsam->getFactorsUnsafe();
            if (factorGraph.size() > 0u) {
                const Eigen::Vector4f localFactorColor(255.f, 160.f, 0.f, 180.f);
                const Eigen::Vector4f persistentCbsFactorColor(255.f, 60.f, 220.f, 230.f);
                std::vector<Eigen::Vector4f> factorColors(factorGraph.size(), localFactorColor);
                size_t persistentCbsFactorCount = 0u;
                for (size_t i = 0; i < factorGraph.size(); ++i) {
                    const auto factor = factorGraph.at(i);
                    if (!factor) {
                        continue;
                    }
                    if (cbs::isAnchorBeliefFactor(factor)) {
                        factorColors[i] = persistentCbsFactorColor;
                        ++persistentCbsFactorCount;
                    }
                }
                rerunVisualizer->drawFactors(
                    "liorf/factor_graph",
                    factorGraph,
                    isamCurrentEstimate,
                    factorColors,
                    0.75f,
                    false,
                    true);
                rerunVisualizer->drawScalar("liorf/factor_graph/factors_total",
                                            factorGraph.size());
                rerunVisualizer->drawScalar("liorf/factor_graph/persistent_cbs_factors",
                                            persistentCbsFactorCount);
            }

            std::vector<RerunExternalOdomEdge> externalOdomEdges;
            {
                std::lock_guard<std::mutex> lock(mtxRerunCbsVisualization);
                externalOdomEdges = rerunExternalOdomEdges;
            }

            std::vector<std::pair<Point3, Point3>> externalOdomLines;
            externalOdomLines.reserve(externalOdomEdges.size());
            std::vector<Eigen::Vector4f> externalOdomColors;
            externalOdomColors.reserve(externalOdomEdges.size());
            for (const auto& edge : externalOdomEdges) {
                if (!isamCurrentEstimate.exists(edge.fromPoseKey) ||
                    !isamCurrentEstimate.exists(edge.toPoseKey)) {
                    continue;
                }
                const Pose3 fromPose = isamCurrentEstimate.at<Pose3>(edge.fromPoseKey);
                const Pose3 toPose = isamCurrentEstimate.at<Pose3>(edge.toPoseKey);
                externalOdomLines.emplace_back(fromPose.translation(), toPose.translation());
                externalOdomColors.emplace_back(Eigen::Vector4f(0.f, 255.f, 255.f, 255.f));
            }
            if (!externalOdomLines.empty()) {
                rerunVisualizer->drawLines(
                    "liorf/cbs/external_odom_factors",
                    externalOdomLines,
                    externalOdomColors,
                    2.5f);
            }
            rerunVisualizer->drawScalar("liorf/cbs/external_odom_factors/visible",
                                        externalOdomLines.size());
        }

        if (laserCloudSurfFromMapDS && !laserCloudSurfFromMapDS->empty()) {
            pcl::PointCloud<PointType>::Ptr rerunLocalMap(new pcl::PointCloud<PointType>());
            if (rerunLocalMapLeafSize > 0.f) {
                pcl::VoxelGrid<PointType> rerunLocalMapFilter;
                rerunLocalMapFilter.setLeafSize(rerunLocalMapLeafSize,
                                                rerunLocalMapLeafSize,
                                                rerunLocalMapLeafSize);
                rerunLocalMapFilter.setInputCloud(laserCloudSurfFromMapDS);
                rerunLocalMapFilter.filter(*rerunLocalMap);
            } else {
                *rerunLocalMap = *laserCloudSurfFromMapDS;
            }

            const auto local_map = toRerunPoints(*rerunLocalMap, rerunLocalMapMaxPoints);
            if (!local_map.empty()) {
                rerunVisualizer->drawPoints(
                    "liorf/local_map",
                    local_map,
                    Eigen::Vector4f(80.f, 180.f, 255.f, 90.f),
                    1.f);
            }
        }

        if (laserCloudSurfLastDS && !laserCloudSurfLastDS->empty()) {
            const auto registered_cloud =
                transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
            const auto current_scan = toRerunPoints(*registered_cloud, 20000u);
            rerunVisualizer->drawPoints(
                "liorf/current_scan",
                current_scan,
                Eigen::Vector4f(255.f, 255.f, 255.f, 180.f),
                1.5f);
        }
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

    void addOdomFactor()
    {
        const size_t localPoseIndex = cloudKeyPoses3D->size();
        const Key currentKey = ensurePoseKeyForLocalIndex(localPoseIndex);
        const Pose3 poseTo = trans2gtsamPose(transformTobeMapped);

        if (localPoseIndex == 0)
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e-10, 1e-10, 1e-10).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(currentKey, poseTo, priorNoise));
            initialEstimate.insert(currentKey, poseTo);
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            const Key previousKey = ensurePoseKeyForLocalIndex(localPoseIndex - 1);
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtSAMgraph.add(BetweenFactor<Pose3>(previousKey, currentKey, poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(currentKey, poseTo);
        }
    }

    void addGPSFactor()
    {
        if (!useGPS)
            return;

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
        if (poseCovariance.rows() < 5 || poseCovariance.cols() < 5)
            return;

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
                const Key currentKey = ensurePoseKeyForLocalIndex(cloudKeyPoses3D->size());
                gtsam::GPSFactor gps_factor(currentKey, gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
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
            if (indexFrom < 0 || indexTo < 0) {
                continue;
            }

            Key fromKey = 0;
            Key toKey = 0;
            if (!getPoseKeyForLocalIndex(static_cast<size_t>(indexFrom), &fromKey) ||
                !getPoseKeyForLocalIndex(static_cast<size_t>(indexTo), &toKey)) {
                ROS_WARN_STREAM_THROTTLE(1.0, "Skipping loop factor with unmapped indices " << indexFrom << " -> " << indexTo);
                continue;
            }
            gtSAMgraph.add(BetweenFactor<Pose3>(fromKey, toKey, poseBetween, noiseBetween));
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

        const size_t localPoseIndex = cloudKeyPoses3D->size();
        const Key currentPoseKey = ensurePoseKeyForLocalIndex(localPoseIndex);

        // odom factor
        addOdomFactor();

        // gps factor
        addGPSFactor();

        // loop factor
        addLoopFactor();

        // external odometry belief factors (from camera-VIO backend) are pre-matched by timestamp.
        consumeIncomingOdomBeliefsIntoBpsam();

        const KeySet activeCbsFactorKeys = activeBeliefWindowKeys();
        const FactorIndices staleCbsFactorSlots =
            bpsam->cbsFactorSlotsOutsideKeys(activeCbsFactorKeys);
        if (!staleCbsFactorSlots.empty()) {
            ROS_INFO_STREAM_THROTTLE(
                1.0,
                "LiORF CBS pruning " << staleCbsFactorSlots.size()
                                     << " stale external odometry factors outside the "
                                     << beliefExchangeWindowSize
                                     << "-pose active window.");
        }

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update BPSAM backend
        const auto bpsamOptimizationStart = std::chrono::steady_clock::now();
        cbs::BPSAM::UpdateParams bpsamUpdateParams;
        bpsamUpdateParams.removeFactorIndices = staleCbsFactorSlots;
        bpsam->update(gtSAMgraph, initialEstimate, bpsamUpdateParams);
        bpsam->update();

        if (aLoopIsClosed == true)
        {
            bpsam->update();
            bpsam->update();
            bpsam->update();
            bpsam->update();
            bpsam->update();
        }
        atomicAddRelaxed(&bpsamOptimizationTimeMsPerRerunFrame,
                         elapsedMs(bpsamOptimizationStart));
        cbsBpsamRootSizePerRerunFrame.store(
            bpsam->rootFrontalKeyCount(), std::memory_order_relaxed);
        cbsBpsamActivePriorSlotsPerRerunFrame.store(
            bpsam->trackedCbsFactorSlotCount(), std::memory_order_relaxed);
        ROS_INFO_STREAM_THROTTLE(
            1.0,
            "LiORF BPSAM root: size=" << bpsam->rootFrontalKeyCount()
                                      << " cliques=" << bpsam->rootCliqueCount()
                                      << " active_cbs_priors="
                                      << bpsam->trackedCbsFactorSlotCount()
                                      << " max_root=" << cbsBeliefMaxRootSize);

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = bpsam->calculateEstimate();
        if (!isamCurrentEstimate.exists(currentPoseKey)) {
            ROS_ERROR_STREAM("Current pose key is missing from estimate: " << currentPoseKey);
            return;
        }
        latestEstimate = isamCurrentEstimate.at<Pose3>(currentPoseKey);
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
        setPoseTimestamp(localPoseIndex, timeLaserInfoCur);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << bpsam->marginalCovariance(currentPoseKey) << endl << endl;
        try {
            poseCovariance = bpsam->marginalCovariance(currentPoseKey);
        } catch (const std::exception& e) {
            ROS_WARN_STREAM("Failed to fetch pose covariance for key " << currentPoseKey << ": " << e.what());
        }

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

        // This snapshot is consumed by the future Kimera bridge for phase-1 belief exchange.
        refreshOutgoingBeliefs();
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
            int numPoses = cloudKeyPoses3D->size();
            for (int i = 0; i < numPoses; ++i)
            {
                Key key = 0;
                if (!getPoseKeyForLocalIndex(static_cast<size_t>(i), &key) || !isamCurrentEstimate.exists(key)) {
                    continue;
                }
                const Pose3 correctedPose = isamCurrentEstimate.at<Pose3>(key);

                cloudKeyPoses3D->points[i].x = correctedPose.translation().x();
                cloudKeyPoses3D->points[i].y = correctedPose.translation().y();
                cloudKeyPoses3D->points[i].z = correctedPose.translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = correctedPose.rotation().roll();
                cloudKeyPoses6D->points[i].pitch = correctedPose.rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = correctedPose.rotation().yaw();

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
        publishRerunFrame();
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
