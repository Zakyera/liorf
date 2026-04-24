#include <cbs/bpsam/bpsam.h>
#include <cbs/key.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gtsam;

namespace {

cbs::BPSAM::Ptr makeSam(cbs::AgentId id) {
  cbs::BPSAM::Params params;
  params.robot_id = id;
  params.sam_params_.relinearizeThreshold = 0.1;
  params.sam_params_.relinearizeSkip = 1;
  ISAM2GaussNewtonParams gn_params;
  params.sam_params_.optimizationParams = gn_params;
  params.enable_gkcm = false;
  params.enable_belief_dcs = false;
  params.belief_similarity_threshold = 0.0;
  return std::make_shared<cbs::BPSAM>(params);
}

void addPoseChain(cbs::BPSAM& sam, cbs::AgentId robot_id, size_t n_poses) {
  if (n_poses < 2) {
    throw std::runtime_error("Need at least 2 poses for chain test.");
  }

  NonlinearFactorGraph graph;
  Values values;

  const auto prior_noise = noiseModel::Diagonal::Variances(
      (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
  const auto odom_noise = noiseModel::Diagonal::Variances(
      (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

  Key key0 = cbs::toPoseKey(robot_id, 0);
  graph.add(PriorFactor<Pose3>(key0, Pose3(), prior_noise));
  values.insert(key0, Pose3());

  for (size_t i = 1; i < n_poses; ++i) {
    Key key_prev = cbs::toPoseKey(robot_id, i - 1);
    Key key_curr = cbs::toPoseKey(robot_id, i);
    Pose3 delta(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(1.0, 0.0, 0.0));
    graph.add(BetweenFactor<Pose3>(key_prev, key_curr, delta, odom_noise));
    values.insert(key_curr, Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(i, 0.0, 0.0)));
  }

  sam.update(graph, values);
  sam.update();
}

std::map<Key, std::vector<std::pair<cbs::AgentId, gbp::Gaussian>>> buildBeliefs(
    cbs::BPSAM& source,
    cbs::AgentId source_id,
    cbs::AgentId target_id,
    size_t n_poses) {
  std::map<Key, std::vector<std::pair<cbs::AgentId, gbp::Gaussian>>> beliefs;
  for (size_t i = 0; i < n_poses; ++i) {
    Key source_key = cbs::toPoseKey(source_id, i);
    Key target_key = cbs::toPoseKey(target_id, i);
    Pose3 pose = source.calculateEstimate<Pose3>(source_key);
    Vector mu = traits<Pose3>::Logmap(pose);
    Matrix sigma = source.marginalCovariance(source_key);
    gbp::Gaussian g(target_key, mu, sigma, 1);
    g.relax_factor() = 0.0;
    beliefs[target_key].emplace_back(source_id, g);
  }
  return beliefs;
}

}  // namespace

int main() {
  try {
    constexpr cbs::AgentId kLiorf = static_cast<cbs::AgentId>('l');
    constexpr cbs::AgentId kKimera = static_cast<cbs::AgentId>('k');
    constexpr size_t kPoses = 4;

    auto liorf = makeSam(kLiorf);
    auto kimera = makeSam(kKimera);

    addPoseChain(*liorf, kLiorf, kPoses);
    addPoseChain(*kimera, kKimera, kPoses);

    auto beliefs = buildBeliefs(*kimera, kKimera, kLiorf, kPoses);
    int rejected_first = liorf->addBeliefs(beliefs);
    int rejected_second = liorf->addBeliefs(beliefs);
    liorf->update();

    liorf->setMarginalizationGraph(cbs::BPSAM::MarginalizationType::LOCAL);
    KeySet request_keys;
    request_keys.insert(cbs::toPoseKey(kLiorf, kPoses - 1));
    auto out_beliefs = liorf->getBeliefs(request_keys, true);

    if (!liorf->valueExists(cbs::toPoseKey(kLiorf, kPoses - 1))) {
      throw std::runtime_error("Latest LiORF key missing after update.");
    }
    if (out_beliefs.empty()) {
      throw std::runtime_error("No outgoing beliefs produced for latest key.");
    }

    std::cout << "BPSAM smoke test passed."
              << " rejected_first=" << rejected_first
              << " rejected_second=" << rejected_second
              << " outgoing_keys=" << out_beliefs.size() << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "BPSAM smoke test failed: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "BPSAM smoke test failed: unknown exception." << std::endl;
    return 1;
  }
}
