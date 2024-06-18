// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/controllers/bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/similarity_transform.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <ceres/ceres.h>

namespace colmap {
namespace {

// Callback functor called after each bundle adjustment iteration.
class BundleAdjustmentIterationCallback : public ceres::IterationCallback {
 public:
  explicit BundleAdjustmentIterationCallback(BaseController* controller)
      : controller_(controller) {}

  virtual ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) {
    THROW_CHECK_NOTNULL(controller_);
    if (controller_->CheckIfStopped()) {
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    } else {
      return ceres::SOLVER_CONTINUE;
    }
  }

 private:
  BaseController* controller_;
};

}  // namespace

BundleAdjustmentController::BundleAdjustmentController(
    const OptionManager& options,
    std::shared_ptr<Reconstruction> reconstruction,
    std::shared_ptr<Reconstruction> real_pose,
    bool do_adj)
    : options_(options), reconstruction_(std::move(reconstruction)), real_pose_(real_pose), do_adj_(do_adj) {}

void BundleAdjustmentController::Run() {
  THROW_CHECK_NOTNULL(reconstruction_);

  PrintHeading1("Global bundle adjustment");
  Timer run_timer;
  run_timer.Start();

  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  reconstruction_->Normalize();

  if (reg_image_ids.size() < 2) {
    LOG(ERROR) << "Need at least two views.";
    return;
  }

  // Avoid degeneracies in bundle adjustment.
  ObservationManager(*reconstruction_).FilterObservationsWithNegativeDepth();

  BundleAdjustmentOptions ba_options = *options_.bundle_adjustment;

  BundleAdjustmentIterationCallback iteration_callback(this);
  ba_options.solver_options.callbacks.push_back(&iteration_callback);

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }
  ba_config.SetConstantCamPose(reg_image_ids[0]);
  ba_config.SetConstantCamPositions(reg_image_ids[1], {0});

  // Run bundle adjustment.
  if (do_adj_)
  {
    BundleAdjuster bundle_adjuster(ba_options, ba_config);
    bundle_adjuster.Solve(reconstruction_.get());
  }

  // Transform to database pose coordinate
  if (real_pose_)
  {
    std::vector<Eigen::Vector3d> real_positions;
    std::vector<Eigen::Vector3d> sim_positions;
    for (auto const& real_pose : real_pose_->Images())
    {
      auto* sim_pose = reconstruction_->FindImageWithName(real_pose.second.Name());
      if (sim_pose)
      {
        real_positions.push_back(real_pose.second.ProjectionCenter());
        sim_positions.push_back(sim_pose->ProjectionCenter());
      }
    }
    Sim3d sim2real;
    if (real_positions.size() &&
        EstimateSim3d(sim_positions,
                      real_positions,
                      sim2real))
    {
      reconstruction_->Transform(sim2real);
      std::vector<double> res;
      SimilarityTransformEstimator<3>::Residuals(sim_positions, real_positions, sim2real.ToMatrix(), &res);
      double res_mean(0);
      for (auto&r : res)
        res_mean += r;
      res_mean = sqrt(res_mean / res.size());
      LOG(INFO) << "Sim2real residual: " << res_mean;
    }
  }


  run_timer.PrintMinutes();
}

}  // namespace colmap
