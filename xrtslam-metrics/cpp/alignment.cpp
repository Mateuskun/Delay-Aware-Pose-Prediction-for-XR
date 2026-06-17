#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <sophus/se3.hpp>

namespace py = pybind11;

using Scalar = float;
using arg = py::arg;

using Timestamps = Eigen::Matrix<int64_t, Eigen::Dynamic, 1>;
using TimestampsMutRef = Eigen::Ref<Timestamps>;
using TimestampsConstRef = const Eigen::Ref<const Timestamps>;

using Positions = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;
using PositionsMutRef = Eigen::Ref<Positions>;
using PositionsConstRef = const Eigen::Ref<const Positions>;

using Quaternions = Eigen::Matrix<Scalar, 4, Eigen::Dynamic>;  // xyzw order
using QuaternionsMutRef = Eigen::Ref<Quaternions>;
using QuaternionsConstRef = Eigen::Ref<const Quaternions>;

using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
using SE3 = Sophus::SE3<Scalar>;
using Quaternion = Eigen::Quaternion<Scalar>;

template <bool use_quats>
int associate_(TimestampsMutRef inout_est_ts, TimestampsMutRef inout_ref_ts,  //
               PositionsMutRef inout_est_xyz, PositionsMutRef inout_ref_xyz,  //
               QuaternionsMutRef inout_est_quat, QuaternionsMutRef inout_ref_quat) {
  int num_est = inout_est_ts.size();
  int num_ref = inout_ref_ts.size();
  assert(num_est < num_ref);
  assert(num_est == inout_est_xyz.cols() && num_ref == inout_ref_xyz.cols());
  if constexpr (use_quats) assert(num_est == inout_est_quat.cols() && num_ref == inout_ref_quat.cols());

  Positions est_associations(3, num_est);
  Positions ref_associations(3, num_est);
  Quaternions est_quat_associations{};
  Quaternions ref_quat_associations{};

  if constexpr (use_quats) {
    est_quat_associations.resize(4, num_est);
    ref_quat_associations.resize(4, num_est);
  }

  Timestamps ts_associations(num_est);

  int num_assocs = 0;

  Eigen::Index i = 0;  // est index
  Eigen::Index j = 0;  // ref index

  // Advance est to first ref or after
  while (i < num_est && inout_est_ts(i) < inout_ref_ts(0)) i++;

  for (; i < num_est; i++) {
    int64_t t_ns = inout_est_ts(i);

    // j is -1
    for (; j < inout_ref_ts.size(); j++) {
      if (inout_ref_ts(j) > t_ns) break;
    }
    j--;  // j will never be -1 because i starts such that est(i) > ref(0)

    if (j >= inout_ref_ts.size() - 1) continue;

    double dt_ns = t_ns - inout_ref_ts(j);
    double int_t_ns = inout_ref_ts(j + 1) - inout_ref_ts(j);
    assert(dt_ns >= 0 && int_t_ns > 0);

    if (int_t_ns > 1.1e8) continue;  // Skip if >100ms

    double ratio = dt_ns / int_t_ns;
    assert(ratio >= 0 && ratio < 1);

    Vector3 gt = (1 - ratio) * inout_ref_xyz.col(j) + ratio * inout_ref_xyz.col(j + 1);
    ref_associations.col(num_assocs) = gt;
    est_associations.col(num_assocs) = inout_est_xyz.col(i);
    ts_associations(num_assocs) = t_ns;

    if constexpr (use_quats) {
      Quaternion gt_quat = Quaternion{inout_ref_quat.col(j)}.slerp(ratio, Quaternion{inout_ref_quat.col(j + 1)});
      ref_quat_associations.col(num_assocs) = gt_quat.coeffs();
      est_quat_associations.col(num_assocs) = inout_est_quat.col(i);
    }

    num_assocs++;
  }

  for (Eigen::Index i = 0; i < num_assocs; i++) {
    inout_est_xyz.col(i) = est_associations.col(i);
    inout_ref_xyz.col(i) = ref_associations.col(i);
    inout_est_ts(i) = ts_associations(i);
    inout_ref_ts(i) = ts_associations(i);

    if constexpr (use_quats) {
      inout_est_quat.col(i) = est_quat_associations.col(i);
      inout_ref_quat.col(i) = ref_quat_associations.col(i);
    }
  }

  return num_assocs;
}

int associate(TimestampsMutRef est_ts, TimestampsMutRef ref_ts,  //
              PositionsMutRef est_xyz, PositionsMutRef ref_xyz) {
  Quaternions _{};  // Unused
  return associate_<false>(est_ts, ref_ts, est_xyz, ref_xyz, _, _);
}

int associate_full(TimestampsMutRef est_ts, TimestampsMutRef ref_ts,  //
                   PositionsMutRef est_xyz, PositionsMutRef ref_xyz,  //
                   QuaternionsMutRef est_quat, QuaternionsMutRef ref_quat) {
  return associate_<true>(est_ts, ref_ts, est_xyz, ref_xyz, est_quat, ref_quat);
}

Matrix4 align_ref(PositionsConstRef est_xyz, PositionsConstRef ref_xyz, int i, int j) {
  assert(est_xyz.cols() == ref_xyz.cols());
  int pose_count = est_xyz.cols();

  assert(i < j && i >= 0 && i < pose_count && j >= 0 && j <= pose_count);

  // Get block i-j without copy
  auto estb_xyz = est_xyz.block(0, i, 3, j - i);
  auto refb_xyz = ref_xyz.block(0, i, 3, j - i);

  Vector3 mean_est = estb_xyz.rowwise().mean();
  Vector3 mean_ref = refb_xyz.rowwise().mean();

  Positions c_est = estb_xyz.colwise() - mean_est;
  Positions c_ref = refb_xyz.colwise() - mean_ref;
  Matrix3 cov = c_ref * c_est.transpose();
  Eigen::JacobiSVD<Matrix3> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Matrix3 S;
  S.setIdentity();

  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) S(2, 2) = -1;

  Matrix3 rot_gt_est = svd.matrixU() * S * svd.matrixV().transpose();
  Vector3 trans = mean_ref - rot_gt_est * mean_est;

  SE3 T_ref_est(rot_gt_est, trans);

  // Update the reference trajectory with alignment
  // TODO@mateosss: update ref traj?
  // SE3 T_est_ref = T_ref_est.inverse();
  // for (Eigen::Index i = 0; i < pose_count; i++) {
  //   ref_xyz.col(i) = T_est_ref * ref_xyz.col(i);
  // }

  return T_ref_est.matrix();
}

Scalar compute_ate(PositionsConstRef est_xyz, PositionsConstRef ref_xyz, int i, int j,
                   const Eigen::Ref<Matrix4>& T_ref_est_mat) {
  assert(est_xyz.cols() == ref_xyz.cols());
  int pose_count = j - i;

  assert(i < j && i >= 0 && i < pose_count && j >= 0 && j <= pose_count);

  SE3 T_ref_est(T_ref_est_mat);

  Scalar rmse = 0;
  // Scalar mean = 0;
  // Scalar stdev = 0;
  // Scalar min = std::numeric_limits<Scalar>::max();
  // Scalar max = std::numeric_limits<Scalar>::min();

  for (Eigen::Index k = i; k < j; k++) {
    Vector3 res = T_ref_est * est_xyz.col(k) - ref_xyz.col(k);
    rmse += res.transpose() * res;
    // mean += res.norm();
    // min = std::min(min, res.norm());
    // max = std::max(max, res.norm());
  }

  rmse = std::sqrt(rmse / pose_count);
  // mean /= pose_count;

  // for (Eigen::Index k = i; k < j; k++) {
  //   Scalar diff = res.norm() - mean;
  //   stdev += diff * diff;
  // }
  // stdev = std::sqrt(stdev / pose_count);

  // std::cout << "T_align\n" << T_ref_est.matrix() << std::endl;
  // std::cout << "error " << rmse << std::endl;
  // std::cout << "number of associations " << j << std::endl;

  return rmse;
}
Scalar compute_ate_and_align_ref(TimestampsConstRef est_ts, TimestampsConstRef ref_ts,  //
                                 PositionsConstRef est_xyz, PositionsMutRef ref_xyz) {
  assert(est_ts.size() == est_xyz.cols() && ref_ts.size() == ref_xyz.cols());

  int num_est = est_xyz.cols();
  assert(num_est < ref_xyz.cols());

  Positions est_associations(3, num_est);
  Positions ref_associations(3, num_est);
  Timestamps ts_associations(num_est);

  int num_assocs = 0;
  for (Eigen::Index i = 0; i < num_est; i++) {
    int64_t t_ns = est_ts(i);

    Eigen::Index j = 0;
    for (; j < ref_ts.size(); j++)
      if (ref_ts(j) > t_ns) break;
    j--;

    if (j <= 0 || j >= ref_ts.size() - 1) continue;

    double dt_ns = t_ns - ref_ts(j);
    double int_t_ns = ref_ts(j + 1) - ref_ts(j);
    assert(dt_ns >= 0 && int_t_ns > 0);

    if (int_t_ns > 1.1e8) continue;  // Skip if >100ms

    double ratio = dt_ns / int_t_ns;
    assert(ratio >= 0 && ratio < 1);

    Vector3 gt = (1 - ratio) * ref_xyz.col(j) + ratio * ref_xyz.col(j + 1);
    ref_associations.col(num_assocs) = gt;
    est_associations.col(num_assocs) = est_xyz.col(i);
    ts_associations(num_assocs) = t_ns;
    num_assocs++;
  }

  est_associations.conservativeResize(Eigen::NoChange, num_assocs);
  ref_associations.conservativeResize(Eigen::NoChange, num_assocs);
  ts_associations.conservativeResize(num_assocs);

  Positions gt, est;
  gt.setZero(3, num_assocs);
  est.setZero(3, num_assocs);

  // TODO@mateosss: this is just a copy
  for (Eigen::Index i = 0; i < num_assocs; i++) {
    gt.col(i) = ref_associations.col(i);
    est.col(i) = est_associations.col(i);
  }

  Vector3 mean_gt = gt.rowwise().mean();
  Vector3 mean_est = est.rowwise().mean();

  gt.colwise() -= mean_gt;
  est.colwise() -= mean_est;

  Matrix3 cov = gt * est.transpose();

  Eigen::JacobiSVD<Matrix3> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Matrix3 S;
  S.setIdentity();

  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) S(2, 2) = -1;

  Matrix3 rot_gt_est = svd.matrixU() * S * svd.matrixV().transpose();
  Vector3 trans = mean_gt - rot_gt_est * mean_est;

  SE3 T_gt_est(rot_gt_est, trans);

  // Update the reference trajectory with alignment
  SE3 T_est_gt = T_gt_est.inverse();
  for (Eigen::Index i = 0; i < ref_xyz.cols(); i++) {
    ref_xyz.col(i) = T_est_gt * ref_xyz.col(i);
  }

  Scalar error = 0;
  for (Eigen::Index i = 0; i < num_assocs; i++) {
    est_associations.col(i) = T_gt_est * est_associations.col(i);
    Vector3 res = est_associations.col(i) - ref_associations.col(i);
    error += res.transpose() * res;
  }

  error /= num_assocs;
  error = std::sqrt(error);

  std::cout << "T_align\n" << T_gt_est.matrix() << std::endl;
  std::cout << "error " << error << std::endl;
  std::cout << "number of associations " << num_assocs << std::endl;

  return error;
}

float align(float x, float y) { return x + y; }

PYBIND11_MODULE(alignment, m) {
  m.doc() = "Positions alignment and ATE/RTE metric computation";
  m.def("associate", &associate, "Make trajectory translations comparable (associations)",  //
        arg("est_ts"), arg("ref_ts"), arg("est_xyz"), arg("ref_xyz"));
  m.def("associate_full", &associate_full, "Make trajectory translations and rotations comparable (associations)",  //
        arg("est_ts"), arg("ref_ts"), arg("est_xyz"), arg("ref_xyz"), arg("est_quat"), arg("ref_quat"));
  m.def("align_ref", &align_ref, "Align reference trajectory, returns T_ref_est SE3 transform",  //
        arg("est_xyz"), arg("ref_xyz"), arg("i"), arg("j"));
  m.def("compute_ate", &compute_ate, "Compute ATE in associated & aligned trajectories",  //
        arg("est_xyz"), arg("ref_xyz"), arg("i"), arg("j"), arg("T_ref_est"));
  m.def("compute_ate_and_align_ref", &compute_ate_and_align_ref, "Compute ATE",  //
        arg("est_ts"), arg("ref_ts"), arg("est_xyz"), arg("ref_xyz"));
}
