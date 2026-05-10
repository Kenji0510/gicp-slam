use nalgebra::{Isometry3, Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use rayon::prelude::*;

use crate::compute_covariance::{add_diagonal, invert_matrix3_safe};
use crate::find_nearest_points::GaussianCorrespondence;

pub type Matrix6f = SMatrix<f32, 6, 6>;
pub type Vector6f = SVector<f32, 6>;

type ShapeResidual6f = SVector<f32, 6>;
type ShapeJacobian6x3f = SMatrix<f32, 6, 3>;

/// Gaussian mean-to-mean + covariance shape-to-shape 用の線形システム。
#[derive(Debug, Clone)]
pub struct GaussianLinearSystem {
    pub h: Matrix6f,
    pub b: Vector6f,

    /// mean項 + shape項の合計cost。
    pub cost: f32,

    /// mean位置合わせ項のcost。
    pub mean_cost: f32,

    /// covariance形状一致項のcost。
    pub shape_cost: f32,

    /// 実際に使われた対応Gaussian数。
    pub used_count: usize,
}

impl Default for GaussianLinearSystem {
    fn default() -> Self {
        Self {
            h: Matrix6f::zeros(),
            b: Vector6f::zeros(),
            cost: 0.0,
            mean_cost: 0.0,
            shape_cost: 0.0,
            used_count: 0,
        }
    }
}

/// Gaussian shape matching の調整パラメータ。
///
/// 初期値としては、まず shape_weight を小さめにして、
/// mean項で大きく合わせた後に covariance の向きを補助的に使うのが安全。
#[derive(Debug, Clone, Copy)]
pub struct GaussianShapeOptions {
    /// mean位置合わせ項の重み。
    pub mean_weight: f32,

    /// covariance形状一致項の重み。
    /// 大きすぎると廊下・壁面で回転がshape側に引っ張られすぎる。
    pub shape_weight: f32,

    /// (Σ_t + RΣ_sR^T)^-1 を作るときの対角正則化。
    pub covariance_regularization: f32,

    /// 対応点gate用。find側で絞っていても、ここで二重に安全確認する。
    pub max_euclidean_dist_sq: f32,

    /// Mahalanobis gate。不要なら None。
    pub max_mahalanobis_dist: Option<f32>,

    /// covariance shapeの有限差分eps。1e-3 rad程度から開始。
    pub shape_fd_epsilon: f32,

    /// trueにすると covariance を trace で割り、サイズより「形・向き」を主に合わせる。
    /// falseにすると広がりの大きさも含めて合わせる。
    pub normalize_shape_by_trace: bool,

    /// trace正規化時の下限。
    pub min_trace: f32,
}

impl Default for GaussianShapeOptions {
    fn default() -> Self {
        Self {
            mean_weight: 1.0,
            shape_weight: 0.05,
            covariance_regularization: 1.0e-3,
            max_euclidean_dist_sq: 1.0,
            max_mahalanobis_dist: None,
            shape_fd_epsilon: 1.0e-3,
            normalize_shape_by_trace: true,
            min_trace: 1.0e-6,
        }
    }
}

#[inline]
fn is_finite_vector6(v: &SVector<f32, 6>) -> bool {
    v.iter().all(|x| x.is_finite())
}

#[inline]
fn is_finite_matrix6(m: &Matrix6f) -> bool {
    m.iter().all(|x| x.is_finite())
}

#[inline]
fn normalized_covariance_shape(
    cov: Matrix3<f32>,
    normalize_by_trace: bool,
    min_trace: f32,
) -> Matrix3<f32> {
    if !normalize_by_trace {
        return cov;
    }

    let trace = cov.trace();
    if !trace.is_finite() || trace.abs() < min_trace {
        return cov / min_trace;
    }

    cov / trace
}

/// 対称3x3行列を6次元ベクトルへ変換する。
/// off-diagonalに sqrt(2) を掛けることで、ベクトルnormがFrobenius normと一致する。
#[inline]
fn symmetric_matrix_to_frobenius_vector(m: Matrix3<f32>) -> ShapeResidual6f {
    let sqrt2 = 2.0_f32.sqrt();
    ShapeResidual6f::from_row_slice(&[
        m[(0, 0)],
        m[(1, 1)],
        m[(2, 2)],
        sqrt2 * m[(0, 1)],
        sqrt2 * m[(0, 2)],
        sqrt2 * m[(1, 2)],
    ])
}

/// covariance shape residual。
/// r = vec( normalize(Σ_t) - normalize(RΣ_sR^T) )
fn covariance_shape_residual(
    target_covariance: Matrix3<f32>,
    rotated_source_covariance: Matrix3<f32>,
    normalize_by_trace: bool,
    min_trace: f32,
) -> ShapeResidual6f {
    let target_shape =
        normalized_covariance_shape(target_covariance, normalize_by_trace, min_trace);
    let source_shape =
        normalized_covariance_shape(rotated_source_covariance, normalize_by_trace, min_trace);

    symmetric_matrix_to_frobenius_vector(target_shape - source_shape)
}

#[inline]
fn delta_rotation_matrix(axis_index: usize, angle: f32) -> Matrix3<f32> {
    let mut rot_vec = Vector3::<f32>::zeros();
    rot_vec[axis_index] = angle;
    UnitQuaternion::from_scaled_axis(rot_vec)
        .to_rotation_matrix()
        .matrix()
        .clone_owned()
}

/// shape residualの回転ヤコビアンを数値微分で求める。
///
/// 形状項は translation には依存しないため、Jacobianは 6x3 の回転成分のみ。
/// 左更新 T <- δT * T に合わせ、現在の rotated_source_covariance に
/// δR を左から掛ける。
fn numerical_shape_jacobian_rotation(
    target_covariance: Matrix3<f32>,
    rotated_source_covariance: Matrix3<f32>,
    normalize_by_trace: bool,
    min_trace: f32,
    eps: f32,
) -> ShapeJacobian6x3f {
    let mut j = ShapeJacobian6x3f::zeros();
    let eps = eps.max(1.0e-6);

    for axis in 0..3 {
        let r_plus = delta_rotation_matrix(axis, eps);
        let r_minus = delta_rotation_matrix(axis, -eps);

        let cov_plus = r_plus * rotated_source_covariance * r_plus.transpose();
        let cov_minus = r_minus * rotated_source_covariance * r_minus.transpose();

        let residual_plus = covariance_shape_residual(
            target_covariance,
            cov_plus,
            normalize_by_trace,
            min_trace,
        );
        let residual_minus = covariance_shape_residual(
            target_covariance,
            cov_minus,
            normalize_by_trace,
            min_trace,
        );

        let col = (residual_plus - residual_minus) / (2.0 * eps);
        j.set_column(axis, &col);
    }

    j
}

fn add_mean_alignment_term(
    local: &mut GaussianLinearSystem,
    corr: &GaussianCorrespondence,
    rotated_source_covariance: Matrix3<f32>,
    options: &GaussianShapeOptions,
) -> Option<()> {
    if options.mean_weight <= 0.0 {
        return Some(());
    }

    if corr.euclidean_dist_sq > options.max_euclidean_dist_sq {
        return None;
    }

    if let Some(max_score) = options.max_mahalanobis_dist {
        if corr.mahalanobis_dist > max_score {
            return None;
        }
    }

    let ps = corr.transformed_source_mean;
    let pt = corr.target_mean;

    let c_sum = add_diagonal(
        corr.target_covariance + rotated_source_covariance,
        options.covariance_regularization,
    );
    let omega = invert_matrix3_safe(c_sum)?;

    // error = target - transformed_source
    let err: Vector3<f32> = pt.coords - ps.coords;
    let we = omega * err;

    let x = ps.x;
    let y = ps.y;
    let z = ps.z;
    let p = ps.coords;

    let w = options.mean_weight;

    // b_rot = p x We
    let b_rot = p.cross(&we);
    local.b[0] += w * b_rot.x;
    local.b[1] += w * b_rot.y;
    local.b[2] += w * b_rot.z;

    // b_trans = We
    local.b[3] += w * we.x;
    local.b[4] += w * we.y;
    local.b[5] += w * we.z;

    // small-angle rotation Jacobian columns for transformed point p.
    // δp = axis × p
    let s0 = Vector3::new(0.0, -z, y);
    let s1 = Vector3::new(z, 0.0, -x);
    let s2 = Vector3::new(-y, x, 0.0);

    let ws0 = omega * s0;
    let ws1 = omega * s1;
    let ws2 = omega * s2;

    // H_rr columns = p x ws
    let hcol0 = p.cross(&ws0);
    let hcol1 = p.cross(&ws1);
    let hcol2 = p.cross(&ws2);

    local.h[(0, 0)] += w * hcol0.x;
    local.h[(1, 0)] += w * hcol0.y;
    local.h[(2, 0)] += w * hcol0.z;

    local.h[(0, 1)] += w * hcol1.x;
    local.h[(1, 1)] += w * hcol1.y;
    local.h[(2, 1)] += w * hcol1.z;

    local.h[(0, 2)] += w * hcol2.x;
    local.h[(1, 2)] += w * hcol2.y;
    local.h[(2, 2)] += w * hcol2.z;

    // H_tt = Omega
    for r in 0..3 {
        for c in 0..3 {
            local.h[(r + 3, c + 3)] += w * omega[(r, c)];
        }
    }

    // H_rt
    local.h[(0, 3)] += w * ws0.x;
    local.h[(0, 4)] += w * ws0.y;
    local.h[(0, 5)] += w * ws0.z;

    local.h[(1, 3)] += w * ws1.x;
    local.h[(1, 4)] += w * ws1.y;
    local.h[(1, 5)] += w * ws1.z;

    local.h[(2, 3)] += w * ws2.x;
    local.h[(2, 4)] += w * ws2.y;
    local.h[(2, 5)] += w * ws2.z;

    // H_tr = H_rt^T
    for r in 0..3 {
        for c in 0..3 {
            local.h[(r + 3, c)] += local.h[(c, r + 3)];
        }
    }

    let mean_cost = w * err.dot(&we);
    local.mean_cost += mean_cost;
    local.cost += mean_cost;

    Some(())
}

fn add_covariance_shape_term(
    local: &mut GaussianLinearSystem,
    corr: &GaussianCorrespondence,
    rotated_source_covariance: Matrix3<f32>,
    options: &GaussianShapeOptions,
) -> Option<()> {
    if options.shape_weight <= 0.0 {
        return Some(());
    }

    let residual = covariance_shape_residual(
        corr.target_covariance,
        rotated_source_covariance,
        options.normalize_shape_by_trace,
        options.min_trace,
    );

    if !is_finite_vector6(&residual) {
        return None;
    }

    let j_rot = numerical_shape_jacobian_rotation(
        corr.target_covariance,
        rotated_source_covariance,
        options.normalize_shape_by_trace,
        options.min_trace,
        options.shape_fd_epsilon,
    );

    if !j_rot.iter().all(|v| v.is_finite()) {
        return None;
    }

    // Gauss-Newton:
    // residual_after_delta ≈ residual + J δ
    // minimize ||residual + Jδ||^2
    // Hδ = -J^T residual
    let w = options.shape_weight;
    let h_rr = j_rot.transpose() * j_rot;
    let b_r = -(j_rot.transpose() * residual);

    for r in 0..3 {
        local.b[r] += w * b_r[r];
        for c in 0..3 {
            local.h[(r, c)] += w * h_rr[(r, c)];
        }
    }

    let shape_cost = w * residual.dot(&residual);
    local.shape_cost += shape_cost;
    local.cost += shape_cost;

    Some(())
}

fn compute_one_gaussian_shape_term(
    corr: &GaussianCorrespondence,
    r_mat: &Matrix3<f32>,
    options: &GaussianShapeOptions,
) -> Option<GaussianLinearSystem> {
    let rotated_source_covariance = r_mat * corr.source_covariance * r_mat.transpose();

    let mut local = GaussianLinearSystem::default();

    add_mean_alignment_term(&mut local, corr, rotated_source_covariance, options)?;
    add_covariance_shape_term(&mut local, corr, rotated_source_covariance, options)?;

    if !is_finite_matrix6(&local.h) || !is_finite_vector6(&local.b) || !local.cost.is_finite() {
        return None;
    }

    local.used_count = 1;
    Some(local)
}

pub fn solve_gaussian_delta(system: &GaussianLinearSystem, damping: f32) -> Option<Vector6f> {
    if system.used_count < 6 {
        return None;
    }

    let mut h = system.h;

    if damping > 0.0 {
        for i in 0..6 {
            h[(i, i)] += damping;
        }
    }

    if let Some(chol) = h.cholesky() {
        return Some(chol.solve(&system.b));
    }

    h.lu().solve(&system.b)
}

pub fn compute_gaussian_shape_linear_system(
    correspondences: &[GaussianCorrespondence],
    source_to_target: &Isometry3<f32>,
    options: &GaussianShapeOptions,
) -> GaussianLinearSystem {
    let r_mat = source_to_target
        .rotation
        .to_rotation_matrix()
        .matrix()
        .clone_owned();

    correspondences
        .par_iter()
        .filter_map(|corr| compute_one_gaussian_shape_term(corr, &r_mat, options))
        .reduce(GaussianLinearSystem::default, |mut acc, item| {
            acc.h += item.h;
            acc.b += item.b;
            acc.cost += item.cost;
            acc.mean_cost += item.mean_cost;
            acc.shape_cost += item.shape_cost;
            acc.used_count += item.used_count;
            acc
        })
}
