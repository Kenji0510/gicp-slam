use nalgebra::{
    Isometry3, Matrix3, SMatrix, SVector, Vector3,
};
use rayon::prelude::*;

use crate::find_nearest_points::GicpCorrespondence;

pub type Matrix6f = SMatrix<f32, 6, 6>;
pub type Vector6f = SVector<f32, 6>;

#[derive(Debug, Clone)]
pub struct GicpLinearSystem {
    pub h: Matrix6f,
    pub b: Vector6f,

    /// sum(error^T Omega error)
    pub cost: f32,

    /// 実際に使われた対応点数
    pub used_count: usize,
}

impl Default for GicpLinearSystem {
    fn default() -> Self {
        Self {
            h: Matrix6f::zeros(),
            b: Vector6f::zeros(),
            cost: 0.0,
            used_count: 0,
        }
    }
}

#[inline]
fn is_finite_matrix3(m: &Matrix3<f32>) -> bool {
    m.iter().all(|v| v.is_finite())
}

fn invert_covariance_safe(cov: Matrix3<f32>) -> Option<Matrix3<f32>> {
    if !is_finite_matrix3(&cov) {
        return None;
    }

    let det = cov.determinant();

    if !det.is_finite() || det.abs() < 1.0e-12 {
        return None;
    }

    cov.try_inverse()
}

fn compute_one_gicp_term(
    corr: &GicpCorrespondence,
    r_mat: &Matrix3<f32>,
    max_dist_sq: f32,
    rotate_source_covariance: bool,
    covariance_regularization: f32,
) -> Option<GicpLinearSystem> {
    if corr.dist_sq > max_dist_sq {
        return None;
    }

    let ps = corr.transformed_source_mean;
    let pt = corr.target_mean;

    let source_cov = if rotate_source_covariance {
        r_mat * corr.source_covariance * r_mat.transpose()
    } else {
        corr.source_covariance
    };

    let mut c_sum = corr.target_covariance + source_cov;

    // 数値安定化用。不要なら 0.0 を指定する。
    if covariance_regularization > 0.0 {
        c_sum[(0, 0)] += covariance_regularization;
        c_sum[(1, 1)] += covariance_regularization;
        c_sum[(2, 2)] += covariance_regularization;
    }

    let omega = invert_covariance_safe(c_sum)?;

    // error = target - transformed_source
    let err: Vector3<f32> = pt.coords - ps.coords;

    // We = Omega * error
    let we = omega * err;

    let x = ps.x;
    let y = ps.y;
    let z = ps.z;

    let p = ps.coords;

    let mut local_h = Matrix6f::zeros();
    let mut local_b = Vector6f::zeros();

    // b_rot = p x We
    let b_rot = p.cross(&we);

    local_b[0] = b_rot.x;
    local_b[1] = b_rot.y;
    local_b[2] = b_rot.z;

    // b_trans = We
    local_b[3] = we.x;
    local_b[4] = we.y;
    local_b[5] = we.z;

    // CUDA側の s_col と同じ
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

    // H_rr
    local_h[(0, 0)] = hcol0.x;
    local_h[(1, 0)] = hcol0.y;
    local_h[(2, 0)] = hcol0.z;

    local_h[(0, 1)] = hcol1.x;
    local_h[(1, 1)] = hcol1.y;
    local_h[(2, 1)] = hcol1.z;

    local_h[(0, 2)] = hcol2.x;
    local_h[(1, 2)] = hcol2.y;
    local_h[(2, 2)] = hcol2.z;

    // H_tt = Omega
    for r in 0..3 {
        for c in 0..3 {
            local_h[(r + 3, c + 3)] = omega[(r, c)];
        }
    }

    // H_rt
    local_h[(0, 3)] = ws0.x;
    local_h[(0, 4)] = ws0.y;
    local_h[(0, 5)] = ws0.z;

    local_h[(1, 3)] = ws1.x;
    local_h[(1, 4)] = ws1.y;
    local_h[(1, 5)] = ws1.z;

    local_h[(2, 3)] = ws2.x;
    local_h[(2, 4)] = ws2.y;
    local_h[(2, 5)] = ws2.z;

    // H_tr = H_rt^T
    for r in 0..3 {
        for c in 0..3 {
            local_h[(r + 3, c)] = local_h[(c, r + 3)];
        }
    }

    let cost = err.dot(&we);

    Some(GicpLinearSystem {
        h: local_h,
        b: local_b,
        cost,
        used_count: 1,
    })
}

pub fn solve_gicp_delta(
    system: &GicpLinearSystem,
    damping: f32,
) -> Option<Vector6f> {
    if system.used_count < 6 {
        return None;
    }

    let mut h = system.h;

    // Levenberg-Marquardt 的な安定化
    if damping > 0.0 {
        for i in 0..6 {
            h[(i, i)] += damping;
        }
    }

    // まずCholeskyを試す
    if let Some(chol) = h.cholesky() {
        return Some(chol.solve(&system.b));
    }

    // ダメならLUでフォールバック
    h.lu().solve(&system.b)
}

pub fn compute_gicp_linear_system(
    correspondences: &[GicpCorrespondence],
    source_to_target: &Isometry3<f32>,
    max_dist_sq: f32,
    rotate_source_covariance: bool,
    covariance_regularization: f32,
) -> GicpLinearSystem {
    let r_mat = source_to_target
        .rotation
        .to_rotation_matrix()
        .matrix()
        .clone_owned();

    correspondences
        .par_iter()
        .filter_map(|corr| {
            compute_one_gicp_term(
                corr,
                &r_mat,
                max_dist_sq,
                rotate_source_covariance,
                covariance_regularization,
            )
        })
        .reduce(
            GicpLinearSystem::default,
            |mut acc, item| {
                acc.h += item.h;
                acc.b += item.b;
                acc.cost += item.cost;
                acc.used_count += item.used_count;
                acc
            },
        )
}