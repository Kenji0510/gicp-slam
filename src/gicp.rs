use nalgebra::{SMatrix, SVector, Vector3};
use rayon::prelude::*;

use crate::find_nearest_points::GicpCorrespondence;

pub type Matrix6f = SMatrix<f32, 6, 6>;
pub type Vector6f = SVector<f32, 6>;

#[derive(Debug, Clone)]
pub struct GicpLinearSystem {
    pub h: Matrix6f,
    pub b: Vector6f,

    /// sum(e^2)
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
fn is_finite_vector3(v: &Vector3<f32>) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn compute_one_p2plane_term(
    corr: &GicpCorrespondence,
    max_dist_sq: f32,
) -> Option<GicpLinearSystem> {
    if corr.dist_sq > max_dist_sq {
        return None;
    }

    let q = corr.transformed_source_mean.coords;
    let pt = corr.target_mean.coords;
    let n = corr.target_normal;

    if !is_finite_vector3(&n) || !is_finite_vector3(&q) || !is_finite_vector3(&pt) {
        return None;
    }

    // Point-to-Plane 誤差: e = n^T * (pt - q)
    let e = n.dot(&(pt - q));

    // J = [-(q×n)^T, -n^T]  →  b = -e * J^T = e * [q×n; n]
    let q_cross_n = q.cross(&n);

    let j = Vector6f::new(
        q_cross_n.x,
        q_cross_n.y,
        q_cross_n.z,
        n.x,
        n.y,
        n.z,
    );

    // H = J^T * J = j * j^T  (outer product)
    let local_h: Matrix6f = j * j.transpose();
    let local_b: Vector6f = j * e;

    Some(GicpLinearSystem {
        h: local_h,
        b: local_b,
        cost: e * e,
        used_count: 1,
    })
}

pub fn solve_gicp_delta(system: &GicpLinearSystem, damping: f32) -> Option<Vector6f> {
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
    max_dist_sq: f32,
) -> GicpLinearSystem {
    correspondences
        .par_iter()
        .filter_map(|corr| compute_one_p2plane_term(corr, max_dist_sq))
        .reduce(GicpLinearSystem::default, |mut acc, item| {
            acc.h += item.h;
            acc.b += item.b;
            acc.cost += item.cost;
            acc.used_count += item.used_count;
            acc
        })
}
