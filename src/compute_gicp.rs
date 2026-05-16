use std::ops::AddAssign;

use nalgebra::{Matrix3, Matrix4, Matrix6, UnitQuaternion, Vector3, Vector6};

use crate::find_nearest_points::Correspondence;

pub struct GicpLinearSystem {
    pub h: Matrix6<f32>,
    pub b: Vector6<f32>,
}

/// p のskew-symmetric行列 [p]×
/// [p]× v = p × v
fn skew(v: &Vector3<f32>) -> Matrix3<f32> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// GICP線形システムを構築する。
///
/// CUDAコードと同じ定式化:
///   J = [-[p_s]×  |  I]   (Lie代数 [rot; trans] に対するJacobian)
///   H = J^T Ω J,  b = J^T Ω err,  err = p_t - p_s
///
/// source/target のボクセルはすでにグローバル座標に変換済みであること。
pub fn compute_gicp_linear_system(correspondences: &[Correspondence]) -> GicpLinearSystem {
    let mut h = Matrix6::<f32>::zeros();
    let mut b = Vector6::<f32>::zeros();

    for corr in correspondences {
        let ps = corr.source_cell.mean.coords; // Vector3<f32>
        let pt = corr.target_cell.mean.coords; // Vector3<f32>
        let cs = corr.source_cell.covariance; // Matrix3<f32>
        let ct = corr.target_cell.covariance; // Matrix3<f32>

        // C_sum = C_t + C_s,  Omega = C_sum^{-1}
        let c_sum = ct + cs;
        let Some(omega) = c_sum.try_inverse() else {
            continue;
        };

        // err = p_t - p_s,  We = Ω err
        let err = pt - ps;
        let we = omega * err;

        // b = [p_s × We ; We]
        let b_rot = ps.cross(&we);
        let b_trans = we;
        b.fixed_rows_mut::<3>(0).add_assign(b_rot);
        b.fixed_rows_mut::<3>(3).add_assign(b_trans);

        // S = -[p_s]×  (CUDA の s0,s1,s2 がこの各列)
        let neg_skew_ps = -skew(&ps);

        // WS = Ω * S  (列 i = Ω * s_i)
        let ws = omega * neg_skew_ps;

        let ws0: Vector3<f32> = ws.column(0).into();
        let ws1: Vector3<f32> = ws.column(1).into();
        let ws2: Vector3<f32> = ws.column(2).into();

        // H_rr: 列 i = p_s × ws_i
        let h_rr = Matrix3::from_columns(&[ps.cross(&ws0), ps.cross(&ws1), ps.cross(&ws2)]);

        // H_rt: 行 i = ws_i  →  ws^T
        // H_tr = H_rt^T = ws
        // H_tt = Ω
        h.fixed_view_mut::<3, 3>(0, 0).add_assign(h_rr);
        h.fixed_view_mut::<3, 3>(0, 3).add_assign(ws.transpose()); // H_rt
        h.fixed_view_mut::<3, 3>(3, 0).add_assign(ws); // H_tr
        h.fixed_view_mut::<3, 3>(3, 3).add_assign(omega); // H_tt
    }

    GicpLinearSystem { h, b }
}

/// H * δx = b を解き、delta変換行列 (4x4, f64) を返す。
/// δx = [δrot(3); δtrans(3)]  (Lie代数表現)
/// damping: Levenberg-Marquardt ダンピング係数（発散防止）
pub fn solve_gicp(system: &GicpLinearSystem, damping: f32) -> Option<Matrix4<f64>> {
    let mut h_damped = system.h;
    for i in 0..6 {
        h_damped[(i, i)] += damping;
    }

    let delta = h_damped.lu().solve(&system.b)?;

    let rot_vec = Vector3::new(delta[0], delta[1], delta[2]);
    let trans_vec = Vector3::new(delta[3], delta[4], delta[5]);

    let angle = rot_vec.norm();
    let rotation = if angle < 1.0e-10 {
        UnitQuaternion::identity()
    } else {
        UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(rot_vec), angle)
    };

    let mut mat = rotation.to_homogeneous().cast::<f64>();
    mat[(0, 3)] = trans_vec.x as f64;
    mat[(1, 3)] = trans_vec.y as f64;
    mat[(2, 3)] = trans_vec.z as f64;

    Some(mat)
}
