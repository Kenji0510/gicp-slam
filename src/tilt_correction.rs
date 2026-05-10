use nalgebra::{Point3, UnitQuaternion, Vector3};

use crate::types::IMU;

/// フレーム時刻の前後 `window_sec` 秒以内のIMUサンプルから
/// linear_acceleration を平均し、重力方向ベクトルを推定する。
/// 該当するサンプルがない場合は全サンプルの平均を使う。
pub fn estimate_gravity_from_imu(
    imu_data: &[IMU],
    frame_time: f64,
    window_sec: f64,
) -> Vector3<f32> {
    let samples: Vec<_> = imu_data
        .iter()
        .filter(|s| (s.timestamp - frame_time).abs() <= window_sec)
        .collect();

    let samples = if samples.is_empty() {
        imu_data.iter().collect::<Vec<_>>()
    } else {
        samples
    };

    if samples.is_empty() {
        return Vector3::new(0.0_f32, 0.0, 1.0);
    }

    let mut sum = Vector3::<f32>::zeros();
    for s in &samples {
        sum += Vector3::new(
            s.linear_acceleration[0],
            s.linear_acceleration[1],
            s.linear_acceleration[2],
        );
    }
    sum / samples.len() as f32
}

/// 測定された重力ベクトルから、センサを水平にする補正クォータニオンを計算する。
///
/// 加速度計が示す重力方向を最近傍のZ軸（±Z）に揃える最小回転を返す。
/// これを点群に適用すると、地面が水平になるよう傾きが補正される。
pub fn compute_tilt_correction(gravity: &Vector3<f32>) -> UnitQuaternion<f32> {
    let norm = gravity.norm();
    if norm < 1e-6 {
        return UnitQuaternion::identity();
    }
    let gravity_dir = gravity / norm;

    // センサのZ軸が上向き・下向きどちらでも対応できるよう、
    // 重力方向に近い方のZ軸を参照方向とする
    let reference = if gravity_dir.z >= 0.0 {
        Vector3::new(0.0_f32, 0.0, 1.0)
    } else {
        Vector3::new(0.0_f32, 0.0, -1.0)
    };

    UnitQuaternion::rotation_between(&gravity_dir, &reference)
        .unwrap_or(UnitQuaternion::identity())
}

/// 点群に傾き補正回転を適用して水平化した点群を返す。
pub fn apply_tilt_correction(
    points: &[Point3<f32>],
    correction: &UnitQuaternion<f32>,
) -> Vec<Point3<f32>> {
    points.iter().map(|p| correction * p).collect()
}
