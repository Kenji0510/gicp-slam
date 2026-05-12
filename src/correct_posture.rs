use nalgebra::{Point3, UnitQuaternion, Vector3};

use crate::types::{IMU, PointXYZIT};

pub enum CoordSystem {
    /// IMU座標系の重力をLiDAR座標系に変換して補正（デフォルト）
    Lidar,
    /// LiDAR点群をIMU座標系に変換して補正
    Imu,
}

/// IMU → LiDAR 外部パラメータ（DIFOP取得値）
/// g_imu を LiDAR座標系に変換する: g_lidar = R_imu_to_lidar * g_imu
pub const IMU_TO_LIDAR: (f64, f64, f64, f64) = (
    -0.705437, // x
     0.708767, // y
    -0.00246579, // z
     0.00097028, // w
);

pub fn get_lidar_posture(imu_data: &Vec<IMU>, coord: CoordSystem) -> UnitQuaternion<f32> {
    let first_imu_time = imu_data[0].timestamp;
    let one_sec_samples: Vec<_> = imu_data
        .iter()
        .take_while(|d| d.timestamp - first_imu_time <= 1.0)
        .collect();

    let g_in_imu = one_sec_samples
        .iter()
        .fold(Vector3::<f64>::zeros(), |acc, d| {
            acc + Vector3::new(
                d.linear_acceleration[0] as f64,
                d.linear_acceleration[1] as f64,
                d.linear_acceleration[2] as f64,
            )
        })
        / one_sec_samples.len() as f64;

    let norm = g_in_imu.norm();
    log::debug!(
        "Tilt estimation: g_measured=({:.4},{:.4},{:.4}), norm={:.4} [g]",
        g_in_imu.x,
        g_in_imu.y,
        g_in_imu.z,
        norm
    );

    let (ix, iy, iz, iw) = IMU_TO_LIDAR;
    let r_imu_to_lidar = UnitQuaternion::from_quaternion(
        nalgebra::Quaternion::new(iw, ix, iy, iz)
    );

    let g_ref = match coord {
        CoordSystem::Lidar => {
            // IMU座標系の重力をLiDAR座標系に変換
            let g = r_imu_to_lidar * g_in_imu;
            // DIFOPの変換方向が逆の場合は .inverse() を試す:
            // let g = r_imu_to_lidar.inverse() * g_in_imu;
            log::debug!(
                "Tilt estimation: g_in_lidar=({:.4},{:.4},{:.4})",
                g.x, g.y, g.z,
            );
            g
        }
        CoordSystem::Imu => {
            // IMU座標系の重力をそのまま使用（g_in_imuは既にIMU座標系）
            // CoordSystem::Lidarと数学的に等価なため、結果は同じになる
            log::debug!(
                "Tilt estimation: g_in_imu=({:.4},{:.4},{:.4})",
                g_in_imu.x, g_in_imu.y, g_in_imu.z,
            );
            r_imu_to_lidar * g_in_imu // LiDAR空間で同じ補正を適用するために変換
        }
    };

    let tilt_correction = if (norm - 1.0).abs() < 0.1 {
        UnitQuaternion::rotation_between(
            &g_ref.normalize(),
            &Vector3::new(0.0, 0.0, -1.0),
        )
        .unwrap_or(UnitQuaternion::identity())
    } else {
        log::warn!(
            "Tilt estimation skipped: norm={:.4} is too far from 1.0g",
            norm
        );
        UnitQuaternion::identity()
    };

    tilt_correction.cast::<f32>()
}

pub fn correct_point_posture(points: &[PointXYZIT], posture: &UnitQuaternion<f32>) -> Vec<PointXYZIT> {
    points
        .iter()
        .map(|p| {
            let corrected_point = posture * Point3::new(p.x, p.y, p.z);
            PointXYZIT {
                x: corrected_point.x,
                y: corrected_point.y,
                z: corrected_point.z,
                intensity: p.intensity,
                timestamp: p.timestamp,
            }
        })
        .collect()
}