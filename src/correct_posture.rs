use nalgebra::{Point3, UnitQuaternion, Vector3};

use crate::types::IMU;

pub fn get_lidar_posture(imu_data: &Vec<IMU>) -> UnitQuaternion<f32> {
    let first_imu_time = imu_data[0].timestamp;
    let one_sec_samples: Vec<_> = imu_data
        .iter()
        .take_while(|d| d.timestamp - first_imu_time <= 1.0)
        .collect();

    let init_posture_angle = one_sec_samples
        .iter()
        .fold(Vector3::<f64>::zeros(), |acc, d| {
            acc + Vector3::new(
                d.linear_acceleration[0] as f64,
                d.linear_acceleration[1] as f64,
                d.linear_acceleration[2] as f64,
            )
        })
        / one_sec_samples.len() as f64;

    let norm = init_posture_angle.norm();
    log::debug!(
        "Tilt estimation: g_measured=({:.4},{:.4},{:.4}), norm={:.4} [g]",
        init_posture_angle.x,
        init_posture_angle.y,
        init_posture_angle.z,
        norm
    );

    let tilt_correction = if (norm - 1.0).abs() < 0.1 {
        nalgebra::UnitQuaternion::rotation_between(
            &init_posture_angle.normalize(),
            &nalgebra::Vector3::new(0.0, 0.0, -1.0),
        )
        .unwrap_or(nalgebra::UnitQuaternion::identity())
    } else {
        log::warn!(
            "Tilt estimation skipped: norm={:.4} is too far from 1.0g",
            norm
        );
        nalgebra::UnitQuaternion::identity()
    };

    tilt_correction.cast::<f32>()
}
