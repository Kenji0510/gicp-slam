use nalgebra::{Point3, Unit, UnitQuaternion, Vector3};

use crate::{
    convert_imu_data::DeltaRotation,
    types::{IMU, PointXYZIT},
};

pub fn deskew_points(imu_data: &Vec<DeltaRotation>, pcd: &Vec<PointXYZIT>) -> Vec<Point3<f32>> {
    let (start_time, end_time) = get_time_for_start_and_end(pcd);

    let (start_idx, end_idx) = get_imu_range(&imu_data, (start_time, end_time));
    let relevant_imu_data = &imu_data[start_idx..end_idx];
    let mut deskewed_point_vecs: Vec<Point3<f32>> = Vec::with_capacity(pcd.len());

    // Deskewing each points
    for p in pcd {
        let x = p.x as f32;
        let y = p.y as f32;
        let z = p.z as f32;
        let point_time = p.timestamp;

        let rotation = get_rotation_at_time(relevant_imu_data, point_time);

        let point_vec = Point3::new(x, y, z);
        let deskewed_point = rotation.cast::<f32>() * point_vec;
        deskewed_point_vecs.push(deskewed_point);
    }

    deskewed_point_vecs
}

fn get_time_for_start_and_end(pcd: &Vec<PointXYZIT>) -> (f64, f64) {
    let start = pcd
        .iter()
        .map(|p| p.timestamp)
        .fold(f64::INFINITY, |a, b| a.min(b));
    let end = pcd
        .iter()
        .map(|p| p.timestamp)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    (start, end)
}

fn get_rotation_at_time(imu_data: &[DeltaRotation], timestamp: f64) -> UnitQuaternion<f64> {
    let mut rotation = UnitQuaternion::<f64>::identity();

    for delta in imu_data {
        if delta.timestamp > timestamp {
            break;
        }
        rotation = rotation * delta.delta_rotation; // body-frame ω → right-compose
    }

    rotation
}

pub fn get_imu_range(
    imu_data: &Vec<DeltaRotation>,
    frame_time_range: (f64, f64), // (start_time, end_time) sec
) -> (usize, usize) {
    let start_idx = imu_data
        .iter()
        .position(|s| s.timestamp >= frame_time_range.0)
        .unwrap_or(0);

    let end_idx = imu_data
        .iter()
        .position(|s| s.timestamp >= frame_time_range.1)
        .unwrap_or(imu_data.len() - 1);

    (start_idx, end_idx)
}
