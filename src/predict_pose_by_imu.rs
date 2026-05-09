use nalgebra::{Matrix4, Unit, UnitQuaternion, Vector3};

use crate::types::{IMU, LoadIMU};

pub fn align_imu_timestamps(imu_data: &Vec<LoadIMU>) -> Vec<IMU> {
    let mut revised_imu_data = Vec::<IMU>::with_capacity(imu_data.len());

    for imu in imu_data {
        revised_imu_data.push(IMU {
            timestamp: imu.timestamp as f64 / 1_000_000_000.0,
            angular_velocity: imu.angular_velocity,
            linear_acceleration: imu.linear_acceleration,
        });
    }

    revised_imu_data
}

#[derive(Debug, Clone)]
pub struct PosePrediction {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    // pub rotation: Vector3<f64>,
    pub delta_transform: Matrix4<f64>,
    /// IMUから積分した回転量（重力除去なしのため並進は不正確）
    pub delta_rotation: UnitQuaternion<f64>,
}

const G: f64 = 9.80665;

pub fn predict_pose_by_imu(
    imu_data: &Vec<IMU>,
    frame_time_range: (f64, f64), // (start_time, end_time) sec
) -> PosePrediction {
    let empty_result = PosePrediction {
        position: Vector3::zeros(),
        velocity: Vector3::zeros(),
        // rotation: Vector3::zeros(),
        delta_transform: Matrix4::identity(),
        delta_rotation: UnitQuaternion::identity(),
    };

    // --- Find the imu data from previous start frame time to current start frame time ---
    let (start_idx, end_idx) = get_imu_range(imu_data, frame_time_range);

    if start_idx == 0 || start_idx >= imu_data.len() || end_idx > imu_data.len() {
        return empty_result;
    }

    let relevant_imu_data = &imu_data[start_idx..end_idx];

    let mut last_time = imu_data[start_idx - 1].timestamp; // Use the timestamp of the last IMU data point before the frame start
    let mut q = UnitQuaternion::<f64>::identity();
    let mut velocity = Vector3::<f64>::zeros();
    let mut position = Vector3::<f64>::zeros();

    for sample in relevant_imu_data {
        let dt = sample.timestamp - last_time;
        if dt <= 1e-9 {
            continue;
        }

        // --- Update rotation ---
        let omega = Vector3::new(
            sample.angular_velocity[0] as f64,
            sample.angular_velocity[1] as f64,
            sample.angular_velocity[2] as f64,
        );

        let angle = omega.norm() * dt;
        let axis = if angle < 1e-9 {
            Vector3::x_axis() // Default axis if angular velocity is very small
        } else {
            Unit::new_normalize(omega * dt)
        };

        let delta_q = UnitQuaternion::from_axis_angle(&axis, angle);
        // delta_rotationはボディ座標系の回転なので右積
        // deskew_points.rsとの一貫性: convert_imu_data.rsも右積に遠いう
        q = q * delta_q;

        // --- Update velocity ---
        let acc = Vector3::new(
            sample.linear_acceleration[0] as f64,
            sample.linear_acceleration[1] as f64,
            sample.linear_acceleration[2] as f64,
        );

        // センサ座標系 → ワールド座標系へ変換
        // linear_acceleration の単位は g なので G を掛けて m/s² に変換
        let acc_world = q * acc * G;

        // ワールド座標系で重力を除去
        // センサZ軸は上向き正, 静止時acc≈[0,0,-1]g → ワールド系重力は[0,0,-G]
        let gravity_world = Vector3::new(0.0, 0.0, -G);
        let acc_no_gravity = acc_world - gravity_world;

        // --- Update position and velocity ---
        velocity += acc_no_gravity * dt;
        position += velocity * dt + 0.5 * acc_no_gravity * dt * dt;

        // Update last_time for the next iteration
        last_time = sample.timestamp;
    }

    let rotation_matrix = q.to_rotation_matrix();
    let mut delta_transform = Matrix4::<f64>::identity();
    delta_transform
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation_matrix.matrix());
    delta_transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&position);

    PosePrediction {
        position,
        velocity,
        // rotation: q.euler_angles().into(),
        delta_transform,
        delta_rotation: q,
    }
}

pub fn get_imu_range(
    imu_data: &Vec<IMU>,
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
