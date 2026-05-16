use std::collections::VecDeque;

use anyhow::Result;
use lidar_slam::{
    compute_covariance::{
        build_gicp_voxel_map, merge_points_into_gaussian_voxel_map, transform_voxel_map,
    },
    compute_gicp::{compute_gicp_linear_system, solve_gicp},
    convert_imu_data::convert_imu_data,
    convert_type::{convert_pcd_to_xyz, convert_xyz_to_pcd},
    debug::{DebugData, DebugProcessTime, convert_voxel_map_to_pcd},
    deskew_points::deskew_points,
    file_handler::{
        load_imu_data, load_pcd_files, load_pcd_xyzit, save_pcd_xyzcov, save_pcd_xyzit,
    },
    find_nearest_points::{Correspondence, find_nearest_voxels},
    predict_pose_by_imu::{align_imu_timestamps, build_rotation_trajectory, predict_pose_by_imu},
    tilt_correction::{apply_tilt_correction, compute_tilt_correction, estimate_gravity_from_imu},
    transform::transform_points_to_global_frame,
    types::FrameData,
    voxelization::voxel_downsample_points,
};
use nalgebra::{Isometry3, Matrix4, Point3, Quaternion, Translation3, UnitQuaternion, Vector3};

const LOAD_DIR: &str = "/home/kenji/workspace/rust/get_lidar_data/data/output/05092026/hallway";
const SAVE_DIR: &str = "data/output/debug/05162026";

const DOWNSAMPLE_VOXEL_SIZE: f32 = 0.1; // m
const GICP_ITERATIONS: usize = 7;

const MIN_DIST: f32 = 0.1;
const MAX_DIST: f32 = 48.0;

const MAX_POINTS_PER_VOXEL: usize = 10;
const MIN_POINTS_PER_VOXEL: usize = 3;

const SEARCH_RANGE: i32 = 3; // Range of 5x5x5 voxels
const MAX_DIST_SQ: f32 = 1.0; // Optional maximum distance squared

// IMU coordination to LiDAR coordination (Robosense 96 beam)
// Quaternion (x, y, z, w): -0.705437, 0.708767, -0.00246579, 0.00097028
// Translation (x, y, z)  : 0.00425, 0.00418, -0.00446  [m]
const IMU_TO_LIDAR_QUAT_X: f64 = -0.705437;
const IMU_TO_LIDAR_QUAT_Y: f64 = 0.708767;
const IMU_TO_LIDAR_QUAT_Z: f64 = -0.00246579;
const IMU_TO_LIDAR_QUAT_W: f64 = 0.00097028;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    let pcd_dir = format!("{}/pcd", LOAD_DIR);
    let pcd_files = load_pcd_files(&pcd_dir)?;

    log::debug!(
        "Found {} PCD files in directory: {}",
        pcd_files.len(),
        pcd_dir
    );

    let imu_dir = format!("{}/imu", LOAD_DIR);
    let imu_file = format!("{}/imu_data.json", imu_dir);
    let imu_data = load_imu_data(&imu_file)?;
    let imu_data = align_imu_timestamps(&imu_data); // Align IMU timestamps to seconds

    log::debug!(
        "Loaded {} IMU data entries from file: {}",
        imu_data.len(),
        imu_file
    );

    println!("{}", imu_data[0].timestamp);

    let mut debug_data: Vec<DebugData> = Vec::new();
    let mut debug_process_times: Vec<DebugProcessTime> = Vec::new();

    // IMU coord to LiDAR coord transformation
    let imu_to_lidar = UnitQuaternion::new_normalize(Quaternion::new(
        IMU_TO_LIDAR_QUAT_W,
        IMU_TO_LIDAR_QUAT_X,
        IMU_TO_LIDAR_QUAT_Y,
        IMU_TO_LIDAR_QUAT_Z,
    ));

    let mut current_global_pose = Matrix4::<f64>::identity();
    let mut current_velocity = Vector3::<f64>::zeros();
    let mut last_frame_time = imu_data[0].timestamp;

    let mut local_map_queue: VecDeque<FrameData> = VecDeque::new();
    let mut global_map_queue: VecDeque<FrameData> = VecDeque::new();
    let mut global_map_accumulator: Vec<Point3<f32>> = Vec::new();
    // global_map_queue.push()

    let pcd = load_pcd_xyzit(&pcd_files[0].to_string_lossy())?;
    let points = convert_pcd_to_xyz(&pcd);

    // --- Downsample for density normalization ---
    let downsample_voxel_size = DOWNSAMPLE_VOXEL_SIZE;

    let downsampled_init_points = voxel_downsample_points(&points, downsample_voxel_size);
    // --- Downsample for density normalization ---

    // global_map_accumulator.extend(downsampled_init_points);

    let mut target_voxel_map = build_gicp_voxel_map(
        &downsampled_init_points,
        downsample_voxel_size,
        MAX_POINTS_PER_VOXEL,
        MIN_POINTS_PER_VOXEL,
    );

    let mut prev_frame_start_time = pcd
        .iter()
        .map(|p| p.timestamp)
        .fold(f64::INFINITY, f64::min);

    for (i, pcd_path) in pcd_files.iter().enumerate().skip(1) {
        log::info!("Processing frame {}: {}", i, pcd_path.to_string_lossy());

        let source_pcd = load_pcd_xyzit(&pcd_path.to_string_lossy())?;

        // --- Predict pose by IMU ---
        let current_frame_start_time = source_pcd
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::INFINITY, f64::min);
        let current_frame_end_time = source_pcd
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::NEG_INFINITY, f64::max);

        // 既にGlobal座標でのcurrent_global_poseがある状態で、次フレームの開始時刻までのIMU積分を行う。
        let pose_prediction = predict_pose_by_imu(
            &imu_data,
            &imu_to_lidar,
            &current_global_pose,
            &current_velocity,
            prev_frame_start_time,
            current_frame_start_time,
        );

        let rotation_traj = build_rotation_trajectory(
            &imu_data,
            current_frame_start_time,
            current_frame_end_time,
            &imu_to_lidar,
        );

        // --- Deskew source pcd ---
        let deskewed_points = deskew_points(
            &source_pcd,
            &rotation_traj,
            &imu_to_lidar,
            current_frame_start_time,
            MIN_DIST,
            MAX_DIST,
        );
        // --- Deskew source pcd --- 05132026

        // --- Downsample the deskewed point cloud ---
        let start = std::time::Instant::now();
        let downsampled_points = voxel_downsample_points(&deskewed_points, downsample_voxel_size);
        let voxelization_time_ms = start.elapsed().as_secs_f32() * 1000.0;
        // --- Downsample the deskewed point cloud ---

        let mut current_transform = pose_prediction.0;

        // --- Build source Voxel voxel map ---
        let start = std::time::Instant::now();
        let source_voxel_map = build_gicp_voxel_map(
            &downsampled_points,
            downsample_voxel_size,
            MAX_POINTS_PER_VOXEL,
            MIN_POINTS_PER_VOXEL,
        );
        let create_voxel_map_time_ms = start.elapsed().as_secs_f32() * 1000.0;
        // --- Build source Voxel voxel map ---

        let mut dist = 0.0;
        let mut cnt = 0usize;
        let mut correspondence_num: usize = 0;

        let start = std::time::Instant::now();
        let mut find_nearest_time_ms = 0.0f32;
        let mut gicp_time_ms = 0.0f32;
        for i in 0..GICP_ITERATIONS {
            // --- Transform source voxel map to global frame ---
            let transformed_source_voxel_map =
                transform_voxel_map(&source_voxel_map, &current_transform, downsample_voxel_size);
            // --- Transform source voxel map to global frame ---

            // --- find nearest points ---
            let start_find = std::time::Instant::now();
            let correspondences = find_nearest_voxels(
                &transformed_source_voxel_map,
                &target_voxel_map,
                downsample_voxel_size,
                SEARCH_RANGE,
                Some(MAX_DIST_SQ),
            );
            find_nearest_time_ms += start_find.elapsed().as_secs_f32() * 1000.0;
            // --- find nearest points ---

            // --- DEBUG ---
            correspondence_num = correspondences.len();
            for c in &correspondences {
                dist += c.dist_sq;
            }
            log::debug!(
                "Average euclidean distance of correspondences: {}",
                dist / correspondences.len() as f32
            );
            for c in &correspondences {
                if MAX_DIST_SQ >= c.dist_sq {
                    cnt += 1;
                }
            }
            log::debug!(
                "Number of correspondences within max distance: {} / {}",
                cnt,
                correspondences.len()
            );
            // --- DEBUG ---


            // --- compute GICP ---
            let start_gicp = std::time::Instant::now();
            let gicp_result = compute_gicp_linear_system(&correspondences);
            if let Some(delta) = solve_gicp(&gicp_result, 1.0e-4) {
                current_transform = delta * current_transform;
            }
            gicp_time_ms += start_gicp.elapsed().as_secs_f32() * 1000.0;
            // --- compute GICP ---
        }

        debug_data.push(DebugData {
            correspondences_num: correspondence_num,
            dist: dist / correspondence_num as f32,
        });

        // --- Update target_voxel_map for the next frame ---
        let start = std::time::Instant::now();
        merge_points_into_gaussian_voxel_map(
            &mut target_voxel_map,
            &downsampled_points,
            &current_transform,
            downsample_voxel_size,
            MAX_POINTS_PER_VOXEL,
            MIN_POINTS_PER_VOXEL,
        );
        let merge_time_ms = start.elapsed().as_secs_f32() * 1000.0;
        // --- Update target_voxel_map for the next frame ---

        debug_process_times.push(DebugProcessTime {
            voxelization_time_ms,
            create_voxel_map_time_ms,
            find_correspondences_time_ms: find_nearest_time_ms / GICP_ITERATIONS as f32,
            gicp_time_ms: gicp_time_ms / GICP_ITERATIONS as f32,
            total_gicp_time_ms: (find_nearest_time_ms + gicp_time_ms) / GICP_ITERATIONS as f32,
            merge_time_ms,
        });

        // GICP補正後の位置差分から速度を推定（IMU積分のバイアス蓄積を避ける）
        let prev_pos = current_global_pose.fixed_view::<3, 1>(0, 3).into_owned();
        let new_pos = current_transform.fixed_view::<3, 1>(0, 3).into_owned();
        let dt = (current_frame_start_time - prev_frame_start_time).max(1e-6);
        current_velocity = ((new_pos - prev_pos) / dt).cap_magnitude(2.0); // 速度の上限を2 m/sに設定
        current_global_pose = current_transform;
        prev_frame_start_time = current_frame_start_time; // 次フレームのIMU積分の開始時刻を更新
    }

    // --- Debug: Save final voxel map as PCD ---
    let final_pcd = convert_voxel_map_to_pcd(&target_voxel_map);
    let save_file_path = format!(
        "{}/final_gicp_map_v-{}.pcd",
        SAVE_DIR, downsample_voxel_size
    );
    save_pcd_xyzcov(&final_pcd, &save_file_path)?;
    log::info!("Saved final GICP voxel map as PCD: {}", save_file_path);

    // --- Save debug_data as JSON ---
    let debug_log_path = format!("{}/debug_gicp_data.json", SAVE_DIR);
    std::fs::write(&debug_log_path, serde_json::to_string_pretty(&debug_data)?)?;
    log::info!("Saved debug data ({} frames): {}", debug_data.len(), debug_log_path);

    let debug_log_path = format!("{}/debug_process_times.json", SAVE_DIR);
    std::fs::write(&debug_log_path, serde_json::to_string_pretty(&debug_process_times)?)?;
    log::info!("Saved debug process times ({} frames): {}", debug_process_times.len(), debug_log_path);
    // --- Save debug_data as JSON ---

    Ok(())
}
// --- Debug: Save final voxel map as PCD ---

// for i in 0..GAUSSIAN_ITERATIONS {
//     // --- find nearest points ---
//     let search_range = 5; // 3x3x3 voxels
//     let max_euclidean_dist_sq = Some(1.0_f32);
//     let max_mahalanobis_dist = Some(25.0_f32);
//     let correspondences = find_gaussian_correspondences(
//         &source_voxel_map,
//         &target_voxel_map,
//         gaussian_voxel_size,
//         &source_to_target,
//         search_range,
//         max_euclidean_dist_sq,
//         max_mahalanobis_dist,
//         covariance_regularization,
//     );
//     last_num_correspondences = correspondences.len();

//     if correspondences.is_empty() {
//         log::warn!(
//             "Frame {}: No correspondences found at iteration {}, skipping frame",
//             i,
//             i
//         );
//         break;
//     }

//     let mut dist = 0.0;
//     for c in &correspondences {
//         dist += c.euclidean_dist_sq;
//     }
//     log::debug!(
//         "Average euclidean distance of correspondences: {}",
//         dist / correspondences.len() as f32
//     );
//     // --- find nearest points ---

//     // --- compute Gaussian shape matching ---
//     // mean-to-mean項 + covariance shape-to-shape項。
//     let shape_options = GaussianShapeOptions {
//         mean_weight: 1.0,
//         shape_weight: 0.05,
//         covariance_regularization,
//         max_euclidean_dist_sq: max_euclidean_dist_sq.unwrap(),
//         max_mahalanobis_dist,
//         shape_fd_epsilon: 1.0e-3,
//         normalize_shape_by_trace: true,
//         min_trace: 1.0e-6,
//     };

//     let system = compute_gaussian_shape_linear_system(
//         &correspondences,
//         &source_to_target,
//         &shape_options,
//     );

//     // H_ttの最小固有値 ≈ 0.5（壁接線方向）。1e-6では無効 → 0.1で抑制
//     let damping = 0.1;

//     // Update source_to_target for the next iteration
//     if let Some(delta) = solve_gaussian_delta(&system, damping) {
//         // delta[0..3] = 回転（Lie代数ベクトル）, delta[3..6] = 並進
//         // 発散防止：deltaが大きすぎる場合はスケールダウン
//         let max_rot_norm = 0.1_f32; // rad
//         let max_trans_norm = 0.5_f32; // m
//         let rot_vec = nalgebra::Vector3::new(delta[0], delta[1], delta[2]);
//         let trans_vec = nalgebra::Vector3::new(delta[3], delta[4], delta[5]);
//         let rot_vec = if rot_vec.norm() > max_rot_norm {
//             rot_vec.normalize() * max_rot_norm
//         } else {
//             rot_vec
//         };
//         let trans_vec = if trans_vec.norm() > max_trans_norm {
//             trans_vec.normalize() * max_trans_norm
//         } else {
//             trans_vec
//         };

//         let angle = rot_vec.norm();
//         let rotation = if angle < 1.0e-10 {
//             UnitQuaternion::identity()
//         } else {
//             UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(rot_vec), angle)
//         };
//         let translation = Translation3::new(trans_vec.x, trans_vec.y, trans_vec.z);
//         let delta_isometry = Isometry3::from_parts(translation, rotation);
//         source_to_target = delta_isometry * source_to_target;

//         let rot_norm = rot_vec.norm();
//         let trans_norm = trans_vec.norm();
//         log::debug!(
//             "Gaussian iter {}: rot={:.4} rad, trans={:.4} m",
//             i,
//             rot_norm,
//             trans_norm
//         );
//         if rot_norm < 1.0e-4 && trans_norm < 1.0e-4 {
//             log::debug!("Gaussian converged at iteration {}", i);
//             break;
//         }
//     } else {
//         log::warn!(
//             "Gaussian failed to solve (insufficient correspondences) at iteration {}",
//             i
//         );
//         break;
//     }
//     // --- compute GICP ---
// }

// // 発散チェック：1フレームで1m以上動いたら棄却
// let translation_diff =
//     (source_to_target.translation.vector - last_good_pose.translation.vector).norm();
// let proposed_tx = source_to_target.translation.x;
// let proposed_ty = source_to_target.translation.y;
// let proposed_tz = source_to_target.translation.z;
// let (r, p, y) = source_to_target.rotation.euler_angles();
// log::info!(
//     "Frame {:>4}: trans_diff={:.4}m  pose=({:.3},{:.3},{:.3})  rot_rpy=({:.3},{:.3},{:.3})",
//     i,
//     translation_diff,
//     source_to_target.translation.x,
//     source_to_target.translation.y,
//     source_to_target.translation.z,
//     r,
//     p,
//     y,
// );

// if translation_diff > 1.0 {
//     log::warn!(
//         "Frame {}: Gaussian diverged ({:.4}m), reverting pose and skipping map update",
//         i,
//         translation_diff
//     );
//     source_to_target = last_good_pose;
//     consecutive_skip_count += 1;
// } else {
//     // --- Update target_voxel_map for the next frame ---
//     merge_points_into_gaussian_voxel_map(
//         &mut target_voxel_map,
//         &downsampled_points,
//         &source_to_target,
//         gaussian_voxel_size,
//         max_points_per_gaussian,
//         min_points_per_gaussian,
//         covariance_min_variance,
//         covariance_max_variance,
//         covariance_regularization,
//     );
//     if consecutive_skip_count >= MAX_CONSECUTIVE_SKIPS {
//         log::warn!(
//             "Frame {}: forced map update after {} consecutive skips",
//             i,
//             consecutive_skip_count
//         );
//     }
//     consecutive_skip_count = 0;
//     log::debug!(
//         "Frame {}: Map updated. pose = {:?}",
//         i,
//         source_to_target.translation
//     );
//     //  --- Update target_voxel_map for the next frame ---
// }

// pose_log.push(serde_json::json!({
//     "frame":              i,
//     "timestamp":          source_pcd_start_time,
//     "proposed_tx":        proposed_tx,
//     "proposed_ty":        proposed_ty,
//     "proposed_tz":        proposed_tz,
//     "proposed_roll":      r,
//     "proposed_pitch":     p,
//     "proposed_yaw":       y,
//     "translation_diff":   translation_diff,
//     "consecutive_skips":  consecutive_skip_count,
//     "num_correspondences":last_num_correspondences,
// }));

//     prev_frame_time = current_frame_start_time; // 次フレームのIMU積分の開始時刻を更新
// }

// --- Save pose log as JSON ---
// let log_path = format!("{}/pose_log.json", SAVE_DIR);
// std::fs::write(&log_path, serde_json::to_string_pretty(&pose_log)?)?;
// log::info!("Saved pose log ({} frames): {}", pose_log.len(), log_path);
// // --- Save pose log as JSON ---

// // --- Debug: Save final voxel map as PCD ---
// let final_pcd = convert_voxel_map_to_pcd(&target_voxel_map);
// let save_file_path = format!(
//     "{}/final_gaussian_map_v-{}.pcd",
//     SAVE_DIR, gaussian_voxel_size
// );
// save_pcd_xyzcov(&final_pcd, &save_file_path)?;
// log::info!("Saved final Gaussian voxel map as PCD: {}", save_file_path);
// --- Debug: Save final voxel map as PCD ---

// let downsampled_pcd = convert_xyz_to_pcd(&downsampled_points);

// let save_file_path = format!(
//     "{}/downsampled_{}_{}.pcd",
//     SAVE_DIR,
//     voxel_size,
//     pcd_files[0].file_stem().unwrap().to_string_lossy()
// );
// save_pcd_xyzit(&downsampled_pcd, &save_file_path)?;
// --- Downsample the point cloud ---

// --- Deskew the point cloud ---
// let imu_data = convert_imu_data(&imu_data);
// let deskewed_points = deskew_points(&imu_data, &pcd);

// let deskewed_pcd = convert_xyz_to_pcd(&deskewed_points);

// let save_file_path = format!(
//     "{}/deskewed_{}.pcd",
//     SAVE_DIR,
//     pcd_files[0].file_stem().unwrap().to_string_lossy()
// );
// save_pcd_xyzit(&deskewed_pcd, &save_file_path)?;
// --- Deskew the point cloud ---

// --- Compute covariance matrices ---
// let start = std::time::Instant::now();

// let voxel_size = 0.1;
// let max_points_per_voxel = 15;
// let k_neighbors = 10;
// let source_voxel_map = build_gicp_voxel_map(
//     &downsampled_init_points,
//     voxel_size,
//     max_points_per_voxel,
//     k_neighbors,
// );

// let duration = start.elapsed();
// log::debug!("Covariance computation took: {:?}", duration);

// let pcd = convert_voxel_map_to_pcd(&source_voxel_map);
// let save_file_path = format!(
//     "{}/voxel_covariances_{}.pcd",
//     SAVE_DIR,
//     pcd_files[0].file_stem().unwrap().to_string_lossy()
// );
// save_pcd_xyzcov(&pcd, &save_file_path)?;

// // --- Compute covariance matrices ---

// // --- find nearest points ---
// let debug_voxel_map = source_voxel_map.clone();
// let search_range = 3; // Range of 7x7x7 voxels
// let max_dist_sq = Some(1.0); // Optional maximum distance squared
// let source_to_target =
//     Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());

// let correspondences = find_gicp_correspondences(
//     &source_voxel_map,
//     &debug_voxel_map,
//     voxel_size,
//     &source_to_target,
//     search_range,
//     max_dist_sq,
// );

// let mut dist = 0.0;
// for c in &correspondences {
//     dist += c.euclidean_dist_sq;
// }
// log::debug!(
//     "Average euclidean distance of correspondences: {}",
//     dist / correspondences.len() as f32
// );

// log::debug!("Found {} correspondences", correspondences.len());

// // --- find nearest points ---

// // --- compute GICP ---
// let rotate_source_covariance = true;
// let covariance_regularization = 1.0e-6;

// let system = compute_gicp_linear_system(
//     &correspondences,
//     &source_to_target,
//     max_dist_sq.unwrap(),
//     rotate_source_covariance,
//     covariance_regularization,
// );

// println!("used correspondences: {}", system.used_count);
// println!("cost: {}", system.cost);

// let damping = 1.0e-6;

// if let Some(delta) = solve_gaussian_delta(&system, damping) {
//     println!("GICP delta (tx, ty, tz, rx, ry, rz):");
//     println!(
//         "  translation: ({:.6}, {:.6}, {:.6})",
//         delta[0], delta[1], delta[2]
//     );
//     println!(
//         "  rotation:    ({:.6}, {:.6}, {:.6})",
//         delta[3], delta[4], delta[5]
//     );

//     let translation = Translation3::new(delta[0], delta[1], delta[2]);
//     let rotation = UnitQuaternion::from_euler_angles(delta[3], delta[4], delta[5]);
//     let delta_isometry = Isometry3::from_parts(translation, rotation);
//     let updated_pose = delta_isometry * source_to_target;
//     println!("Updated pose:");
//     println!("  translation: {:?}", updated_pose.translation);
//     println!("  rotation:    {:?}", updated_pose.rotation);
// } else {
//     println!("Gaussian failed to solve (insufficient correspondences)");
// }
// --- compute GICP ---
