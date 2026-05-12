use anyhow::Result;
use lidar_slam::{
    compute_covariance::{build_gicp_voxel_map, merge_points_into_voxel_map},
    convert_imu_data::convert_imu_data,
    convert_type::{convert_pcd_to_xyz, convert_xyz_to_pcd},
    correct_posture::{CoordSystem, correct_point_posture, get_lidar_posture},
    debug::convert_voxel_map_to_pcd,
    deskew_points::deskew_points,
    file_handler::{
        load_imu_data, load_pcd_files, load_pcd_xyzit, save_pcd_xyzcov, save_pcd_xyzit,
    },
    find_nearest_points::find_gicp_correspondences,
    gicp::{compute_gicp_linear_system, solve_gicp_delta},
    predict_pose_by_imu::{align_imu_timestamps, predict_pose_by_imu},
    voxelization::voxel_downsample_points,
};
use nalgebra::{Isometry3, Translation3, UnitQuaternion};

const LOAD_DIR: &str = "/home/kenji/workspace/rust/get_lidar_data/data/output/05092026/park05";
const SAVE_DIR: &str = "data/output/debug/05102026/debug";

const GICP_ITERATIONS: usize = 7;

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

    // --- Get lidar posture ---
    let lidar_posture = get_lidar_posture(&imu_data, CoordSystem::Lidar);
    log::debug!(
        "Initial lidar posture: quaternion=({:.4}, {:.4}, {:.4}, {:.4})",
        lidar_posture.w,
        lidar_posture.i,
        lidar_posture.j,
        lidar_posture.k
    );
    // --- Get lidar posture ---

    // --- Correction lidar posture ---
    let original_pcd = load_pcd_xyzit(&pcd_files[0].to_string_lossy())?;
    let corrected_points = correct_point_posture(&original_pcd, &lidar_posture);

    let save_file_path = format!(
        "{}/corrected_posture.pcd",
        SAVE_DIR,
    );
    save_pcd_xyzit(&corrected_points, &save_file_path)?;
    // --- Correction lidar posture ---






    // println!("{}", imu_data[0].timestamp);

    // let pcd = load_pcd_xyzit(&pcd_files[0].to_string_lossy())?;
    // let points = convert_pcd_to_xyz(&pcd);

    // // --- Downsample the point cloud ---
    // let voxel_size = 0.1;
    // let downsampled_init_points = voxel_downsample_points(&points, voxel_size);
    // // --- Downsample the point cloud ---
    // let max_points_per_voxel = 15;
    // let k_neighbors = 8;

    // let mut target_voxel_map = build_gicp_voxel_map(
    //     &downsampled_init_points,
    //     voxel_size,
    //     max_points_per_voxel, // max points per voxel
    //     k_neighbors,          // k neighbors for covariance estimation
    // );

    // let converted_imu_data = convert_imu_data(&imu_data);
    // let pcd_start_time = pcd
    //     .iter()
    //     .map(|p| p.timestamp)
    //     .fold(f64::INFINITY, f64::min);
    // let mut source_to_target =
    //     Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());

    // let max_points_per_voxel = 15;
    // let k_neighbors = 8;
    // let mut prev_frame_time = pcd_start_time; // フレーム間のIMU積分用

    // for (i, pcd_path) in pcd_files.iter().enumerate().skip(1) {
    //     let source_pcd = load_pcd_xyzit(&pcd_path.to_string_lossy())?;

    //     // --- Predict pose by IMU ---
    //     let source_pcd_start_time = source_pcd
    //         .iter()
    //         .map(|p| p.timestamp)
    //         .fold(f64::INFINITY, f64::min);

    //     let frame_time_range = (prev_frame_time, source_pcd_start_time);
    //     let pose_prediction = predict_pose_by_imu(&imu_data, frame_time_range);
    //     // IMU回転量を取得
    //     let imu_delta_rot = pose_prediction.delta_rotation.cast::<f32>();
    //     let (roll, pitch, yaw) = imu_delta_rot.euler_angles();
    //     let imu_rot_norm = (roll * roll + pitch * pitch + yaw * yaw).sqrt();
    //     // 回転が大きいフレームはマップ更新しない
    //     const MAX_ROT_FOR_MAP_UPDATE: f32 = 0.025; // rad（約1.43度）
    //     let allow_map_update = imu_rot_norm < MAX_ROT_FOR_MAP_UPDATE;
    //     // IMU回転をsource_to_targetの初期推定に適用（並進は重力未除去のため使わない）
    //     // source_to_target.rotation = imu_delta_rot * source_to_target.rotation;
    //     source_to_target.rotation = source_to_target.rotation * imu_delta_rot.inverse();
    //     log::debug!("Frame {}: IMU rot_norm={:.4} rad, allow_map_update={}", i, imu_rot_norm, allow_map_update);
    //     // --- Predict pose by IMU ---

    //     // --- Deskew source pcd ---
    //     let deskewed_points = deskew_points(&converted_imu_data, &source_pcd);
    //     // --- Deskew source pcd ---

    //     // --- Downsample the deskewed point cloud ---
    //     let downsampled_points = voxel_downsample_points(&deskewed_points, voxel_size);
    //     // --- Downsample the deskewed point cloud ---

    //     // --- Compute covariance matrices ---
    //     let source_voxel_map = build_gicp_voxel_map(
    //         &downsampled_points,
    //         voxel_size,
    //         max_points_per_voxel, // max points per voxel
    //         k_neighbors,          // k neighbors for covariance estimation
    //     );
    //     // --- Compute covariance matrices ---

    //     let last_good_pose = source_to_target;

    //     for i in 0..GICP_ITERATIONS {
    //         // --- find nearest points ---
    //         let search_range = 3; // 3x3x3 voxels
    //         let max_dist_sq = Some(1.0_f32);
    //         let correspondences = find_gicp_correspondences(
    //             &source_voxel_map,
    //             &target_voxel_map,
    //             voxel_size,
    //             &source_to_target,
    //             search_range,
    //             max_dist_sq,
    //         );

    //         if correspondences.is_empty() {
    //             log::warn!("Frame {}: No correspondences found at iteration {}, skipping frame", i, i);
    //             break;
    //         }

    //         let mut dist = 0.0;
    //         for c in &correspondences {
    //             dist += c.dist_sq;
    //         }
    //         log::debug!(
    //             "Average distance of correspondences: {}",
    //             dist / correspondences.len() as f32
    //         );
    //         // --- find nearest points ---

    //         // --- compute GICP ---
    //         let rotate_source_covariance = true;
    //         // 壁接線方向のc_sum固有値は約2.0なので、それに対して有効な正則化を加える
    //         let covariance_regularization = 1.0e-3;

    //         let system = compute_gicp_linear_system(
    //             &correspondences,
    //             &source_to_target,
    //             max_dist_sq.unwrap(),
    //             rotate_source_covariance,
    //             covariance_regularization,
    //         );

    //         // H_ttの最小固有値 ≈ 0.5（壁接線方向）。1e-6では無効 → 0.1で抑制
    //         let damping = 0.1;

    //         // Update source_to_target for the next iteration
    //         if let Some(delta) = solve_gicp_delta(&system, damping) {
    //             // delta[0..3] = 回転（Lie代数ベクトル）, delta[3..6] = 並進
    //             // 発散防止：deltaが大きすぎる場合はスケールダウン
    //             let max_rot_norm = 0.1_f32;   // rad
    //             let max_trans_norm = 0.5_f32; // m
    //             let rot_vec = nalgebra::Vector3::new(delta[0], delta[1], delta[2]);
    //             let trans_vec = nalgebra::Vector3::new(delta[3], delta[4], delta[5]);
    //             let rot_vec = if rot_vec.norm() > max_rot_norm {
    //                 rot_vec.normalize() * max_rot_norm
    //             } else {
    //                 rot_vec
    //             };
    //             let trans_vec = if trans_vec.norm() > max_trans_norm {
    //                 trans_vec.normalize() * max_trans_norm
    //             } else {
    //                 trans_vec
    //             };

    //             let angle = rot_vec.norm();
    //             let rotation = if angle < 1.0e-10 {
    //                 UnitQuaternion::identity()
    //             } else {
    //                 UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(rot_vec), angle)
    //             };
    //             let translation = Translation3::new(trans_vec.x, trans_vec.y, trans_vec.z);
    //             let delta_isometry = Isometry3::from_parts(translation, rotation);
    //             source_to_target = delta_isometry * source_to_target;

    //             let rot_norm = rot_vec.norm();
    //             let trans_norm = trans_vec.norm();
    //             log::debug!("GICP iter {}: rot={:.4} rad, trans={:.4} m", i, rot_norm, trans_norm);
    //             if rot_norm < 1.0e-4 && trans_norm < 1.0e-4 {
    //                 log::debug!("GICP converged at iteration {}", i);
    //                 break;
    //             }
    //         } else {
    //             log::warn!(
    //                 "GICP failed to solve (insufficient correspondences) at iteration {}",
    //                 i
    //             );
    //             break;
    //         }
    //         // --- compute GICP ---
    //     }

    //     // 発散チェック：1フレームで1m以上動いたら棄却
    //     let translation_diff = (source_to_target.translation.vector - last_good_pose.translation.vector).norm();
    //     let (r, p, y) = source_to_target.rotation.euler_angles();
    //     log::info!(
    //         "Frame {:>4}: trans_diff={:.4}m  pose=({:.3},{:.3},{:.3})  rot_rpy=({:.3},{:.3},{:.3})  imu_rot={:.4}  map_update={}",
    //         i,
    //         translation_diff,
    //         source_to_target.translation.x,
    //         source_to_target.translation.y,
    //         source_to_target.translation.z,
    //         r, p, y,
    //         imu_rot_norm,
    //         if translation_diff > 1.0 { "DIVERGED" }
    //         else if !allow_map_update { "SKIP(rot)" }
    //         else { "OK" }
    //     );

    //     if translation_diff > 1.0 {
    //         log::warn!("Frame {}: GICP diverged ({:.4}m), reverting pose and skipping map update", i, translation_diff);
    //         source_to_target = last_good_pose;
    //     } else if !allow_map_update {
    //         log::debug!("Frame {}: skipping map update due to large IMU rotation ({:.4} rad)", i, imu_rot_norm);
    //     } else {
    //         // --- Update target_voxel_map for the next frame ---
    //         merge_points_into_voxel_map(
    //             &mut target_voxel_map,
    //             &downsampled_points,
    //             &source_to_target,
    //             voxel_size,
    //             max_points_per_voxel,
    //             k_neighbors,
    //         );
    //         log::debug!("Frame {}: Map updated. pose = {:?}", i, source_to_target.translation);
    //         //  --- Update target_voxel_map for the next frame ---
    //     }

    //     prev_frame_time = source_pcd_start_time; // 次フレームのIMU積分の開始時刻を更新
    // }

    // // --- Debug: Save final voxel map as PCD ---
    // let final_pcd = convert_voxel_map_to_pcd(&target_voxel_map);
    // let save_file_path = format!("{}/final_voxel_map_gicp_v-{}.pcd", SAVE_DIR, voxel_size);
    // save_pcd_xyzcov(&final_pcd, &save_file_path)?;
    // log::debug!("Saved final voxel map as PCD: {}", save_file_path);
    // // --- Debug: Save final voxel map as PCD ---

    // // let downsampled_pcd = convert_xyz_to_pcd(&downsampled_points);

    // // let save_file_path = format!(
    // //     "{}/downsampled_{}_{}.pcd",
    // //     SAVE_DIR,
    // //     voxel_size,
    // //     pcd_files[0].file_stem().unwrap().to_string_lossy()
    // // );
    // // save_pcd_xyzit(&downsampled_pcd, &save_file_path)?;
    // // --- Downsample the point cloud ---

    // // --- Deskew the point cloud ---
    // // let imu_data = convert_imu_data(&imu_data);
    // // let deskewed_points = deskew_points(&imu_data, &pcd);

    // // let deskewed_pcd = convert_xyz_to_pcd(&deskewed_points);

    // // let save_file_path = format!(
    // //     "{}/deskewed_{}.pcd",
    // //     SAVE_DIR,
    // //     pcd_files[0].file_stem().unwrap().to_string_lossy()
    // // );
    // // save_pcd_xyzit(&deskewed_pcd, &save_file_path)?;
    // // --- Deskew the point cloud ---

    // // --- Compute covariance matrices ---
    // // let start = std::time::Instant::now();

    // // let voxel_size = 0.1;
    // // let max_points_per_voxel = 15;
    // // let k_neighbors = 10;
    // // let source_voxel_map = build_gicp_voxel_map(
    // //     &downsampled_init_points,
    // //     voxel_size,
    // //     max_points_per_voxel,
    // //     k_neighbors,
    // // );

    // // let duration = start.elapsed();
    // // log::debug!("Covariance computation took: {:?}", duration);

    // // let pcd = convert_voxel_map_to_pcd(&source_voxel_map);
    // // let save_file_path = format!(
    // //     "{}/voxel_covariances_{}.pcd",
    // //     SAVE_DIR,
    // //     pcd_files[0].file_stem().unwrap().to_string_lossy()
    // // );
    // // save_pcd_xyzcov(&pcd, &save_file_path)?;

    // // // --- Compute covariance matrices ---

    // // // --- find nearest points ---
    // // let debug_voxel_map = source_voxel_map.clone();
    // // let search_range = 3; // Range of 7x7x7 voxels
    // // let max_dist_sq = Some(1.0); // Optional maximum distance squared
    // // let source_to_target =
    // //     Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());

    // // let correspondences = find_gicp_correspondences(
    // //     &source_voxel_map,
    // //     &debug_voxel_map,
    // //     voxel_size,
    // //     &source_to_target,
    // //     search_range,
    // //     max_dist_sq,
    // // );

    // // let mut dist = 0.0;
    // // for c in &correspondences {
    // //     dist += c.dist_sq;
    // // }
    // // log::debug!(
    // //     "Average distance of correspondences: {}",
    // //     dist / correspondences.len() as f32
    // // );

    // // log::debug!("Found {} correspondences", correspondences.len());

    // // // --- find nearest points ---

    // // // --- compute GICP ---
    // // let rotate_source_covariance = true;
    // // let covariance_regularization = 1.0e-6;

    // // let system = compute_gicp_linear_system(
    // //     &correspondences,
    // //     &source_to_target,
    // //     max_dist_sq.unwrap(),
    // //     rotate_source_covariance,
    // //     covariance_regularization,
    // // );

    // // println!("used correspondences: {}", system.used_count);
    // // println!("cost: {}", system.cost);

    // // let damping = 1.0e-6;

    // // if let Some(delta) = solve_gicp_delta(&system, damping) {
    // //     println!("GICP delta (tx, ty, tz, rx, ry, rz):");
    // //     println!(
    // //         "  translation: ({:.6}, {:.6}, {:.6})",
    // //         delta[0], delta[1], delta[2]
    // //     );
    // //     println!(
    // //         "  rotation:    ({:.6}, {:.6}, {:.6})",
    // //         delta[3], delta[4], delta[5]
    // //     );

    // //     let translation = Translation3::new(delta[0], delta[1], delta[2]);
    // //     let rotation = UnitQuaternion::from_euler_angles(delta[3], delta[4], delta[5]);
    // //     let delta_isometry = Isometry3::from_parts(translation, rotation);
    // //     let updated_pose = delta_isometry * source_to_target;
    // //     println!("Updated pose:");
    // //     println!("  translation: {:?}", updated_pose.translation);
    // //     println!("  rotation:    {:?}", updated_pose.rotation);
    // // } else {
    // //     println!("GICP failed to solve (insufficient correspondences)");
    // // }
    // // --- compute GICP ---

    Ok(())
}
