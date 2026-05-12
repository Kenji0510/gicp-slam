use anyhow::Result;
use lidar_slam::{
    compute_covariance::{build_gaussian_voxel_map, merge_points_into_gaussian_voxel_map},
    convert_imu_data::convert_imu_data,
    convert_type::{convert_pcd_to_xyz, convert_xyz_to_pcd},
    debug::convert_voxel_map_to_pcd,
    deskew_points::deskew_points,
    file_handler::{
        load_imu_data, load_pcd_files, load_pcd_xyzit, save_pcd_xyzcov, save_pcd_xyzit,
    },
    find_nearest_points::find_gaussian_correspondences,
    gicp_gaussian_shape::{
        GaussianShapeOptions, compute_gaussian_shape_linear_system, solve_gaussian_delta,
    },
    predict_pose_by_imu::{align_imu_timestamps, predict_pose_by_imu},
    tilt_correction::{apply_tilt_correction, compute_tilt_correction, estimate_gravity_from_imu},
    voxelization::voxel_downsample_points,
};
use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};

const LOAD_DIR: &str = "/home/kenji/workspace/rust/get_lidar_data/data/output/05092026/park05";
const SAVE_DIR: &str = "data/output/debug/05112026";

const GAUSSIAN_ITERATIONS: usize = 5;

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

    // IMU coord to LiDAR coord transformation
    let imu_to_lidar = UnitQuaternion::new_normalize(Quaternion::new(
        IMU_TO_LIDAR_QUAT_W,
        IMU_TO_LIDAR_QUAT_X,
        IMU_TO_LIDAR_QUAT_Y,
        IMU_TO_LIDAR_QUAT_Z,
    ));

    let pcd = load_pcd_xyzit(&pcd_files[0].to_string_lossy())?;
    let points = convert_pcd_to_xyz(&pcd);

    // --- Downsample for density normalization ---
    let downsample_voxel_size = 0.1_f32;
    let gaussian_voxel_size = 0.4_f32;

    let downsampled_init_points = voxel_downsample_points(&points, downsample_voxel_size);
    // --- Downsample for density normalization ---

    let converted_imu_data = convert_imu_data(&imu_data);
    let pcd_start_time = pcd
        .iter()
        .map(|p| p.timestamp)
        .fold(f64::INFINITY, f64::min);

    // --- Build initial Gaussian voxel map ---
    let max_points_per_gaussian = 150;
    let min_points_per_gaussian = 5;
    // min_variance: 小さすぎると平面法線方向の固有値が全voxelで揃い共分散が均一に見える
    // voxel_size=0.4m での自然な下限は ~1e-2（≈10cm²）程度
    let covariance_min_variance = 1.0e-2_f32;
    // max_variance: voxel対角線(0.4*√3≈0.69m)の半分の二乗≈0.12。余裕を持たせ0.5
    let covariance_max_variance = 0.5_f32;
    let covariance_regularization = 1.0e-3_f32;

    let mut target_voxel_map = build_gaussian_voxel_map(
        &downsampled_init_points,
        gaussian_voxel_size,
        max_points_per_gaussian,
        min_points_per_gaussian,
        covariance_min_variance,
        covariance_max_variance,
        covariance_regularization,
    );
    // --- Build initial Gaussian voxel map ---
    let mut source_to_target =
        Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());

    let mut prev_frame_time = pcd_start_time; // フレーム間のIMU積分用
    let mut pose_log: Vec<serde_json::Value> = Vec::new();
    let mut consecutive_skip_count: usize = 0;
    // マップが陳腐化しないよう、連続スキップ上限を設ける
    const MAX_CONSECUTIVE_SKIPS: usize = 10;

    for (i, pcd_path) in pcd_files.iter().enumerate().skip(1) {
        let source_pcd = load_pcd_xyzit(&pcd_path.to_string_lossy())?;

        // --- Predict pose by IMU ---
        let source_pcd_start_time = source_pcd
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::INFINITY, f64::min);

        let frame_time_range = (prev_frame_time, source_pcd_start_time);
        let pose_prediction = predict_pose_by_imu(&imu_data, frame_time_range, &imu_to_lidar, None);
        // --- Predict pose by IMU ---

        // --- Deskew source pcd ---
        let deskewed_points = deskew_points(&converted_imu_data, &imu_to_lidar, &source_pcd);
        // --- Deskew source pcd ---

        // --- Downsample the deskewed point cloud ---
        let downsampled_points = voxel_downsample_points(&deskewed_points, downsample_voxel_size);
        // --- Downsample the deskewed point cloud ---

        // // --- Tilt correction ---
        // let frame_gravity = estimate_gravity_from_imu(&imu_data, source_pcd_start_time, 2.0);
        // let tilt_correction = compute_tilt_correction(&frame_gravity);
        // let downsampled_points = apply_tilt_correction(&downsampled_points, &tilt_correction);
        // log::debug!(
        //     "Frame {:>4}: tilt gravity=[{:.3},{:.3},{:.3}]",
        //     i,
        //     frame_gravity.x,
        //     frame_gravity.y,
        //     frame_gravity.z
        // );
        // // --- Tilt correction ---

        // --- Build source Gaussian voxel map ---
        let source_voxel_map = build_gaussian_voxel_map(
            &downsampled_points,
            gaussian_voxel_size,
            max_points_per_gaussian,
            min_points_per_gaussian,
            covariance_min_variance,
            covariance_max_variance,
            covariance_regularization,
        );
        // --- Build source Gaussian voxel map ---

        let last_good_pose = source_to_target;
        let mut last_num_correspondences = 0usize;

        for i in 0..GAUSSIAN_ITERATIONS {
            // --- find nearest points ---
            let search_range = 3; // 3x3x3 voxels
            let max_euclidean_dist_sq = Some(1.0_f32);
            let max_mahalanobis_dist = Some(25.0_f32);
            let correspondences = find_gaussian_correspondences(
                &source_voxel_map,
                &target_voxel_map,
                gaussian_voxel_size,
                &source_to_target,
                search_range,
                max_euclidean_dist_sq,
                max_mahalanobis_dist,
                covariance_regularization,
            );
            last_num_correspondences = correspondences.len();

            if correspondences.is_empty() {
                log::warn!(
                    "Frame {}: No correspondences found at iteration {}, skipping frame",
                    i,
                    i
                );
                break;
            }

            let mut dist = 0.0;
            for c in &correspondences {
                dist += c.euclidean_dist_sq;
            }
            log::debug!(
                "Average euclidean distance of correspondences: {}",
                dist / correspondences.len() as f32
            );
            // --- find nearest points ---

            // --- compute Gaussian shape matching ---
            // mean-to-mean項 + covariance shape-to-shape項。
            let shape_options = GaussianShapeOptions {
                mean_weight: 1.0,
                shape_weight: 0.05,
                covariance_regularization,
                max_euclidean_dist_sq: max_euclidean_dist_sq.unwrap(),
                max_mahalanobis_dist,
                shape_fd_epsilon: 1.0e-3,
                normalize_shape_by_trace: true,
                min_trace: 1.0e-6,
            };

            let system = compute_gaussian_shape_linear_system(
                &correspondences,
                &source_to_target,
                &shape_options,
            );

            // H_ttの最小固有値 ≈ 0.5（壁接線方向）。1e-6では無効 → 0.1で抑制
            let damping = 0.1;

            // Update source_to_target for the next iteration
            if let Some(delta) = solve_gaussian_delta(&system, damping) {
                // delta[0..3] = 回転（Lie代数ベクトル）, delta[3..6] = 並進
                // 発散防止：deltaが大きすぎる場合はスケールダウン
                let max_rot_norm = 0.1_f32; // rad
                let max_trans_norm = 0.5_f32; // m
                let rot_vec = nalgebra::Vector3::new(delta[0], delta[1], delta[2]);
                let trans_vec = nalgebra::Vector3::new(delta[3], delta[4], delta[5]);
                let rot_vec = if rot_vec.norm() > max_rot_norm {
                    rot_vec.normalize() * max_rot_norm
                } else {
                    rot_vec
                };
                let trans_vec = if trans_vec.norm() > max_trans_norm {
                    trans_vec.normalize() * max_trans_norm
                } else {
                    trans_vec
                };

                let angle = rot_vec.norm();
                let rotation = if angle < 1.0e-10 {
                    UnitQuaternion::identity()
                } else {
                    UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(rot_vec), angle)
                };
                let translation = Translation3::new(trans_vec.x, trans_vec.y, trans_vec.z);
                let delta_isometry = Isometry3::from_parts(translation, rotation);
                source_to_target = delta_isometry * source_to_target;

                let rot_norm = rot_vec.norm();
                let trans_norm = trans_vec.norm();
                log::debug!(
                    "Gaussian iter {}: rot={:.4} rad, trans={:.4} m",
                    i,
                    rot_norm,
                    trans_norm
                );
                if rot_norm < 1.0e-4 && trans_norm < 1.0e-4 {
                    log::debug!("Gaussian converged at iteration {}", i);
                    break;
                }
            } else {
                log::warn!(
                    "Gaussian failed to solve (insufficient correspondences) at iteration {}",
                    i
                );
                break;
            }
            // --- compute GICP ---
        }

        // 発散チェック：1フレームで1m以上動いたら棄却
        let translation_diff =
            (source_to_target.translation.vector - last_good_pose.translation.vector).norm();
        let proposed_tx = source_to_target.translation.x;
        let proposed_ty = source_to_target.translation.y;
        let proposed_tz = source_to_target.translation.z;
        let (r, p, y) = source_to_target.rotation.euler_angles();
        log::info!(
            "Frame {:>4}: trans_diff={:.4}m  pose=({:.3},{:.3},{:.3})  rot_rpy=({:.3},{:.3},{:.3})",
            i,
            translation_diff,
            source_to_target.translation.x,
            source_to_target.translation.y,
            source_to_target.translation.z,
            r,
            p,
            y,
        );

        if translation_diff > 1.0 {
            log::warn!(
                "Frame {}: Gaussian diverged ({:.4}m), reverting pose and skipping map update",
                i,
                translation_diff
            );
            source_to_target = last_good_pose;
            consecutive_skip_count += 1;
        } else {
            // --- Update target_voxel_map for the next frame ---
            merge_points_into_gaussian_voxel_map(
                &mut target_voxel_map,
                &downsampled_points,
                &source_to_target,
                gaussian_voxel_size,
                max_points_per_gaussian,
                min_points_per_gaussian,
                covariance_min_variance,
                covariance_max_variance,
                covariance_regularization,
            );
            if consecutive_skip_count >= MAX_CONSECUTIVE_SKIPS {
                log::warn!(
                    "Frame {}: forced map update after {} consecutive skips",
                    i,
                    consecutive_skip_count
                );
            }
            consecutive_skip_count = 0;
            log::debug!(
                "Frame {}: Map updated. pose = {:?}",
                i,
                source_to_target.translation
            );
            //  --- Update target_voxel_map for the next frame ---
        }

        pose_log.push(serde_json::json!({
            "frame":              i,
            "timestamp":          source_pcd_start_time,
            "proposed_tx":        proposed_tx,
            "proposed_ty":        proposed_ty,
            "proposed_tz":        proposed_tz,
            "proposed_roll":      r,
            "proposed_pitch":     p,
            "proposed_yaw":       y,
            "translation_diff":   translation_diff,
            "consecutive_skips":  consecutive_skip_count,
            "num_correspondences":last_num_correspondences,
        }));

        prev_frame_time = source_pcd_start_time; // 次フレームのIMU積分の開始時刻を更新
    }

    // --- Save pose log as JSON ---
    let log_path = format!("{}/pose_log.json", SAVE_DIR);
    std::fs::write(&log_path, serde_json::to_string_pretty(&pose_log)?)?;
    log::info!("Saved pose log ({} frames): {}", pose_log.len(), log_path);
    // --- Save pose log as JSON ---

    // --- Debug: Save final voxel map as PCD ---
    let final_pcd = convert_voxel_map_to_pcd(&target_voxel_map);
    let save_file_path = format!(
        "{}/final_gaussian_map_v-{}.pcd",
        SAVE_DIR, gaussian_voxel_size
    );
    save_pcd_xyzcov(&final_pcd, &save_file_path)?;
    log::info!("Saved final Gaussian voxel map as PCD: {}", save_file_path);
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

    Ok(())
}
