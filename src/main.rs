use anyhow::Result;
use lidar_slam::{
    compute_covariance::build_gicp_voxel_map, convert_imu_data::convert_imu_data, convert_type::{convert_pcd_to_xyz, convert_xyz_to_pcd}, debug::convert_voxel_map_to_pcd, deskew_points::deskew_points, file_handler::{
        load_imu_data, load_pcd_files, load_pcd_xyzit, save_pcd_xyzcov, save_pcd_xyzit,
    }, find_nearest_points::find_gicp_correspondences, gicp::{compute_gicp_linear_system, solve_gicp_delta}, predict_pose_by_imu::align_imu_timestamps, voxelization::voxel_downsample_points
};
use nalgebra::{Isometry3, Translation3, UnitQuaternion};

const LOAD_DIR: &str = "/home/kenji/workspace/rust/get_lidar_data/data/output/05062026/03";
const SAVE_DIR: &str = "data/output/debug";

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

    let pcd = load_pcd_xyzit(&pcd_files[10].to_string_lossy())?;
    let points = convert_pcd_to_xyz(&pcd);

    // --- Downsample the point cloud ---
    let voxel_size = 0.1;
    let downsampled_points = voxel_downsample_points(&points, voxel_size);

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
    let start = std::time::Instant::now();

    let voxel_size = 0.1;
    let max_points_per_voxel = 15;
    let k_neighbors = 10;
    let source_voxel_map = build_gicp_voxel_map(
        &downsampled_points,
        voxel_size,
        max_points_per_voxel,
        k_neighbors,
    );

    let duration = start.elapsed();
    log::debug!("Covariance computation took: {:?}", duration);

    let pcd = convert_voxel_map_to_pcd(&source_voxel_map);
    let save_file_path = format!(
        "{}/voxel_covariances_{}.pcd",
        SAVE_DIR,
        pcd_files[0].file_stem().unwrap().to_string_lossy()
    );
    save_pcd_xyzcov(&pcd, &save_file_path)?;

    // --- Compute covariance matrices ---

    // --- find nearest points ---
    let debug_voxel_map = source_voxel_map.clone();
    let search_range = 3; // Range of 7x7x7 voxels
    let max_dist_sq = Some(1.0); // Optional maximum distance squared
    let source_to_target =
        Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());

    let correspondences = find_gicp_correspondences(
        &source_voxel_map,
        &debug_voxel_map,
        voxel_size,
        &source_to_target,
        search_range,
        max_dist_sq,
    );

    let mut dist = 0.0;
    for c in &correspondences {
        dist += c.dist_sq;
    }
    log::debug!("Average distance of correspondences: {}", dist / correspondences.len() as f32);

    log::debug!("Found {} correspondences", correspondences.len());

    // --- find nearest points ---

    // --- compute GICP ---
    let rotate_source_covariance = true;
    let covariance_regularization = 1.0e-6;

    let system = compute_gicp_linear_system(
        &correspondences,
        &source_to_target,
        max_dist_sq.unwrap(),
        rotate_source_covariance,
        covariance_regularization,
    );

    println!("used correspondences: {}", system.used_count);
    println!("cost: {}", system.cost);

    let damping = 1.0e-6;

    if let Some(delta) = solve_gicp_delta(&system, damping) {
        println!("GICP delta (tx, ty, tz, rx, ry, rz):");
        println!("  translation: ({:.6}, {:.6}, {:.6})", delta[0], delta[1], delta[2]);
        println!("  rotation:    ({:.6}, {:.6}, {:.6})", delta[3], delta[4], delta[5]);

        let translation = Translation3::new(delta[0], delta[1], delta[2]);
        let rotation = UnitQuaternion::from_euler_angles(delta[3], delta[4], delta[5]);
        let delta_isometry = Isometry3::from_parts(translation, rotation);
        let updated_pose = delta_isometry * source_to_target;
        println!("Updated pose:");
        println!("  translation: {:?}", updated_pose.translation);
        println!("  rotation:    {:?}", updated_pose.rotation);
    } else {
        println!("GICP failed to solve (insufficient correspondences)");
    }
    // --- compute GICP ---

    Ok(())
}
