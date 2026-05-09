use nalgebra::{Isometry3, Matrix3, Point3};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::compute_covariance::{VoxelKey, VoxelMap, voxel_key};

#[derive(Debug, Clone)]
pub struct GicpCorrespondence {
    pub source_key: VoxelKey,
    pub target_key: VoxelKey,

    /// source側の元のvoxel代表点
    pub source_mean: Point3<f32>,

    /// 現在姿勢でtarget座標系へ変換したsource代表点
    pub transformed_source_mean: Point3<f32>,

    /// 最近傍target voxelの代表点
    pub target_mean: Point3<f32>,

    /// source側GICP共分散
    pub source_covariance: Matrix3<f32>,

    /// target側GICP共分散
    pub target_covariance: Matrix3<f32>,

    /// 最近傍距離の二乗
    pub dist_sq: f32,
}

fn find_nearest_target_voxel(
    query_point: &Point3<f32>,
    target_map: &VoxelMap,
    voxel_size: f32,
    search_range: i32,
    max_dist_sq: Option<f32>,
) -> Option<(VoxelKey, Point3<f32>, Matrix3<f32>, f32)> {
    let base_key = voxel_key(query_point, voxel_size);

    let mut best_key: Option<VoxelKey> = None;
    let mut best_mean = Point3::new(0.0, 0.0, 0.0);
    let mut best_cov = Matrix3::<f32>::identity();
    let mut best_dist_sq = max_dist_sq.unwrap_or(f32::INFINITY);

    for dz in -search_range..=search_range {
        for dy in -search_range..=search_range {
            for dx in -search_range..=search_range {
                let key = VoxelKey {
                    ix: base_key.ix + dx,
                    iy: base_key.iy + dy,
                    iz: base_key.iz + dz,
                };

                let Some(target_cell) = target_map.get(&key) else {
                    continue;
                };

                if !target_cell.covariance_valid {
                    continue;
                }

                let diff = query_point.coords - target_cell.mean.coords;
                let dist_sq = diff.dot(&diff);

                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best_key = Some(key);
                    best_mean = target_cell.mean;
                    best_cov = target_cell.gicp_covariance;
                }
            }
        }
    }

    best_key.map(|key| (key, best_mean, best_cov, best_dist_sq))
}

pub fn find_gicp_correspondences(
    source_map: &VoxelMap,
    target_map: &VoxelMap,
    voxel_size: f32,
    source_to_target: &Isometry3<f32>,
    search_range: i32,
    max_dist_sq: Option<f32>,
) -> Vec<GicpCorrespondence> {
    source_map
        .par_iter()
        .filter_map(|(&source_key, source_cell)| {
            if !source_cell.covariance_valid {
                return None;
            }

            let transformed_source_mean = source_to_target.transform_point(&source_cell.mean);

            let (target_key, target_mean, target_covariance, dist_sq) = find_nearest_target_voxel(
                &transformed_source_mean,
                target_map,
                voxel_size,
                search_range,
                max_dist_sq,
            )?;

            Some(GicpCorrespondence {
                source_key,
                target_key,
                source_mean: source_cell.mean,
                transformed_source_mean,
                target_mean,
                source_covariance: source_cell.gicp_covariance,
                target_covariance,
                dist_sq,
            })
        })
        .collect()
}
