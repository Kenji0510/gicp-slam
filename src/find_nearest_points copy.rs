use nalgebra::{Isometry3, Matrix3, Point3};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::compute_covariance::{VoxelKey, VoxelMap, add_diagonal, invert_matrix3_safe, voxel_key};

#[derive(Debug, Clone)]
pub struct GaussianCorrespondence {
    pub source_key: VoxelKey,
    pub target_key: VoxelKey,

    /// source側Gaussianの平均。
    pub source_mean: Point3<f32>,

    /// 現在姿勢でtarget/map座標系へ変換したsource平均。
    pub transformed_source_mean: Point3<f32>,

    /// target/map側Gaussianの平均。
    pub target_mean: Point3<f32>,

    /// source側Gaussian共分散。source座標系のΣ_s。
    pub source_covariance: Matrix3<f32>,

    /// target/map側Gaussian共分散。target座標系のΣ_t。
    pub target_covariance: Matrix3<f32>,

    /// mean同士のユークリッド距離二乗。外れ値gateやログ用。
    pub euclidean_dist_sq: f32,

    /// e^T (Σ_t + RΣ_sR^T + λI)^-1 e。
    pub mahalanobis_dist: f32,
}

fn find_nearest_target_gaussian(
    query_point: &Point3<f32>,
    rotated_source_covariance: &Matrix3<f32>,
    target_map: &VoxelMap,
    gaussian_voxel_size: f32,
    search_range: i32,
    max_euclidean_dist_sq: Option<f32>,
    max_mahalanobis_dist: Option<f32>,
    covariance_regularization: f32,
) -> Option<(VoxelKey, Point3<f32>, Matrix3<f32>, f32, f32)> {
    let base_key = voxel_key(query_point, gaussian_voxel_size);

    let mut best_key: Option<VoxelKey> = None;
    let mut best_mean = Point3::new(0.0, 0.0, 0.0);
    let mut best_covariance = Matrix3::<f32>::identity();
    let mut best_euclidean_dist_sq = f32::INFINITY;
    let mut best_mahalanobis_dist = max_mahalanobis_dist.unwrap_or(f32::INFINITY);

    let euclidean_gate = max_euclidean_dist_sq.unwrap_or(f32::INFINITY);

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

                if !target_cell.valid {
                    continue;
                }

                let diff = query_point.coords - target_cell.mean.coords;
                let euclidean_dist_sq = diff.dot(&diff);

                if euclidean_dist_sq > euclidean_gate {
                    continue;
                }

                let c_sum = add_diagonal(
                    target_cell.covariance + rotated_source_covariance,
                    covariance_regularization,
                );

                let Some(omega) = invert_matrix3_safe(c_sum) else {
                    continue;
                };

                let mahalanobis_dist = diff.dot(&(omega * diff));
                if !mahalanobis_dist.is_finite() {
                    continue;
                }

                if mahalanobis_dist < best_mahalanobis_dist {
                    best_mahalanobis_dist = mahalanobis_dist;
                    best_euclidean_dist_sq = euclidean_dist_sq;
                    best_key = Some(key);
                    best_mean = target_cell.mean;
                    best_covariance = target_cell.covariance;
                }
            }
        }
    }

    best_key.map(|key| {
        (
            key,
            best_mean,
            best_covariance,
            best_euclidean_dist_sq,
            best_mahalanobis_dist,
        )
    })
}

pub fn find_gaussian_correspondences(
    source_map: &VoxelMap,
    target_map: &VoxelMap,
    gaussian_voxel_size: f32,
    source_to_target: &Isometry3<f32>,
    search_range: i32,
    max_euclidean_dist_sq: Option<f32>,
    max_mahalanobis_dist: Option<f32>,
    covariance_regularization: f32,
) -> Vec<GaussianCorrespondence> {
    let r_mat = source_to_target
        .rotation
        .to_rotation_matrix()
        .matrix()
        .clone_owned();

    source_map
        .par_iter()
        .filter_map(|(&source_key, source_cell)| {
            if !source_cell.valid {
                return None;
            }

            let transformed_source_mean = source_to_target.transform_point(&source_cell.mean);
            let rotated_source_covariance = r_mat * source_cell.covariance * r_mat.transpose();

            let (target_key, target_mean, target_covariance, euclidean_dist_sq, mahalanobis_dist) =
                find_nearest_target_gaussian(
                    &transformed_source_mean,
                    &rotated_source_covariance,
                    target_map,
                    gaussian_voxel_size,
                    search_range,
                    max_euclidean_dist_sq,
                    max_mahalanobis_dist,
                    covariance_regularization,
                )?;

            Some(GaussianCorrespondence {
                source_key,
                target_key,
                source_mean: source_cell.mean,
                transformed_source_mean,
                target_mean,
                source_covariance: source_cell.covariance,
                target_covariance,
                euclidean_dist_sq,
                mahalanobis_dist,
            })
        })
        .collect()
}
