use nalgebra::{Matrix3, Point3, SymmetricEigen, Vector3};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;

const EPSILON: f32 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelKey {
    pub ix: i32,
    pub iy: i32,
    pub iz: i32,
}

#[derive(Debug, Clone)]
pub struct VoxelCell {
    /// このvoxel内に入った点群
    pub points: Vec<Point3<f32>>,

    /// voxel内点群の平均位置
    /// registration時には、このmeanをvoxel代表点として使える
    pub mean: Point3<f32>,

    /// 実際の点群分布から計算した共分散
    pub raw_covariance: Matrix3<f32>,

    /// GICP用に正則化した共分散
    pub gicp_covariance: Matrix3<f32>,

    /// 十分な近傍点から共分散を計算できたか
    pub covariance_valid: bool,
}

impl VoxelCell {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            mean: Point3::new(0.0, 0.0, 0.0),
            raw_covariance: Matrix3::identity(),
            gicp_covariance: Matrix3::identity(),
            covariance_valid: false,
        }
    }

    pub fn recompute_mean(&mut self) {
        if self.points.is_empty() {
            self.mean = Point3::new(0.0, 0.0, 0.0);
            return;
        }

        let mut sum = Vector3::zeros();

        for p in &self.points {
            sum += p.coords;
        }

        let mean = sum / self.points.len() as f32;
        self.mean = Point3::from(mean);
    }
}

pub type VoxelMap = HashMap<VoxelKey, VoxelCell>;

#[inline]
pub fn voxel_key(p: &Point3<f32>, voxel_size: f32) -> VoxelKey {
    VoxelKey {
        ix: (p.x / voxel_size).floor() as i32,
        iy: (p.y / voxel_size).floor() as i32,
        iz: (p.z / voxel_size).floor() as i32,
    }
}

pub fn build_voxel_map(
    points: &[Point3<f32>],
    voxel_size: f32,
    max_points_per_voxel: usize,
) -> VoxelMap {
    let mut voxel_map = VoxelMap::new();

    for p in points {
        let key = voxel_key(p, voxel_size);

        let cell = voxel_map.entry(key).or_insert_with(VoxelCell::new);

        // mapが無限に肥大化しないように上限を設ける
        if cell.points.len() < max_points_per_voxel {
            cell.points.push(*p);
        }
    }

    for cell in voxel_map.values_mut() {
        cell.recompute_mean();
    }

    voxel_map
}

fn collect_nearest_points_3x3x3(
    voxel_map: &VoxelMap,
    base_key: VoxelKey,
    query: &Point3<f32>,
    k: usize,
) -> Vec<Point3<f32>> {
    let mut candidates: Vec<(f32, Point3<f32>)> = Vec::new();

    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let key = VoxelKey {
                    ix: base_key.ix + dx,
                    iy: base_key.iy + dy,
                    iz: base_key.iz + dz,
                };

                if let Some(cell) = voxel_map.get(&key) {
                    for p in &cell.points {
                        let diff = p.coords - query.coords;
                        let dist2 = diff.dot(&diff);
                        candidates.push((dist2, *p));
                    }
                }
            }
        }
    }

    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    candidates.into_iter().take(k).map(|(_, p)| p).collect()
}

fn compute_raw_covariance_from_points(points: &[Point3<f32>]) -> Option<Matrix3<f32>> {
    if points.len() < 3 {
        return None;
    }

    let n = points.len() as f32;

    let mut mean = Vector3::zeros();

    for p in points {
        mean += p.coords;
    }

    mean /= n;

    let mut cov = Matrix3::<f32>::zeros();

    for p in points {
        let d = p.coords - mean;
        cov += d * d.transpose();
    }

    cov /= n;

    Some(cov)
}

fn regularize_gicp_covariance(cov: Matrix3<f32>) -> Matrix3<f32> {
    let eig = SymmetricEigen::new(cov);

    let mut min_idx = 0;

    if eig.eigenvalues[1] < eig.eigenvalues[min_idx] {
        min_idx = 1;
    }

    if eig.eigenvalues[2] < eig.eigenvalues[min_idx] {
        min_idx = 2;
    }

    let normal = eig.eigenvectors.column(min_idx).into_owned();

    Matrix3::<f32>::identity() + (EPSILON - 1.0) * (normal * normal.transpose())
}

pub fn compute_voxel_covariances_3x3x3(voxel_map: &mut VoxelMap, k_neighbors: usize) {
    let keys: Vec<VoxelKey> = voxel_map.keys().copied().collect();

    let results: Vec<(VoxelKey, Matrix3<f32>, Matrix3<f32>, bool)> = keys
        .par_iter()
        .map(|&key| {
            let cell = voxel_map.get(&key).expect("voxel key should exist");

            let neighbor_points =
                collect_nearest_points_3x3x3(voxel_map, key, &cell.mean, k_neighbors);

            match compute_raw_covariance_from_points(&neighbor_points) {
                Some(raw_cov) => {
                    let gicp_cov = regularize_gicp_covariance(raw_cov);
                    (key, raw_cov, gicp_cov, true)
                }
                None => (
                    key,
                    Matrix3::<f32>::identity(),
                    Matrix3::<f32>::identity(),
                    false,
                ),
            }
        })
        .collect();

    for (key, raw_cov, gicp_cov, valid) in results {
        if let Some(cell) = voxel_map.get_mut(&key) {
            cell.raw_covariance = raw_cov;
            cell.gicp_covariance = gicp_cov;
            cell.covariance_valid = valid;
        }
    }
}

pub fn build_gicp_voxel_map(
    points: &[Point3<f32>],
    voxel_size: f32,
    max_points_per_voxel: usize,
    k_neighbors: usize,
) -> VoxelMap {
    let mut voxel_map = build_voxel_map(points, voxel_size, max_points_per_voxel);

    compute_voxel_covariances_3x3x3(&mut voxel_map, k_neighbors);

    voxel_map
}

/// 変換済み点群を既存のVoxelMapに追加し、変更されたvoxelの共分散を再計算する。
/// `pose` で変換してからマップに登録する（SLAMマップ更新用）。
pub fn merge_points_into_voxel_map(
    map: &mut VoxelMap,
    points: &[Point3<f32>],
    pose: &nalgebra::Isometry3<f32>,
    voxel_size: f32,
    max_points_per_voxel: usize,
    k_neighbors: usize,
) {
    let mut modified_keys: std::collections::HashSet<VoxelKey> = std::collections::HashSet::new();

    for p in points {
        let transformed = pose.transform_point(p);
        let key = voxel_key(&transformed, voxel_size);

        let cell = map.entry(key).or_insert_with(VoxelCell::new);

        if cell.points.len() < max_points_per_voxel {
            cell.points.push(transformed);
            cell.recompute_mean();
            modified_keys.insert(key);
        }
    }

    // 変更されたvoxelとその隣接voxelの共分散を再計算
    let keys_to_recompute: Vec<VoxelKey> = modified_keys
        .iter()
        .flat_map(|k| {
            (-1..=1).flat_map(move |dx| {
                (-1..=1).flat_map(move |dy| {
                    (-1..=1).map(move |dz| VoxelKey {
                        ix: k.ix + dx,
                        iy: k.iy + dy,
                        iz: k.iz + dz,
                    })
                })
            })
        })
        .filter(|k| map.contains_key(k))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let results: Vec<(VoxelKey, Matrix3<f32>, Matrix3<f32>, bool)> = keys_to_recompute
        .par_iter()
        .map(|&key| {
            let cell = map.get(&key).expect("voxel key should exist");
            let neighbor_points = collect_nearest_points_3x3x3(map, key, &cell.mean, k_neighbors);
            match compute_raw_covariance_from_points(&neighbor_points) {
                Some(raw_cov) => {
                    let gicp_cov = regularize_gicp_covariance(raw_cov);
                    (key, raw_cov, gicp_cov, true)
                }
                None => (key, Matrix3::identity(), Matrix3::identity(), false),
            }
        })
        .collect();

    for (key, raw_cov, gicp_cov, valid) in results {
        if let Some(cell) = map.get_mut(&key) {
            cell.raw_covariance = raw_cov;
            cell.gicp_covariance = gicp_cov;
            cell.covariance_valid = valid;
        }
    }
}
