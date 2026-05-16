use nalgebra::{Isometry3, Matrix3, Point3, SymmetricEigen, Vector3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelKey {
    pub ix: i32,
    pub iy: i32,
    pub iz: i32,
}

/// 1つのvoxelを1つの3D Gaussianとして扱うためのcell。
///
/// GICP用の `gicp_covariance` ではなく、Gaussian分布としての
/// `covariance` と、その逆行列 `information` を持つ。
#[derive(Debug, Clone)]
pub struct VoxelCell {
    /// このvoxel内に入った代表点。
    /// 初期段階ではデバッグ性を優先して保持する。
    /// 将来的には Welford の count / mean / m2 のみへ移行可能。
    pub points: Vec<Point3<f32>>,

    /// Gaussianの平均 μ。
    pub mean: Point3<f32>,

    /// voxel内点群から直接計算した共分散。
    pub raw_covariance: Matrix3<f32>,

    /// 固有値clamp後の安定化済みGaussian共分散 Σ。
    pub covariance: Matrix3<f32>,

    /// Gaussianとしてregistrationに使えるだけの点数と数値安定性があるか。
    pub valid: bool,
}

impl VoxelCell {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            mean: Point3::new(0.0, 0.0, 0.0),
            raw_covariance: Matrix3::identity(),
            covariance: Matrix3::identity(),
            valid: false,
        }
    }

    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    pub fn push_point(&mut self, p: Point3<f32>, max_points_per_gaussian: usize) -> bool {
        if self.points.len() >= max_points_per_gaussian {
            return false;
        }

        self.points.push(p);
        true
    }

    pub fn recompute_covariance(&mut self, min_points_per_gaussian: usize) {
        self.valid = false;

        if self.points.is_empty() {
            self.mean = Point3::new(0.0, 0.0, 0.0);
            self.raw_covariance = Matrix3::identity();
            self.covariance = Matrix3::identity();
            self.valid = false;
            return;
        }

        self.mean = compute_mean_from_points(&self.points);

        if self.points.len() < min_points_per_gaussian {
            self.raw_covariance = Matrix3::identity();
            self.covariance = Matrix3::identity();
            self.valid = false;
            return;
        }

        let Some(raw_covariance) = compute_raw_covariance_from_points(&self.points, &self.mean)
        else {
            self.raw_covariance = Matrix3::identity();
            self.covariance = Matrix3::identity();
            self.valid = false;
            return;
        };

        let covariance = regularize_gaussian_covariance(raw_covariance);

        self.raw_covariance = raw_covariance;
        self.covariance = covariance;
        self.valid = true;
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

#[inline]
pub fn is_finite_matrix3(m: &Matrix3<f32>) -> bool {
    m.iter().all(|v| v.is_finite())
}

pub fn invert_matrix3_safe(m: Matrix3<f32>) -> Option<Matrix3<f32>> {
    if !is_finite_matrix3(&m) {
        return None;
    }

    let det = m.determinant();
    if !det.is_finite() || det.abs() < 1.0e-12 {
        return None;
    }

    m.try_inverse()
}

fn compute_mean_from_points(points: &[Point3<f32>]) -> Point3<f32> {
    let mut sum = Vector3::zeros();

    for p in points {
        sum += p.coords;
    }

    Point3::from(sum / points.len() as f32)
}

fn compute_raw_covariance_from_points(
    points: &[Point3<f32>],
    mean: &Point3<f32>,
) -> Option<Matrix3<f32>> {
    if points.len() < 3 {
        return None;
    }

    let mut cov = Matrix3::<f32>::zeros();

    for p in points {
        let d = p.coords - mean.coords;
        cov += d * d.transpose();
    }

    // 標本共分散。voxel内点数が少ない場合に過小評価しにくい。
    cov /= (points.len() - 1) as f32;

    Some(cov)
}

/// Gaussian covarianceとして使うため、固有値を範囲内にclampする。
/// GICPの平面法線方向だけを強くする正則化とは違い、分布形状を残す。
pub fn regularize_gaussian_covariance(cov: Matrix3<f32>) -> Matrix3<f32> {
    let eig = SymmetricEigen::new(cov);
    let mut d = Matrix3::<f32>::zeros();

    let eigen = SymmetricEigen::new(cov);
    let rot = eigen.eigenvectors;
    let mut vals = eigen.eigenvalues;

    let mut pairs: Vec<(f32, usize)> = vals
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let min_idx = pairs[0].1;
    vals[min_idx] = 1e-3; // 法線方向を薄くする
    vals[pairs[1].1] = 1.0;
    vals[pairs[2].1] = 5.0;

    // C = R * S * R^T
    let regularized_cov = rot * Matrix3::from_diagonal(&vals) * rot.transpose();
    regularized_cov
}

pub fn build_gicp_voxel_map(
    points: &[Point3<f32>],
    gaussian_voxel_size: f32,
    max_points_per_gaussian: usize,
    min_points_per_gaussian: usize,
) -> VoxelMap {
    let mut voxel_map = VoxelMap::new();

    for &p in points {
        let key = voxel_key(&p, gaussian_voxel_size);
        let cell = voxel_map.entry(key).or_insert_with(VoxelCell::new);
        cell.push_point(p, max_points_per_gaussian);
    }

    recompute_all_covariance(&mut voxel_map, min_points_per_gaussian);

    voxel_map
}

pub fn recompute_all_covariance(voxel_map: &mut VoxelMap, min_points_per_gaussian: usize) {
    voxel_map.par_iter_mut().for_each(|(_, cell)| {
        cell.recompute_covariance(min_points_per_gaussian);
    });
}

pub fn recompute_gaussians_for_keys(
    voxel_map: &mut VoxelMap,
    keys: &[VoxelKey],
    min_points_per_gaussian: usize,
    min_variance: f32,
    max_variance: f32,
    information_regularization: f32,
) {
    let results: Vec<(VoxelKey, VoxelCell)> = keys
        .par_iter()
        .filter_map(|&key| {
            let mut cell = voxel_map.get(&key)?.clone();
            cell.recompute_covariance(min_points_per_gaussian);
            Some((key, cell))
        })
        .collect();

    for (key, cell) in results {
        if let Some(dst) = voxel_map.get_mut(&key) {
            *dst = cell;
        }
    }
}

/// 変換済み点群を既存のGaussian voxel mapに追加する。
///
/// Gaussian版では「1 voxel = 1 Gaussian」なので、GICP版のように周辺3x3x3を
/// 再計算しない。変更されたvoxel自身だけを再計算する。
pub fn merge_points_into_gaussian_voxel_map(
    map: &mut VoxelMap,
    points: &[Point3<f32>],
    pose: &Isometry3<f32>,
    gaussian_voxel_size: f32,
    max_points_per_gaussian: usize,
    min_points_per_gaussian: usize,
    min_variance: f32,
    max_variance: f32,
    information_regularization: f32,
) {
    let mut modified_keys = HashSet::<VoxelKey>::new();

    for p in points {
        let transformed = pose.transform_point(p);
        let key = voxel_key(&transformed, gaussian_voxel_size);
        let cell = map.entry(key).or_insert_with(VoxelCell::new);

        if cell.push_point(transformed, max_points_per_gaussian) {
            modified_keys.insert(key);
        }
    }

    let keys: Vec<VoxelKey> = modified_keys.into_iter().collect();
    recompute_gaussians_for_keys(
        map,
        &keys,
        min_points_per_gaussian,
        min_variance,
        max_variance,
        information_regularization,
    );
}
