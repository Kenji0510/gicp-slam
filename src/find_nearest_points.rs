use nalgebra::Point3;
use rayon::prelude::*;

use crate::compute_covariance::{VoxelCell, VoxelKey, VoxelMap, voxel_key};

pub struct Correspondence<'a> {
    pub source_cell: &'a VoxelCell,
    pub target_cell: &'a VoxelCell,
    pub dist_sq: f32,
}

/// source_voxel_map の各ボクセルに対して、target_voxel_map の近傍 (-search_range..=search_range)^3
/// から最近傍（mean間距離が最小）のボクセルを探索する。
pub fn find_nearest_voxels<'a>(
    source_voxel_map: &'a VoxelMap,
    target_voxel_map: &'a VoxelMap,
    voxel_size: f32,
    search_range: i32,
    max_dist_sq: Option<f32>,
) -> Vec<Correspondence<'a>> {
    source_voxel_map
        .par_iter()
        .filter_map(|(_, src_cell)| {
            if !src_cell.valid {
                return None;
            }

            let base_key = voxel_key(&src_cell.mean, voxel_size);

            let mut best_dist_sq = f32::MAX;
            let mut best_cell: Option<&VoxelCell> = None;

            for dx in -search_range..=search_range {
                for dy in -search_range..=search_range {
                    for dz in -search_range..=search_range {
                        let candidate_key = VoxelKey {
                            ix: base_key.ix + dx,
                            iy: base_key.iy + dy,
                            iz: base_key.iz + dz,
                        };

                        let Some(tgt_cell) = target_voxel_map.get(&candidate_key) else {
                            continue;
                        };
                        if !tgt_cell.valid {
                            continue;
                        }

                        let diff = src_cell.mean - tgt_cell.mean;
                        let dist_sq = diff.norm_squared();

                        if dist_sq < best_dist_sq {
                            best_dist_sq = dist_sq;
                            best_cell = Some(tgt_cell);
                        }
                    }
                }
            }

            let tgt_cell = best_cell?;
            if max_dist_sq.map_or(true, |limit| best_dist_sq <= limit) {
                Some(Correspondence {
                    source_cell: src_cell,
                    target_cell: tgt_cell,
                    dist_sq: best_dist_sq,
                })
            } else {
                None
            }
        })
        .collect()
}
