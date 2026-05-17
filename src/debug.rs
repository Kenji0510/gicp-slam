use std::collections::HashMap;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{
    compute_covariance::{VoxelCell, VoxelKey},
    types::PointXYZCov,
};

pub fn convert_voxel_map_to_pcd(voxel_map: &FxHashMap<VoxelKey, VoxelCell>) -> Vec<PointXYZCov> {
    voxel_map
        .iter()
        .filter(|(_, cell)| cell.valid)
        .map(|(_, cell)| PointXYZCov {
            x: cell.mean.x,
            y: cell.mean.y,
            z: cell.mean.z,
            cov_xx: cell.raw_covariance[(0, 0)],
            cov_xy: cell.raw_covariance[(0, 1)],
            cov_xz: cell.raw_covariance[(0, 2)],
            cov_yy: cell.raw_covariance[(1, 1)],
            cov_yz: cell.raw_covariance[(1, 2)],
            cov_zz: cell.raw_covariance[(2, 2)],
        })
        // .map(|(_, cell)| {
        //     cell.points.iter().map(|p| PointXYZCov {
        //         x: p.x,
        //         y: p.y,
        //         z: p.z,
        //         cov_xx: cell.covariance[(0, 0)],
        //         cov_xy: cell.covariance[(0, 1)],
        //         cov_xz: cell.covariance[(0, 2)],
        //         cov_yy: cell.covariance[(1, 1)],
        //         cov_yz: cell.covariance[(1, 2)],
        //         cov_zz: cell.covariance[(2, 2)],
        //     })
        // })
        // .flatten()
        .collect()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DebugData {
    pub correspondences_num: usize,
    pub dist: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DebugProcessTime {
    pub voxelization_time_ms: f32,
    // pub covariance_time_ms: f32,
    pub create_voxel_map_time_ms: f32,
    pub find_correspondences_time_ms: f32,
    pub total_gicp_time_ms: f32,
    pub gicp_time_ms: f32,
    pub merge_time_ms: f32,
}