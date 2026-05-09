use std::collections::HashMap;

use crate::{
    compute_covariance::{VoxelCell, VoxelKey},
    types::PointXYZCov,
};

pub fn convert_voxel_map_to_pcd(voxel_map: &HashMap<VoxelKey, VoxelCell>) -> Vec<PointXYZCov> {
    voxel_map
        .iter()
        .filter(|(_, cell)| cell.covariance_valid)
        .map(|(_, cell)| PointXYZCov {
            x: cell.mean.x,
            y: cell.mean.y,
            z: cell.mean.z,
            cov_xx: cell.gicp_covariance[(0, 0)],
            cov_xy: cell.gicp_covariance[(0, 1)],
            cov_xz: cell.gicp_covariance[(0, 2)],
            cov_yy: cell.gicp_covariance[(1, 1)],
            cov_yz: cell.gicp_covariance[(1, 2)],
            cov_zz: cell.gicp_covariance[(2, 2)],
        })
        .collect()
}
