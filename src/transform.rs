use nalgebra::{Matrix4, Point3, Vector4};

pub fn transform_points_to_global_frame(
    points: &[Point3<f32>],
    pose: &Matrix4<f64>,
) -> Vec<Point3<f32>> {
    points
        .iter()
        .map(|p| {
            let p_hom = Vector4::new(p.x as f64, p.y as f64, p.z as f64, 1.0);
            let transformed = pose * p_hom;
            Point3::new(
                transformed.x as f32,
                transformed.y as f32,
                transformed.z as f32,
            )
        })
        .collect()
}
