use anyhow::Result;
use lidar_slam::{
    file_handler::load_imu_data,
    predict_pose_by_imu::align_imu_timestamps,
};
use plotters::prelude::*;

const LOAD_DIR: &str =
    "/home/kenji/workspace/rust/get_lidar_data/data/output/05092026/hallway04";
const OUTPUT_PNG: &str = "data/output/debug/imu_accel.png";

fn main() -> Result<()> {
    let imu_file = format!("{}/imu/imu_data.json", LOAD_DIR);
    let raw = load_imu_data(&imu_file)?;
    let imu = align_imu_timestamps(&raw);

    println!("Loaded {} IMU samples", imu.len());

    // タイムスタンプをゼロ起点に正規化
    let t0 = imu[0].timestamp;
    let times: Vec<f64> = imu.iter().map(|s| s.timestamp - t0).collect();

    let ax: Vec<f32> = imu.iter().map(|s| s.linear_acceleration[0]).collect();
    let ay: Vec<f32> = imu.iter().map(|s| s.linear_acceleration[1]).collect();
    let az: Vec<f32> = imu.iter().map(|s| s.linear_acceleration[2]).collect();

    let t_max = times.last().copied().unwrap_or(1.0);
    let a_min = ax.iter().chain(ay.iter()).chain(az.iter())
        .copied().fold(f32::INFINITY, f32::min);
    let a_max = ax.iter().chain(ay.iter()).chain(az.iter())
        .copied().fold(f32::NEG_INFINITY, f32::max);
    let margin = (a_max - a_min) * 0.05 + 0.01;

    std::fs::create_dir_all(std::path::Path::new(OUTPUT_PNG).parent().unwrap())?;
    let root = BitMapBackend::new(OUTPUT_PNG, (1600, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("IMU linear_acceleration (g)", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..t_max, (a_min - margin) as f64..(a_max + margin) as f64)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("acceleration (g)")
        .draw()?;

    let n = times.len();

    chart
        .draw_series(LineSeries::new(
            (0..n).map(|i| (times[i], ax[i] as f64)),
            &RED,
        ))?
        .label("ax")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            (0..n).map(|i| (times[i], ay[i] as f64)),
            &GREEN,
        ))?
        .label("ay")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart
        .draw_series(LineSeries::new(
            (0..n).map(|i| (times[i], az[i] as f64)),
            &BLUE,
        ))?
        .label("az")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Saved: {}", OUTPUT_PNG);
    Ok(())
}
