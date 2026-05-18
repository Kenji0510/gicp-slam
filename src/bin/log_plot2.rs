/// debug_gicp_data.json と debug_process_times.json を読み込み、plotters で PNG 出力する。
///
/// 1. correspondences_num の時系列
/// 2. dist (平均対応点距離) の時系列
/// 3. 各処理のフレームごとの処理時間 (ms)
///
use anyhow::{Context, Result, bail};
use plotters::prelude::*;
use serde::Deserialize;

const GICP_DATA_PATH: &str = "data/output/debug/05162026/park08/debug_gicp_data.json";
const PROCESS_TIME_PATH: &str = "data/output/debug/05162026/park08/debug_process_times.json";
const OUT_DIR: &str = "data/output/debug/05162026/outdoor08";

#[derive(Debug, Deserialize)]
struct DebugData {
    correspondences_num: usize,
    dist: f32,
}

#[derive(Debug, Deserialize)]
struct DebugProcessTime {
    voxelization_time_ms: f32,
    create_voxel_map_time_ms: f32,
    find_correspondences_time_ms: f32,
    total_gicp_time_ms: f32,
    gicp_time_ms: f32,
    merge_time_ms: f32,
}

fn main() -> Result<()> {
    std::fs::create_dir_all(OUT_DIR)?;

    let raw = std::fs::read_to_string(GICP_DATA_PATH)
        .with_context(|| format!("Cannot read {}", GICP_DATA_PATH))?;
    let gicp_data: Vec<DebugData> =
        serde_json::from_str(&raw).with_context(|| "Failed to parse debug_gicp_data.json")?;

    let raw = std::fs::read_to_string(PROCESS_TIME_PATH)
        .with_context(|| format!("Cannot read {}", PROCESS_TIME_PATH))?;
    let process_times: Vec<DebugProcessTime> =
        serde_json::from_str(&raw).with_context(|| "Failed to parse debug_process_times.json")?;

    if gicp_data.is_empty() || process_times.is_empty() {
        bail!("JSON data is empty");
    }

    plot_correspondences(&gicp_data)?;
    plot_dist(&gicp_data)?;
    plot_process_times(&process_times)?;

    println!("Saved plots to {}", OUT_DIR);
    Ok(())
}

// ---------------------------------------------------------------------------
// 1. correspondences_num
// ---------------------------------------------------------------------------
fn plot_correspondences(data: &[DebugData]) -> Result<()> {
    let path = format!("{}/correspondences_num.png", OUT_DIR);
    let root = BitMapBackend::new(&path, (1200, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let n = data.len();
    let max_val = data
        .iter()
        .map(|d| d.correspondences_num)
        .max()
        .unwrap_or(1) as f32;

    let mut chart = ChartBuilder::on(&root)
        .caption("Correspondences per Frame", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(70)
        .build_cartesian_2d(0f32..n as f32, 0f32..(max_val * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("Frame")
        .y_desc("Correspondences")
        .draw()?;

    chart.draw_series(LineSeries::new(
        data.iter()
            .enumerate()
            .map(|(i, d)| (i as f32, d.correspondences_num as f32)),
        &BLUE,
    ))?;

    root.present()?;
    println!("Saved: {}", path);
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. 平均対応点距離
// ---------------------------------------------------------------------------
fn plot_dist(data: &[DebugData]) -> Result<()> {
    let path = format!("{}/avg_dist.png", OUT_DIR);
    let root = BitMapBackend::new(&path, (1200, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let n = data.len();
    let max_val = data
        .iter()
        .map(|d| d.dist)
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Average Correspondence Distance per Frame",
            ("sans-serif", 24),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(70)
        .build_cartesian_2d(0f32..n as f32, 0f32..(max_val * 1.1).max(0.1))?;

    chart
        .configure_mesh()
        .x_desc("Frame")
        .y_desc("Avg Dist [m²]")
        .draw()?;

    chart.draw_series(LineSeries::new(
        data.iter()
            .enumerate()
            .filter(|(_, d)| d.dist.is_finite())
            .map(|(i, d)| (i as f32, d.dist)),
        &RED,
    ))?;

    root.present()?;
    println!("Saved: {}", path);
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. 各処理の処理時間 (複数系列)
// ---------------------------------------------------------------------------
fn plot_process_times(data: &[DebugProcessTime]) -> Result<()> {
    let path = format!("{}/process_times.png", OUT_DIR);
    let root = BitMapBackend::new(&path, (1200, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let n = data.len();

    let series: &[(&str, RGBColor, Box<dyn Fn(&DebugProcessTime) -> f32>)] = &[
        (
            "Voxelization",
            RGBColor(31, 119, 180),
            Box::new(|d| d.voxelization_time_ms),
        ),
        (
            "Create VoxelMap",
            RGBColor(255, 127, 14),
            Box::new(|d| d.create_voxel_map_time_ms),
        ),
        (
            "Find Corresp",
            RGBColor(44, 160, 44),
            Box::new(|d| d.find_correspondences_time_ms),
        ),
        (
            "GICP Solve",
            RGBColor(214, 39, 40),
            Box::new(|d| d.gicp_time_ms),
        ),
        (
            "Merge",
            RGBColor(148, 103, 189),
            Box::new(|d| d.merge_time_ms),
        ),
    ];

    let max_val = series
        .iter()
        .flat_map(|(_, _, f)| data.iter().map(move |d| f(d)))
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Processing Time per Frame", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(70)
        .build_cartesian_2d(0f32..n as f32, 0f32..(max_val * 1.15).max(1.0))?;

    chart
        .configure_mesh()
        .x_desc("Frame")
        .y_desc("Time [ms]")
        .draw()?;

    for (label, color, getter) in series {
        chart
            .draw_series(LineSeries::new(
                data.iter()
                    .enumerate()
                    .filter(|(_, d)| getter(d).is_finite())
                    .map(|(i, d)| (i as f32, getter(d))),
                color,
            ))?
            .label(*label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], *color));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Saved: {}", path);
    Ok(())
}
