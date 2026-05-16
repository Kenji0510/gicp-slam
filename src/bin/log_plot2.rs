/// debug_data.json を読み込み、plotters で2グラフを PNG 出力する。
///
/// 1. correspondences_num の時系列
/// 2. dist (平均対応点距離) の時系列
///
use anyhow::{Context, Result, bail};
use plotters::prelude::*;
use serde::Deserialize;

const LOG_PATH: &str = "data/output/debug/05162026/debug_data.json";
const OUT_DIR: &str = "data/output/debug/05162026";

#[derive(Debug, Deserialize)]
struct DebugData {
    correspondences_num: usize,
    dist: f32,
}

fn main() -> Result<()> {
    std::fs::create_dir_all(OUT_DIR)?;

    let raw = std::fs::read_to_string(LOG_PATH)
        .with_context(|| format!("Cannot read {}", LOG_PATH))?;
    let data: Vec<DebugData> =
        serde_json::from_str(&raw).with_context(|| "Failed to parse debug_data.json")?;

    if data.is_empty() {
        bail!("debug_data.json is empty");
    }

    plot_correspondences(&data)?;
    plot_dist(&data)?;

    println!("Saved plots to {}", OUT_DIR);
    Ok(())
}

fn plot_correspondences(data: &[DebugData]) -> Result<()> {
    let path = format!("{}/correspondences_num.png", OUT_DIR);
    let root = BitMapBackend::new(&path, (1200, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let n = data.len();
    let max_val = data.iter().map(|d| d.correspondences_num).max().unwrap_or(1) as f32;

    let mut chart = ChartBuilder::on(&root)
        .caption("Correspondences per Frame", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
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
        .caption("Average Correspondence Distance per Frame", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..n as f32, 0f32..(max_val * 1.1).max(1.0))?;

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
