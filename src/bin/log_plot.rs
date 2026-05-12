/// pose_log.json を読み込み、plotters で以下の4グラフを PNG 出力する。
///
/// 1. XY軌跡     — OK/SKIP_ROT/DIVERGED で色分け
/// 2. translation_diff の時系列 — 発散タイミング確認
/// 3. num_correspondences の時系列 — マッチング品質
/// 4. imu_rot_norm の時系列 — IMU回転量
///
use anyhow::{Context, Result, bail};
use plotters::prelude::*;
use serde::Deserialize;

const LOG_PATH: &str = "data/output/debug/05092026/pose_log.json";
const OUT_DIR: &str = "data/output/debug/05092026";

#[derive(Debug, Deserialize)]
struct FrameLog {
    frame: usize,
    proposed_tx: f32,
    proposed_ty: f32,
    proposed_roll: f32,
    proposed_pitch: f32,
    proposed_yaw: f32,
    translation_diff: f32,
    imu_rot_norm: f32,
    num_correspondences: usize,
    consecutive_skips: usize,
    status: String,
}

fn main() -> Result<()> {
    let log_path = LOG_PATH;
    let out_dir = OUT_DIR;
    std::fs::create_dir_all(out_dir)?;

    let raw =
        std::fs::read_to_string(log_path).with_context(|| format!("Cannot read {}", log_path))?;
    let logs: Vec<FrameLog> =
        serde_json::from_str(&raw).with_context(|| "Failed to parse pose_log.json")?;

    if logs.is_empty() {
        bail!("pose_log.json is empty");
    }

    plot_trajectory(&logs, out_dir)?;
    plot_translation_diff(&logs, out_dir)?;
    plot_correspondences(&logs, out_dir)?;
    plot_imu_rot(&logs, out_dir)?;

    println!("Saved plots to {}", out_dir);
    Ok(())
}

// ---------------------------------------------------------------------------
// 1. XY 軌跡
// ---------------------------------------------------------------------------
fn plot_trajectory(logs: &[FrameLog], out_dir: &str) -> Result<()> {
    let path = format!("{}/trajectory_xy.png", out_dir);
    let root = BitMapBackend::new(&path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let xs: Vec<f32> = logs.iter().map(|l| l.proposed_tx).collect();
    let ys: Vec<f32> = logs.iter().map(|l| l.proposed_ty).collect();
    let x_min = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y_min = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let margin = ((x_max - x_min).max(y_max - y_min) * 0.05).max(0.5);

    let mut chart = ChartBuilder::on(&root)
        .caption("XY Trajectory", ("sans-serif", 28))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (x_min - margin)..(x_max + margin),
            (y_min - margin)..(y_max + margin),
        )?;
    chart
        .configure_mesh()
        .x_desc("X [m]")
        .y_desc("Y [m]")
        .draw()?;

    // 線（全体を薄くつなぐ）
    chart.draw_series(LineSeries::new(
        logs.iter().map(|l| (l.proposed_tx, l.proposed_ty)),
        &RGBColor(180, 180, 180),
    ))?;

    // 点を status 別に色分け
    for l in logs {
        let color = status_color(&l.status);
        chart.draw_series(std::iter::once(Circle::new(
            (l.proposed_tx, l.proposed_ty),
            4,
            color.filled(),
        )))?;
    }

    // 凡例
    let legend_items: &[(&str, RGBColor)] = &[
        ("OK", RGBColor(0, 150, 0)),
        ("FORCE_UPDATE", RGBColor(0, 100, 200)),
        ("SKIP_ROT", RGBColor(200, 130, 0)),
        ("DIVERGED", RGBColor(200, 0, 0)),
    ];
    for (label, color) in legend_items {
        chart
            .draw_series(std::iter::once(Circle::new(
                (x_min, y_max),
                0,
                color.filled(),
            )))?
            .label(*label)
            .legend(move |(x, y)| Circle::new((x + 8, y), 6, color.filled()));
    }
    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("  trajectory_xy.png");
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. translation_diff の時系列
// ---------------------------------------------------------------------------
fn plot_translation_diff(logs: &[FrameLog], out_dir: &str) -> Result<()> {
    let path = format!("{}/translation_diff.png", out_dir);
    let root = BitMapBackend::new(&path, (1200, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_val = logs
        .iter()
        .map(|l| l.translation_diff)
        .fold(0.0_f32, f32::max)
        .max(0.1);
    let n = logs.len();

    let mut chart = ChartBuilder::on(&root)
        .caption("Translation Diff per Frame", ("sans-serif", 22))
        .margin(20)
        .x_label_area_size(35)
        .y_label_area_size(55)
        .build_cartesian_2d(0..n, 0.0_f32..(max_val * 1.1))?;
    chart
        .configure_mesh()
        .x_desc("Frame")
        .y_desc("Trans diff [m]")
        .draw()?;

    // 発散しきい値ライン（1.0m）
    chart
        .draw_series(LineSeries::new(
            [(0, 1.0_f32), (n, 1.0_f32)],
            RED.stroke_width(1),
        ))?
        .label("diverge threshold (1m)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 16, y)], RED));

    // 棒グラフ風に status で色分け
    chart.draw_series(logs.iter().map(|l| {
        let color = status_color(&l.status);
        Rectangle::new(
            [(l.frame, 0.0), (l.frame + 1, l.translation_diff)],
            color.filled(),
        )
    }))?;

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;
    root.present()?;
    println!("  translation_diff.png");
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. num_correspondences の時系列
// ---------------------------------------------------------------------------
fn plot_correspondences(logs: &[FrameLog], out_dir: &str) -> Result<()> {
    let path = format!("{}/num_correspondences.png", out_dir);
    let root = BitMapBackend::new(&path, (1200, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_val = logs
        .iter()
        .map(|l| l.num_correspondences)
        .max()
        .unwrap_or(1);
    let n = logs.len();

    let mut chart = ChartBuilder::on(&root)
        .caption("Number of Correspondences per Frame", ("sans-serif", 22))
        .margin(20)
        .x_label_area_size(35)
        .y_label_area_size(55)
        .build_cartesian_2d(0..n, 0usize..(max_val + 10))?;
    chart
        .configure_mesh()
        .x_desc("Frame")
        .y_desc("# correspondences")
        .draw()?;

    chart.draw_series(LineSeries::new(
        logs.iter().map(|l| (l.frame, l.num_correspondences)),
        &RGBColor(0, 100, 200),
    ))?;

    root.present()?;
    println!("  num_correspondences.png");
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. imu_rot_norm の時系列
// ---------------------------------------------------------------------------
fn plot_imu_rot(logs: &[FrameLog], out_dir: &str) -> Result<()> {
    let path = format!("{}/imu_rot_norm.png", out_dir);
    let root = BitMapBackend::new(&path, (1200, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_val = logs
        .iter()
        .map(|l| l.imu_rot_norm)
        .fold(0.0_f32, f32::max)
        .max(0.05);
    let n = logs.len();

    let mut chart = ChartBuilder::on(&root)
        .caption("IMU Rotation Norm per Frame", ("sans-serif", 22))
        .margin(20)
        .x_label_area_size(35)
        .y_label_area_size(55)
        .build_cartesian_2d(0..n, 0.0_f32..(max_val * 1.1))?;
    chart
        .configure_mesh()
        .x_desc("Frame")
        .y_desc("IMU rot norm [rad]")
        .draw()?;

    // マップ更新スキップのしきい値（0.025 rad）
    chart
        .draw_series(LineSeries::new(
            [(0, 0.025_f32), (n, 0.025_f32)],
            RED.stroke_width(1),
        ))?
        .label("skip threshold (0.025 rad)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 16, y)], RED));

    chart.draw_series(LineSeries::new(
        logs.iter().map(|l| (l.frame, l.imu_rot_norm)),
        &RGBColor(150, 0, 200),
    ))?;

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;
    root.present()?;
    println!("  imu_rot_norm.png");
    Ok(())
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
fn status_color(status: &str) -> RGBColor {
    match status {
        "OK" => RGBColor(0, 150, 0),
        "FORCE_UPDATE" => RGBColor(0, 100, 200),
        "SKIP_ROT" => RGBColor(200, 130, 0),
        "DIVERGED" => RGBColor(200, 0, 0),
        _ => RGBColor(100, 100, 100),
    }
}
