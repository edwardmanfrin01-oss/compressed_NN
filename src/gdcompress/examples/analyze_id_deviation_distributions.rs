use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use bitvec::prelude::*;
use gdcompress::compression::data::DEFAULT_ALIGN_ROWS_TO_WORD;
use gdcompress::prelude::*;
use gdcompress::{EntroGdError, ImageColorModel, init_logging};

fn main() -> Result<(), EntroGdError> {
    unsafe {
        env::set_var("ENTRO_GD_LOG_TO_STDERR", "1");
        env::set_var("RUST_LOG", "info");
    }
    let _log_handle = init_logging();

    let args: Vec<String> = env::args().collect();

    let mut input_path: Option<PathBuf> = None;
    let mut output_path = PathBuf::from("target/distributions");

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--path" => {
                i += 1;
                if i < args.len() {
                    input_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output_path = PathBuf::from(&args[i]);
                }
            }
            "--help" | "-h" => {
                println!(
                    "Usage: analyze_id_deviation_distributions --path <image_or_folder> [--output <output_folder>]"
                );
                return Ok(());
            }
            other => {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("unknown argument: {}", other),
                });
            }
        }
        i += 1;
    }

    let input_path =
        input_path.unwrap_or_else(|| PathBuf::from("data/bench_datasets/kodak_dataset"));

    let files = collect_image_files(&input_path)?;
    if files.is_empty() {
        tracing::warn!("No image files found at {}", input_path.display());
        return Ok(());
    }

    let image_opener = OpenImage {};

    let pipeline = BuildImageBitDataSet {
        colorspace: ImageColorSpace::SrgbWithLinearAlpha,
        color_model: ImageColorModel::YCoCgR,
        pixel_grouping: PixelGrouping::new(2, 2),
        grouping_transform: ImageGroupingTransform::ForFirstPixel,
        pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
    }
    .then(Entropy {})
    .then(SelectBasesAdaptive {
        patience: 5,
        base_bit_impl: BaseBitImpl::Naive,
        width_decay: 0.4,
    })
    .then(BuildSortedBaseTable {});

    fs::create_dir_all(&output_path)?;

    let mut agg_id_counts: HashMap<u64, u64> = HashMap::new();
    let mut agg_dev_counts: HashMap<u64, u64> = HashMap::new();
    let mut summary_rows: Vec<SummaryRow> = Vec::new();

    for file in &files {
        let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("image");

        tracing::info!("Processing: {}", file.display());

        let image = image_opener.process(file.to_path_buf())?;
        let pre_encode = pipeline.process(image)?;

        let dev_positions = pre_encode.layout.deviation_bit_positions();
        let num_deviation_bits = dev_positions.len();

        let mut id_counts: HashMap<u64, u64> = HashMap::new();
        for &base_id in &pre_encode.row_to_base_id {
            *id_counts.entry(base_id as u64).or_insert(0) += 1;
        }

        let mut dev_counts: HashMap<u64, u64> = HashMap::new();
        let num_rows = pre_encode.bit_data.data.num_rows;
        for row_idx in 0..num_rows {
            let chunk = pre_encode.bit_data.data.get_chunk(row_idx);
            let dev_val = extract_bits_at_positions(chunk, &dev_positions);
            *dev_counts.entry(dev_val).or_insert(0) += 1;
        }

        let per_image_dir = output_path.join(stem);
        fs::create_dir_all(&per_image_dir)?;
        write_frequency_csv(&per_image_dir.join("base_ids.csv"), &id_counts)?;
        write_frequency_csv(&per_image_dir.join("deviations.csv"), &dev_counts)?;

        let num_samples = pre_encode.row_to_base_id.len();
        let num_bases = pre_encode.variable_base_table.len();
        let id_entropy = entropy(&id_counts);
        let dev_entropy = entropy(&dev_counts);

        tracing::info!(
            "  {} samples, {} bases (entropy {:.3}/{:.3} bits), {} unique deviations / {} possible (entropy {:.3}/{:.3} bits)",
            num_samples,
            num_bases,
            id_entropy,
            (num_bases as f64).log2(),
            dev_counts.len(),
            1u64.checked_shl(num_deviation_bits.min(63) as u32)
                .unwrap_or(u64::MAX),
            dev_entropy,
            num_deviation_bits as f64,
        );

        summary_rows.push(SummaryRow {
            image: stem.to_string(),
            num_samples,
            num_bases,
            num_unique_base_ids: id_counts.len(),
            base_id_entropy_bits: id_entropy,
            max_base_id_entropy_bits: (num_bases as f64).log2(),
            num_deviation_bits,
            num_unique_deviations: dev_counts.len(),
            deviation_entropy_bits: dev_entropy,
        });

        for (&k, &v) in &id_counts {
            *agg_id_counts.entry(k).or_insert(0) += v;
        }
        for (&k, &v) in &dev_counts {
            *agg_dev_counts.entry(k).or_insert(0) += v;
        }
    }

    write_frequency_csv(&output_path.join("aggregate_base_ids.csv"), &agg_id_counts)?;
    write_frequency_csv(
        &output_path.join("aggregate_deviations.csv"),
        &agg_dev_counts,
    )?;
    write_summary_csv(&output_path.join("summary.csv"), &summary_rows)?;

    tracing::info!("Done. Output written to: {}", output_path.display());
    Ok(())
}

struct SummaryRow {
    image: String,
    num_samples: usize,
    num_bases: usize,
    num_unique_base_ids: usize,
    base_id_entropy_bits: f64,
    max_base_id_entropy_bits: f64,
    num_deviation_bits: usize,
    num_unique_deviations: usize,
    deviation_entropy_bits: f64,
}

fn collect_image_files(path: &Path) -> Result<Vec<PathBuf>, EntroGdError> {
    if path.is_dir() {
        let mut files = Vec::new();
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let p = entry.path();
            if p.is_file()
                && let Some(ext) = p.extension()
            {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if matches!(ext_str.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "gif") {
                    files.push(p);
                }
            }
        }
        files.sort();
        Ok(files)
    } else {
        Ok(vec![path.to_path_buf()])
    }
}

fn extract_bits_at_positions(chunk: &BitSlice<usize, Lsb0>, positions: &[usize]) -> u64 {
    let capped = positions.len().min(64);
    positions[..capped]
        .iter()
        .enumerate()
        .fold(0u64, |acc, (bit_idx, &pos)| {
            if chunk.get(pos).map(|b| *b).unwrap_or(false) {
                acc | (1u64 << bit_idx)
            } else {
                acc
            }
        })
}

fn entropy(counts: &HashMap<u64, u64>) -> f64 {
    let total: u64 = counts.values().sum();
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    counts.values().filter(|&&c| c > 0).fold(0.0, |acc, &c| {
        let p = c as f64 / total_f;
        acc - p * p.log2()
    })
}

fn write_frequency_csv(output_path: &Path, counts: &HashMap<u64, u64>) -> Result<(), EntroGdError> {
    let mut sorted: Vec<(u64, u64)> = counts.iter().map(|(&k, &v)| (k, v)).collect();
    sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let mut writer = BufWriter::new(std::fs::File::create(output_path)?);
    writeln!(writer, "value,count")?;
    for (value, count) in &sorted {
        writeln!(writer, "{},{}", value, count)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_summary_csv(output_path: &Path, rows: &[SummaryRow]) -> Result<(), EntroGdError> {
    let mut writer = BufWriter::new(std::fs::File::create(output_path)?);
    writeln!(
        writer,
        "image,num_samples,num_bases,num_unique_base_ids,base_id_entropy_bits,max_base_id_entropy_bits,num_deviation_bits,num_unique_deviations,deviation_entropy_bits"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{:.4},{:.4},{},{},{:.4}",
            row.image,
            row.num_samples,
            row.num_bases,
            row.num_unique_base_ids,
            row.base_id_entropy_bits,
            row.max_base_id_entropy_bits,
            row.num_deviation_bits,
            row.num_unique_deviations,
            row.deviation_entropy_bits,
        )?;
    }
    writer.flush()?;
    Ok(())
}
