use bitvec::prelude::*;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use gdcompress::load_csv;
use gdcompress::prelude::*;
use gdcompress::{
    BitDataSet, BuildBaseTable, BuildBitDataSet, BuildImageBitDataSet, BuildSortedBaseTable,
    EntroGdError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputKind {
    Image,
    Csv,
}

fn main() -> Result<(), EntroGdError> {
    let args: Vec<String> = env::args().collect();

    let mut input_path: Option<PathBuf> = None;
    let mut recursive = false;
    let mut use_sorted = true;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--path" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --path".to_string(),
                    });
                }
                input_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--recursive" => {
                recursive = true;
                i += 1;
            }
            "--unsorted" => {
                use_sorted = false;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage(&args[0]);
                return Ok(());
            }
            other => {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("unknown argument: {}", other),
                });
            }
        }
    }

    let Some(input_path) = input_path else {
        print_usage(&args[0]);
        return Ok(());
    };

    let image_build_options = image_build_options();

    let raw_files = collect_files(&input_path, recursive)?;
    let mut typed_files: Vec<(PathBuf, InputKind)> = Vec::new();
    let mut skipped: Vec<PathBuf> = Vec::new();
    for file in raw_files {
        match classify(&file) {
            Some(kind) => typed_files.push((file, kind)),
            None => skipped.push(file),
        }
    }

    if typed_files.is_empty() {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "no supported image or csv files found under {}",
                input_path.display()
            ),
        });
    }

    let corpus_kind = typed_files[0].1;
    if let Some((mixed_path, _)) = typed_files.iter().find(|(_, k)| *k != corpus_kind) {
        return Err(EntroGdError::InvalidMetadata {
            message: format!(
                "mixed input kinds in corpus: {:?} expected, but {} is the other kind",
                corpus_kind,
                mixed_path.display()
            ),
        });
    }

    let run_dir = build_run_dir(&input_path, use_sorted);
    fs::create_dir_all(&run_dir)?;

    println!("=== Base-table delta lower-bound batch evaluation ===");
    println!("Input root:       {}", input_path.display());
    println!("Files in corpus:  {}", typed_files.len());
    println!("Corpus kind:      {:?}", corpus_kind);
    println!(
        "Table order:      {}",
        if use_sorted { "sorted" } else { "unsorted" }
    );
    println!("Output directory: {}", run_dir.display());
    if !skipped.is_empty() {
        println!("Skipped (unsupported extension): {}", skipped.len());
    }
    println!();
    println!(
        "  {:>40} | {:>6} | {:>5} | {:>12} | {:>12} | {:>6}",
        "file", "rows", "width", "delta_bits", "fixed_bits", "ratio"
    );
    println!(
        "  ------------------------------------------------------------------------------------------------"
    );

    let mut combined_distribution: BTreeMap<usize, usize> = BTreeMap::new();
    let mut combined_total_bits: usize = 0;
    let mut combined_fixed_raw_bits: usize = 0;
    let mut combined_absolute_per_row_bits: usize = 0;
    let mut delta_metric_label: &'static str = if use_sorted { "unsigned" } else { "signed" };

    for (file, _kind) in &typed_files {
        let result = evaluate_file(file, use_sorted, &image_build_options)?;

        let per_file_csv = run_dir.join(build_per_file_csv_name(
            file,
            use_sorted,
            &image_build_options,
        ));
        write_delta_distribution_csv(&per_file_csv, &result.delta_rows)?;

        for (bit_len, count) in &result.distribution {
            *combined_distribution.entry(*bit_len).or_insert(0) += count;
        }
        combined_total_bits += result.total_bits;
        combined_fixed_raw_bits += result.fixed_raw_bits;
        combined_absolute_per_row_bits += result.absolute_per_row_bits;
        delta_metric_label = result.delta_metric_label;

        let ratio = if result.fixed_raw_bits == 0 {
            0.0
        } else {
            result.total_bits as f64 / result.fixed_raw_bits as f64
        };
        let label = file
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unnamed>");
        println!(
            "  {:>40} | {:>6} | {:>5} | {:>12} | {:>12} | {:>6.4}",
            truncate(label, 40),
            result.entry_count,
            result.row_width,
            result.total_bits,
            result.fixed_raw_bits,
            ratio,
        );
    }

    let combined_rows = rows_from_distribution(&combined_distribution);
    let combined_csv = run_dir.join("__combined.csv");
    write_delta_distribution_csv(&combined_csv, &combined_rows)?;

    println!();
    println!("=== Combined across {} files ===", typed_files.len());
    println!(
        "Baseline raw fixed-width storage: {} bits ({:.2} bytes)",
        combined_fixed_raw_bits,
        combined_fixed_raw_bits as f64 / 8.0
    );
    println!(
        "Absolute per-row minimal width:   {} bits ({:.2} bytes)",
        combined_absolute_per_row_bits,
        combined_absolute_per_row_bits as f64 / 8.0
    );
    println!(
        "Delta-to-previous ({}) total:    {} bits ({:.2} bytes)",
        delta_metric_label,
        combined_total_bits,
        combined_total_bits as f64 / 8.0
    );
    if combined_fixed_raw_bits > 0 {
        println!(
            "Absolute minimal / fixed: {:.4}",
            combined_absolute_per_row_bits as f64 / combined_fixed_raw_bits as f64
        );
        println!(
            "Delta minimal / fixed:    {:.4}",
            combined_total_bits as f64 / combined_fixed_raw_bits as f64
        );
    }
    println!();
    println!("Combined CSV written to: {}", combined_csv.display());
    println!(
        "Combined delta {}-bit-length distribution:",
        delta_metric_label
    );
    if combined_rows.is_empty() {
        println!("  (no deltas)");
    } else {
        println!(
            "  {:>8} | {:>9} | {:>10} | {:>10} | {:>10}",
            "bit_len", "count", "cum_count", "cum<=len%", "cum_bits%"
        );
        println!("  ----------------------------------------------------------");
        for row in &combined_rows {
            println!(
                "  {:>6}b | {:>9} | {:>10} | {:>9.2}% | {:>9.2}%",
                row.bit_len, row.count, row.cum_count, row.cum_len_pct, row.cum_bits_pct
            );
        }
    }

    Ok(())
}

fn print_usage(program: &str) {
    println!(
        "Usage: {} --path <file_or_dir> [--recursive] [--unsorted]\n  default: sorted base table; --unsorted for signed deltas",
        program
    );
}

#[derive(Debug, Clone, Copy)]
struct DeltaDistributionRow {
    bit_len: usize,
    count: usize,
    cum_count: usize,
    cum_len_pct: f64,
    cum_bits_pct: f64,
}

struct FileEvalResult {
    delta_rows: Vec<DeltaDistributionRow>,
    distribution: BTreeMap<usize, usize>,
    total_bits: usize,
    fixed_raw_bits: usize,
    absolute_per_row_bits: usize,
    delta_metric_label: &'static str,
    entry_count: usize,
    row_width: usize,
}

fn evaluate_file(
    input_path: &Path,
    use_sorted: bool,
    _image_options: &BuildImageBitDataSet,
) -> Result<FileEvalResult, EntroGdError> {
    let bit_data = load_bit_data(input_path)?;

    let context = if use_sorted {
        Entropy {}
            .then(SelectBases { patience: 10 })
            .then(BuildSortedBaseTable {})
            .process(bit_data)?
    } else {
        Entropy {}
            .then(SelectBases { patience: 10 })
            .then(BuildBaseTable {})
            .process(bit_data)?
    };

    let rows: Vec<&BitSlice<usize, Lsb0>> = context
        .variable_base_table
        .iter()
        .map(|(bits, _)| bits.as_bitslice())
        .collect();

    if rows.is_empty() {
        return Ok(FileEvalResult {
            delta_rows: Vec::new(),
            distribution: BTreeMap::new(),
            total_bits: 0,
            fixed_raw_bits: 0,
            absolute_per_row_bits: 0,
            delta_metric_label: if use_sorted { "unsigned" } else { "signed" },
            entry_count: 0,
            row_width: 0,
        });
    }

    let row_width = rows[0].len();
    let entry_count = rows.len();
    let fixed_raw_bits = entry_count * row_width;

    let sort_key_order = build_sort_key_local_order(&context);
    let sort_key_bits: Vec<BitVec<usize, Lsb0>> = rows
        .iter()
        .map(|row| build_sort_key_bits(row, &sort_key_order))
        .collect();
    let sort_key_views: Vec<&BitSlice<usize, Lsb0>> = sort_key_bits
        .iter()
        .map(|bits| bits.as_bitslice())
        .collect();

    let absolute_per_row_bits: usize = sort_key_views
        .iter()
        .map(|key| bits_needed_unsigned(key))
        .sum();

    let (distribution, total_bits, delta_metric_label) =
        build_delta_distribution(&sort_key_views, use_sorted);

    let delta_rows = rows_from_distribution(&distribution);

    Ok(FileEvalResult {
        delta_rows,
        distribution,
        total_bits,
        fixed_raw_bits,
        absolute_per_row_bits,
        delta_metric_label,
        entry_count,
        row_width,
    })
}

fn image_build_options() -> BuildImageBitDataSet {
    BuildImageBitDataSet {
        colorspace: ImageColorSpace::SrgbWithLinearAlpha,
        color_model: ImageColorModel::YCoCgR,
        pixel_grouping: PixelGrouping::new(4, 4),
        grouping_transform: ImageGroupingTransform::ForFirstPixel,
        pad_rows_to_word: false,
    }
}

fn collect_files(path: &Path, recursive: bool) -> Result<Vec<PathBuf>, EntroGdError> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if !path.is_dir() {
        return Err(EntroGdError::InvalidMetadata {
            message: format!("input path does not exist: {}", path.display()),
        });
    }
    let mut out = Vec::new();
    walk_dir(path, recursive, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk_dir(path: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> Result<(), EntroGdError> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        if entry_path.is_file() {
            out.push(entry_path);
        } else if recursive && entry_path.is_dir() {
            walk_dir(&entry_path, recursive, out)?;
        }
    }
    Ok(())
}

fn classify(path: &Path) -> Option<InputKind> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())?
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "bmp" | "gif" => Some(InputKind::Image),
        "csv" => Some(InputKind::Csv),
        _ => None,
    }
}

fn build_run_dir(input_path: &Path, use_sorted: bool) -> PathBuf {
    let base = PathBuf::from("target/base_table_delta_compression");
    let label = if input_path.is_dir() {
        input_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("corpus")
            .to_string()
    } else {
        input_path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("dataset")
            .to_string()
    };
    let order_label = if use_sorted { "sorted" } else { "unsorted" };
    base.join(format!("{}__{}", label, order_label))
}

fn build_per_file_csv_name(
    input_path: &Path,
    use_sorted: bool,
    image_options: &BuildImageBitDataSet,
) -> String {
    let is_image = classify(input_path) == Some(InputKind::Image);
    let input_label = if is_image {
        input_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("image_dataset")
            .to_string()
    } else {
        input_path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("dataset")
            .to_string()
    };

    let order_label = if use_sorted { "sorted" } else { "unsorted" };
    let delta_label = if use_sorted { "unsigned" } else { "signed" };
    if is_image {
        format!(
            "{}__{}__{}__cs-{:?}__cm-{:?}__pg-{}__gt-{:?}.csv",
            input_label,
            order_label,
            delta_label,
            image_options.colorspace,
            image_options.color_model,
            image_options.pixel_grouping,
            image_options.grouping_transform,
        )
    } else {
        format!("{}__{}__{}__csv.csv", input_label, order_label, delta_label)
    }
}

fn write_delta_distribution_csv(
    output_path: &Path,
    rows: &[DeltaDistributionRow],
) -> Result<(), EntroGdError> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut writer = BufWriter::new(File::create(output_path)?);
    writeln!(writer, "bit_len,count,cum_count,cum<=len%,cum_bits%")?;

    for row in rows {
        writeln!(
            writer,
            "{},{},{},{:.2},{:.2}",
            row.bit_len, row.count, row.cum_count, row.cum_len_pct, row.cum_bits_pct
        )?;
    }

    writer.flush()?;
    Ok(())
}

fn load_bit_data(input_path: &Path) -> Result<BitDataSet, EntroGdError> {
    if classify(input_path) == Some(InputKind::Image) {
        OpenImage
            .then(image_build_options())
            .process(input_path.to_path_buf())
    } else {
        let df = load_csv(input_path, true, None)?;
        BuildBitDataSet::default().process(df)
    }
}

fn build_delta_distribution(
    rows: &[&BitSlice<usize, Lsb0>],
    use_sorted: bool,
) -> (BTreeMap<usize, usize>, usize, &'static str) {
    let label = if use_sorted { "unsigned" } else { "signed" };
    if rows.is_empty() {
        return (BTreeMap::new(), 0, label);
    }

    let mut distribution: BTreeMap<usize, usize> = BTreeMap::new();
    let mut total_bits = bits_needed_unsigned(rows[0]);

    // Pre-build a bitvec for 1, used to bias sorted deltas by 1.
    let mut one = BitVec::<usize, Lsb0>::with_capacity(1);
    one.push(true);

    for pair in rows.windows(2) {
        let prev = pair[0];
        let current = pair[1];

        let magnitude = abs_diff_unsigned(current, prev);

        // For sorted tables all entries are unique so delta >= 1. Bias by 1
        // (encode magnitude - 1) so 1 bit covers deltas 1 and 2, etc.
        let delta_bits = if use_sorted {
            let biased = subtract_unsigned(magnitude.as_bitslice(), one.as_bitslice());
            bits_needed_unsigned(biased.as_bitslice()).max(1)
        } else {
            let magnitude_bits = bits_needed_unsigned(magnitude.as_bitslice());
            if magnitude_bits == 0 {
                0
            } else {
                1 + magnitude_bits
            }
        };

        *distribution.entry(delta_bits).or_insert(0) += 1;
        total_bits += delta_bits;
    }

    (distribution, total_bits, label)
}

fn rows_from_distribution(distribution: &BTreeMap<usize, usize>) -> Vec<DeltaDistributionRow> {
    let total_deltas: usize = distribution.values().sum();
    let total_delta_bits_only: usize = distribution
        .iter()
        .map(|(bit_len, count)| bit_len * count)
        .sum();

    let mut cumulative_bits = 0usize;
    let mut cumulative_count = 0usize;
    let mut rows_out = Vec::with_capacity(distribution.len());

    for (bit_len, count) in distribution {
        cumulative_count += count;
        cumulative_bits += bit_len * count;

        let cum_len_pct = if total_deltas == 0 {
            0.0
        } else {
            100.0 * cumulative_count as f64 / total_deltas as f64
        };
        let cum_bits_pct = if total_delta_bits_only == 0 {
            0.0
        } else {
            100.0 * cumulative_bits as f64 / total_delta_bits_only as f64
        };

        rows_out.push(DeltaDistributionRow {
            bit_len: *bit_len,
            count: *count,
            cum_count: cumulative_count,
            cum_len_pct,
            cum_bits_pct,
        });
    }

    rows_out
}

fn build_sort_key_local_order(context: &gdcompress::PreEncodeContext) -> Vec<usize> {
    let row_width = context
        .variable_base_table
        .first()
        .map(|(bits, _)| bits.len())
        .unwrap_or(0);

    if let Some(order) = &context.entropy_sorted_column_order {
        let filtered: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&idx| idx < row_width)
            .collect();
        if !filtered.is_empty() {
            return filtered;
        }
    }

    (0..row_width).collect()
}

fn build_sort_key_bits(row: &BitSlice<usize, Lsb0>, order: &[usize]) -> BitVec<usize, Lsb0> {
    let mut out = BitVec::<usize, Lsb0>::with_capacity(order.len());
    for &col_idx in order.iter().rev() {
        out.push(row.get(col_idx).map(|b| *b).unwrap_or(false));
    }
    out
}

fn bits_needed_unsigned(bits: &BitSlice<usize, Lsb0>) -> usize {
    for idx in (0..bits.len()).rev() {
        if bits.get(idx).map(|b| *b).unwrap_or(false) {
            return idx + 1;
        }
    }
    0
}

fn abs_diff_unsigned(
    lhs: &BitSlice<usize, Lsb0>,
    rhs: &BitSlice<usize, Lsb0>,
) -> BitVec<usize, Lsb0> {
    match compare_unsigned(lhs, rhs) {
        Ordering::Greater | Ordering::Equal => subtract_unsigned(lhs, rhs),
        Ordering::Less => subtract_unsigned(rhs, lhs),
    }
}

fn compare_unsigned(lhs: &BitSlice<usize, Lsb0>, rhs: &BitSlice<usize, Lsb0>) -> Ordering {
    let max_len = lhs.len().max(rhs.len());
    for idx in (0..max_len).rev() {
        let l = lhs.get(idx).map(|b| *b).unwrap_or(false);
        let r = rhs.get(idx).map(|b| *b).unwrap_or(false);
        match l.cmp(&r) {
            Ordering::Equal => continue,
            non_equal => return non_equal,
        }
    }
    Ordering::Equal
}

fn subtract_unsigned(
    minuend: &BitSlice<usize, Lsb0>,
    subtrahend: &BitSlice<usize, Lsb0>,
) -> BitVec<usize, Lsb0> {
    let max_len = minuend.len().max(subtrahend.len());
    let mut out = BitVec::<usize, Lsb0>::with_capacity(max_len);

    let mut borrow: i8 = 0;
    for idx in 0..max_len {
        let a = if minuend.get(idx).map(|b| *b).unwrap_or(false) {
            1
        } else {
            0
        };
        let b = if subtrahend.get(idx).map(|b| *b).unwrap_or(false) {
            1
        } else {
            0
        };

        let mut diff = a - b - borrow;
        if diff < 0 {
            diff += 2;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.push(diff == 1);
    }

    while out.last().map(|bit| *bit) == Some(false) {
        out.pop();
    }

    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let cut = max.saturating_sub(1);
        let mut t: String = s.chars().take(cut).collect();
        t.push('…');
        t
    }
}
