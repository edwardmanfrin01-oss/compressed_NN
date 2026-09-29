use gdcompress::compression::BaseBitHyperLogLogCount;
use gdcompress::compression::base_bits::BaseBitGroups;
use gdcompress::compression::data::DEFAULT_ALIGN_ROWS_TO_WORD;
use gdcompress::load_csv;
use gdcompress::{
    BitDataSet, BuildImageBitDataSet, EntroGdError, Filter, FilterExt, ImageColorModel,
    ImageColorSpace, ImageGroupingTransform, OpenImage, PixelGrouping, PreprocessOptions,
    calculate_entropy, init_logging,
};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::path::PathBuf;

#[derive(Clone, Copy)]
enum InputKind {
    Auto,
    Csv,
    Image,
}

struct CliOptions {
    input: PathBuf,
    output: Option<PathBuf>,
    has_headers: bool,
    limit_bits: Option<usize>,
    input_kind: InputKind,
}

struct StepRecord {
    step: usize,
    bit_position: usize,
    entropy: f64,
    is_constant: bool,
    exact_num_bases: usize,
    approx_num_bases: usize,
    abs_error: usize,
    relative_error: f64,
}

fn parse_bool(value: &str) -> Result<bool, EntroGdError> {
    match value {
        "1" | "true" | "TRUE" | "yes" | "YES" => Ok(true),
        "0" | "false" | "FALSE" | "no" | "NO" => Ok(false),
        _ => Err(EntroGdError::InvalidMetadata {
            message: format!("invalid bool value '{}'; use true/false", value),
        }),
    }
}

fn parse_args() -> Result<CliOptions, EntroGdError> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut has_headers = true;
    let mut limit_bits: Option<usize> = None;
    let mut input_kind = InputKind::Auto;

    let args: Vec<String> = env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --input".to_string(),
                    });
                }
                input = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--output" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --output".to_string(),
                    });
                }
                output = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--headers" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --headers".to_string(),
                    });
                }
                has_headers = parse_bool(&args[i + 1])?;
                i += 2;
            }
            "--limit-bits" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --limit-bits".to_string(),
                    });
                }
                let value =
                    args[i + 1]
                        .parse::<usize>()
                        .map_err(|err| EntroGdError::InvalidMetadata {
                            message: format!(
                                "invalid --limit-bits value '{}': {}",
                                args[i + 1],
                                err
                            ),
                        })?;
                limit_bits = Some(value);
                i += 2;
            }
            "--input-kind" => {
                if i + 1 >= args.len() {
                    return Err(EntroGdError::InvalidMetadata {
                        message: "missing value for --input-kind".to_string(),
                    });
                }
                input_kind = match args[i + 1].as_str() {
                    "auto" => InputKind::Auto,
                    "csv" => InputKind::Csv,
                    "image" => InputKind::Image,
                    other => {
                        return Err(EntroGdError::InvalidMetadata {
                            message: format!(
                                "invalid --input-kind value '{}'; use auto|csv|image",
                                other
                            ),
                        });
                    }
                };
                i += 2;
            }
            other => {
                return Err(EntroGdError::InvalidMetadata {
                    message: format!("unknown argument: {}", other),
                });
            }
        }
    }

    let input = input.ok_or_else(|| EntroGdError::InvalidMetadata {
        message: "--input is required".to_string(),
    })?;

    Ok(CliOptions {
        input,
        output,
        has_headers,
        limit_bits,
        input_kind,
    })
}

fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            matches!(
                e.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "bmp" | "gif"
            )
        })
        .unwrap_or(false)
}

fn infer_input_kind(path: &Path) -> InputKind {
    if is_image_path(path) {
        InputKind::Image
    } else {
        InputKind::Csv
    }
}

fn collect_image_paths(dir: &Path) -> Result<Vec<PathBuf>, EntroGdError> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && is_image_path(&path) {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn load_bit_data(
    path: &Path,
    kind: InputKind,
    has_headers: bool,
) -> Result<BitDataSet, EntroGdError> {
    let resolved_kind = match kind {
        InputKind::Auto => infer_input_kind(path),
        explicit => explicit,
    };

    match resolved_kind {
        InputKind::Csv => {
            let df = load_csv(path, has_headers, None)?;
            BitDataSet::from_dataframe(df, PreprocessOptions::default(), false)
        }
        InputKind::Image => OpenImage
            .then(BuildImageBitDataSet {
                colorspace: ImageColorSpace::SrgbWithLinearAlpha,
                color_model: ImageColorModel::YCoCgR,
                pixel_grouping: PixelGrouping::new(2, 2),
                grouping_transform: ImageGroupingTransform::ForFirstPixel,
                pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
            })
            .process(path.to_path_buf()),
        InputKind::Auto => unreachable!("auto input kind should be resolved before load"),
    }
}

fn evaluate(bit_data: &BitDataSet, limit_bits: Option<usize>) -> Vec<StepRecord> {
    let mut entropy = calculate_entropy(bit_data);
    entropy.sort_by(|a, b| a.1.total_cmp(&b.1));

    let max_steps = limit_bits
        .map(|limit| limit.min(entropy.len()))
        .unwrap_or(entropy.len());

    let mut exact = BaseBitGroups::new(bit_data.num_rows(), bit_data.chunk_size());
    let mut approx = BaseBitHyperLogLogCount::new(bit_data.num_rows(), bit_data.chunk_size());

    let mut records = Vec::with_capacity(max_steps);
    for (step, (bit_position, entropy_value)) in entropy.into_iter().take(max_steps).enumerate() {
        let is_constant = entropy_value == 0.0;
        if is_constant {
            exact.add_constant_bit_positions(&[bit_position]);
            approx.add_constant_bit_positions(&[bit_position]);
        } else {
            exact.add_bit_position(bit_data, bit_position);
            approx.add_bit_position(bit_data, bit_position);
        }

        let exact_num_bases = exact.get_num_bases();
        let approx_num_bases = approx.get_num_bases();
        let abs_error = approx_num_bases.abs_diff(exact_num_bases);
        let relative_error = if exact_num_bases == 0 {
            0.0
        } else {
            (approx_num_bases as f64 - exact_num_bases as f64) / exact_num_bases as f64
        };

        records.push(StepRecord {
            step: step + 1,
            bit_position,
            entropy: entropy_value,
            is_constant,
            exact_num_bases,
            approx_num_bases,
            abs_error,
            relative_error,
        });
    }
    records
}

fn write_records(path: &Path, records: &[StepRecord]) -> Result<(), EntroGdError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "step,bit_position,entropy,is_constant,exact_num_bases,approx_num_bases,abs_error,relative_error"
    )?;
    for r in records {
        writeln!(
            writer,
            "{},{},{:.8},{},{},{},{},{:.8}",
            r.step,
            r.bit_position,
            r.entropy,
            r.is_constant,
            r.exact_num_bases,
            r.approx_num_bases,
            r.abs_error,
            r.relative_error
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn write_combined(path: &Path, per_image: &[Vec<StepRecord>]) -> Result<(), EntroGdError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "step,num_images,avg_entropy,avg_exact_num_bases,avg_approx_num_bases,avg_abs_error,avg_relative_error"
    )?;

    let max_steps = per_image.iter().map(|v| v.len()).max().unwrap_or(0);
    for step in 0..max_steps {
        let mut count = 0usize;
        let mut sum_entropy = 0.0f64;
        let mut sum_exact = 0.0f64;
        let mut sum_approx = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut sum_rel = 0.0f64;
        for records in per_image {
            if let Some(r) = records.get(step) {
                count += 1;
                sum_entropy += r.entropy;
                sum_exact += r.exact_num_bases as f64;
                sum_approx += r.approx_num_bases as f64;
                sum_abs += r.abs_error as f64;
                sum_rel += r.relative_error;
            }
        }
        if count == 0 {
            continue;
        }
        let n = count as f64;
        writeln!(
            writer,
            "{},{},{:.8},{:.8},{:.8},{:.8},{:.8}",
            step + 1,
            count,
            sum_entropy / n,
            sum_exact / n,
            sum_approx / n,
            sum_abs / n,
            sum_rel / n,
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn log_summary(input: &Path, output: &Path, records: &[StepRecord]) {
    let mut sum_abs_rel = 0.0f64;
    let mut sum_sq_rel = 0.0f64;
    let mut sum_rel = 0.0f64;
    let mut measured = 0usize;
    for r in records {
        if !r.is_constant {
            sum_abs_rel += r.relative_error.abs();
            sum_sq_rel += r.relative_error * r.relative_error;
            sum_rel += r.relative_error;
            measured += 1;
        }
    }

    if measured > 0 {
        let n = measured as f64;
        let mape = sum_abs_rel / n;
        let rmse = (sum_sq_rel / n).sqrt();
        let bias = sum_rel / n;
        tracing::info!(
            input = %input.display(),
            output = %output.display(),
            measured_steps = measured,
            mape,
            rmse,
            bias,
            "HLL precision evaluation completed"
        );
    } else {
        tracing::warn!(
            input = %input.display(),
            "No steps evaluated; check input data and --limit-bits"
        );
    }
}

fn output_path_for_image(output_dir: &Path, image_path: &Path) -> PathBuf {
    let stem = image_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    output_dir.join(format!("{}.csv", stem))
}

fn main() -> Result<(), EntroGdError> {
    let _log_handle = init_logging();
    let options = parse_args()?;

    if options.input.is_dir() {
        let image_paths = collect_image_paths(&options.input)?;
        if image_paths.is_empty() {
            return Err(EntroGdError::InvalidMetadata {
                message: format!("no images found in directory '{}'", options.input.display()),
            });
        }

        let output_dir = options.output.clone().unwrap_or_else(|| {
            let folder_name = options
                .input
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("input");
            PathBuf::from("target/basebit_precision").join(folder_name)
        });
        std::fs::create_dir_all(&output_dir)?;

        let mut per_image: Vec<Vec<StepRecord>> = Vec::with_capacity(image_paths.len());
        for image_path in &image_paths {
            let bit_data = load_bit_data(image_path, InputKind::Image, options.has_headers)?;
            let records = evaluate(&bit_data, options.limit_bits);
            let out_path = output_path_for_image(&output_dir, image_path);
            write_records(&out_path, &records)?;
            log_summary(image_path, &out_path, &records);
            per_image.push(records);
        }

        let combined_path = output_dir.join("combined.csv");
        write_combined(&combined_path, &per_image)?;
        tracing::info!(
            output = %combined_path.display(),
            images = per_image.len(),
            "Wrote combined averaged precision data"
        );
    } else {
        let output_path = options
            .output
            .clone()
            .unwrap_or_else(|| PathBuf::from("target/basebit_precision.csv"));
        let bit_data = load_bit_data(&options.input, options.input_kind, options.has_headers)?;
        let records = evaluate(&bit_data, options.limit_bits);
        write_records(&output_path, &records)?;
        log_summary(&options.input, &output_path, &records);
    }

    Ok(())
}
