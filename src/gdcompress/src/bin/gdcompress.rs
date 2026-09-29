use argh::FromArgs;
use gdcompress::prelude::*;
use gdcompress::{
    BitDataInfo, BitDataReconstructionInfo, CompressedData, CondensedSamples,
    DEFAULT_ALIGN_ROWS_TO_WORD, DecompressFileData, DecompressRandomAccessHandle, EntroGdError,
    FloatScalingMode, PixelGrouping, PreprocessOptions, load_csv, reconstruct_feature_value,
    reconstruct_to_dataframe, write_bitdata_as_image, write_cropped_bitdata_as_image,
};
use polars::prelude::{CsvWriter, SerWriter};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "gif", "tiff", "webp"];

#[derive(FromArgs)]
/// gdcompress — EntroGD compression/decompression tool
struct Args {
    /// input file to compress or decompress (auto-detect based on extension)
    #[argh(positional)]
    input: String,

    /// output path (auto-derived from input if omitted)
    #[argh(option, short = 'o')]
    output: Option<String>,

    /// show condensed samples (analytics) from a compressed file
    #[argh(switch)]
    analytics: bool,

    /// verbose output with detailed statistics
    #[argh(switch, short = 'v')]
    verbose: bool,

    /// pixel grouping size for image compression (default: "3x3")
    #[argh(option, default = "String::from(\"3x3\")")]
    pixel_grouping: String,

    /// convert floats to integers for compression: on or off (tabular only, default: on)
    #[argh(option, default = "String::from(\"on\")")]
    float_scaling: String,

    /// decimal places preserved when float scaling is on (tabular only, default: 9)
    #[argh(option, default = "9")]
    float_precision: u8,

    /// decompress only specific rows, e.g., "0,5,10-20" (tabular only)
    #[argh(option)]
    rows: Option<String>,

    /// base-bit counting: precise or approximate (default: precise)
    #[argh(option, default = "String::from(\"precise\")")]
    base_counting: String,

    /// encoding strategy: normal, rle, or huffman (default: normal for tabular, huffman for image)
    #[argh(option)]
    encoding: Option<String>,

    /// delta base-table compression: on or off (default: off for tabular, on for image)
    #[argh(option)]
    base_compression: Option<String>,

    /// crop region for image decompression: "x1,y1,x2,y2" (upper-left, lower-right inclusive; .igd only)
    #[argh(option)]
    crop: Option<String>,
}

#[derive(Debug)]
enum Action {
    CompressTabular {
        input: PathBuf,
        output: PathBuf,
        float_scaling: FloatScalingMode,
        float_precision: u8,
        base_impl: BaseBitImpl,
        encoding: EncodingMode,
        base_compression: bool,
    },
    CompressImage {
        input: PathBuf,
        output: PathBuf,
        pixel_grouping: PixelGrouping,
        base_impl: BaseBitImpl,
        encoding: EncodingMode,
        base_compression: bool,
    },
    DecompressTabular {
        input: PathBuf,
        output: PathBuf,
        rows: Option<Vec<usize>>,
    },
    DecompressImage {
        input: PathBuf,
        output: PathBuf,
        crop: Option<(u32, u32, u32, u32)>,
    },
    ShowAnalytics {
        input: PathBuf,
        output: PathBuf,
    },
}

#[derive(Debug, Clone, Copy)]
enum EncodingMode {
    Normal,
    Rle,
    Huffman,
}

fn parse_pixel_grouping(s: &str) -> Result<PixelGrouping, String> {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() != 2 {
        return Err(format!(
            "invalid pixel grouping '{}', expected format WxH (e.g., 3x3)",
            s
        ));
    }
    let w: u32 = parts[0]
        .parse()
        .map_err(|_| format!("invalid width in '{}'", s))?;
    let h: u32 = parts[1]
        .parse()
        .map_err(|_| format!("invalid height in '{}'", s))?;
    if w == 0 || h == 0 {
        return Err(format!(
            "pixel grouping dimensions must be positive, got {}",
            s
        ));
    }
    Ok(PixelGrouping::new(w, h))
}

fn parse_float_scaling(s: &str) -> Result<FloatScalingMode, String> {
    match s.to_lowercase().as_str() {
        "on" => Ok(FloatScalingMode::ScaledOffsetSignedInt),
        "off" => Ok(FloatScalingMode::Disabled),
        _ => Err(format!(
            "invalid float scaling '{}', expected: on or off",
            s
        )),
    }
}

fn parse_rows(s: &str) -> Result<Vec<usize>, String> {
    let mut indices = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some(pos) = part.find('-') {
            let start: usize = part[..pos]
                .trim()
                .parse()
                .map_err(|_| format!("invalid range start in '{}'", part))?;
            let end: usize = part[pos + 1..]
                .trim()
                .parse()
                .map_err(|_| format!("invalid range end in '{}'", part))?;
            if start > end {
                return Err(format!("invalid range '{}': start > end", part));
            }
            for i in start..=end {
                indices.push(i);
            }
        } else {
            let idx: usize = part
                .parse()
                .map_err(|_| format!("invalid index '{}'", part))?;
            indices.push(idx);
        }
    }
    if indices.is_empty() {
        return Err("no valid row indices provided".to_string());
    }
    indices.sort_unstable();
    indices.dedup();
    Ok(indices)
}

fn parse_base_counting(s: &str) -> Result<BaseBitImpl, String> {
    match s.to_lowercase().as_str() {
        "precise" => Ok(BaseBitImpl::Naive),
        "approximate" => Ok(BaseBitImpl::HyperLogLogCount),
        _ => Err(format!(
            "invalid base-counting '{}', expected: precise or approximate",
            s
        )),
    }
}

fn parse_crop(s: &str) -> Result<(u32, u32, u32, u32), String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        return Err(format!(
            "invalid crop '{}', expected format: x1,y1,x2,y2",
            s
        ));
    }
    let x1: u32 = parts[0]
        .trim()
        .parse()
        .map_err(|_| format!("invalid x1 in '{}'", s))?;
    let y1: u32 = parts[1]
        .trim()
        .parse()
        .map_err(|_| format!("invalid y1 in '{}'", s))?;
    let x2: u32 = parts[2]
        .trim()
        .parse()
        .map_err(|_| format!("invalid x2 in '{}'", s))?;
    let y2: u32 = parts[3]
        .trim()
        .parse()
        .map_err(|_| format!("invalid y2 in '{}'", s))?;
    if x1 > x2 || y1 > y2 {
        return Err(format!(
            "invalid crop '{}': x1 <= x2 and y1 <= y2 required",
            s
        ));
    }
    Ok((x1, y1, x2, y2))
}

fn parse_encoding(s: &str) -> Result<EncodingMode, String> {
    match s.to_lowercase().as_str() {
        "normal" => Ok(EncodingMode::Normal),
        "rle" => Ok(EncodingMode::Rle),
        "huffman" => Ok(EncodingMode::Huffman),
        _ => Err(format!(
            "invalid encoding '{}', expected: normal, rle, or huffman",
            s
        )),
    }
}

fn parse_base_compression(s: &str) -> Result<bool, String> {
    match s.to_lowercase().as_str() {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => Err(format!(
            "invalid base-compression '{}', expected: on or off",
            s
        )),
    }
}

fn derive_output(input: &Path, target_ext: &str) -> PathBuf {
    let stem = input.file_stem().unwrap_or_else(|| input.as_os_str());
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let mut name = stem.to_os_string();
    name.push(".");
    name.push(target_ext);
    parent.join(name)
}

fn detect_action(args: &Args) -> Result<Action, String> {
    let input = PathBuf::from(&args.input);
    if !input.exists() {
        return Err(format!("file not found: {}", input.display()));
    }

    let ext = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if args.analytics {
        if !matches!(ext.as_str(), "tgd" | "igd" | "egd") {
            return Err(format!(
                "--analytics requires a compressed file (.tgd, .igd, .egd), got '{}'",
                ext
            ));
        }
        let output = derive_output(&input, "analytics.csv");
        return Ok(Action::ShowAnalytics { input, output });
    }

    if args.crop.is_some() && ext.as_str() != "igd" {
        return Err("--crop is only valid for .igd image decompression".to_string());
    }

    match ext.as_str() {
        "csv" => {
            let output = args
                .output
                .clone()
                .map(PathBuf::from)
                .unwrap_or_else(|| derive_output(&input, "tgd"));
            let float_scaling = parse_float_scaling(&args.float_scaling)?;
            let float_precision = args.float_precision;
            let base_impl = parse_base_counting(&args.base_counting)?;
            let encoding = args
                .encoding
                .as_deref()
                .map(parse_encoding)
                .unwrap_or(Ok(EncodingMode::Normal))?;
            let base_compression = args
                .base_compression
                .as_deref()
                .map(parse_base_compression)
                .unwrap_or(Ok(false))?;
            Ok(Action::CompressTabular {
                input,
                output,
                float_scaling,
                float_precision,
                base_impl,
                encoding,
                base_compression,
            })
        }
        "tgd" => {
            let rows = args.rows.as_deref().map(parse_rows).transpose()?;
            let output = args
                .output
                .clone()
                .map(PathBuf::from)
                .unwrap_or_else(|| derive_output(&input, "csv"));
            Ok(Action::DecompressTabular {
                input,
                output,
                rows,
            })
        }
        "igd" => {
            let crop = args.crop.as_deref().map(parse_crop).transpose()?;
            let output = args
                .output
                .clone()
                .map(PathBuf::from)
                .unwrap_or_else(|| derive_output(&input, "png"));
            Ok(Action::DecompressImage { input, output, crop })
        }
        "egd" => {
            if args.output.is_none() {
                return Err(
                    "--output is required for .egd files (unknown original format)".to_string(),
                );
            }
            let output = PathBuf::from(args.output.as_ref().unwrap());
            // Try to infer output type from output extension
            let out_ext = output
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            match out_ext.as_str() {
                "csv" => Ok(Action::DecompressTabular {
                    input,
                    output,
                    rows: None,
                }),
                "png" | "jpg" | "jpeg" | "bmp" | "gif" => {
                    Ok(Action::DecompressImage { input, output, crop: None })
                }
                _ => Err(format!(
                    "cannot determine output format from extension '{}', use -o with .csv or .png",
                    out_ext
                )),
            }
        }
        ext if IMAGE_EXTENSIONS.contains(&ext) => {
            let output = args
                .output
                .clone()
                .map(PathBuf::from)
                .unwrap_or_else(|| derive_output(&input, "igd"));
            let pixel_grouping = parse_pixel_grouping(&args.pixel_grouping)?;
            let base_impl = parse_base_counting(&args.base_counting)?;
            let encoding = args
                .encoding
                .as_deref()
                .map(parse_encoding)
                .unwrap_or(Ok(EncodingMode::Huffman))?;
            let base_compression = args
                .base_compression
                .as_deref()
                .map(parse_base_compression)
                .unwrap_or(Ok(true))?;
            Ok(Action::CompressImage {
                input,
                output,
                pixel_grouping,
                base_impl,
                encoding,
                base_compression,
            })
        }
        _ => Err(format!(
            "unsupported file extension '{}'\n\
             supported input formats:\n\
               compress:   .csv, .png, .jpg, .jpeg, .bmp, .gif, .tiff, .webp\n\
               decompress: .tgd (tabular), .igd (image), .egd (generic, requires -o)",
            ext
        )),
    }
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.3} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn format_duration(d: std::time::Duration) -> String {
    if d.as_secs() >= 1 {
        format!("{:.2}s", d.as_secs_f64())
    } else if d.as_millis() >= 1 {
        format!("{}ms", d.as_millis())
    } else {
        format!("{}us", d.as_micros())
    }
}

fn compress_tabular(
    input: &Path,
    output: &Path,
    float_scaling: FloatScalingMode,
    float_precision: u8,
    base_impl: BaseBitImpl,
    encoding: EncodingMode,
    base_compression: bool,
    verbose: bool,
) -> Result<(), EntroGdError> {
    let total_start = Instant::now();

    eprint!("Loading... ");
    io::stderr().flush().unwrap();
    let load_start = Instant::now();
    let df = load_csv(input, true, None)?;
    let original_rows = df.height();
    let load_time = load_start.elapsed();
    eprintln!("done ({} rows, {} columns)", original_rows, df.width());

    if float_scaling != FloatScalingMode::Disabled {
        eprintln!(
            "warning: float scaling is on — floats are converted to integers,\n\
             preserving up to {} decimal place(s). Data loss may occur if your floats\n\
             need more precision. Tune with --float-precision or disable with\n\
             --float-scaling off. (Too high a value will be rejected if it causes\n\
             overflow.)",
            float_precision,
        );
    }

    let preprocess_options = PreprocessOptions {
        float_scaling,
        decimal_scale: float_precision,
        integer_zero_normalization: true,
    };

    eprint!("Compressing... ");
    io::stderr().flush().unwrap();
    let compress_start = Instant::now();
    let bit_data = BuildBitDataSet {
        options: preprocess_options,
        pad_rows_to_word: false,
    }
    .process(df)?;

    let base_pipeline = Entropy {}
        .then(GenCondensedSamples { m_max: 30 })
        .then(SelectBasesAdaptive {
            width_decay: 0.5,
            patience: 10,
            base_bit_impl: base_impl,
        })
        .then(BuildBaseTable {});

    let compressed = match encoding {
        EncodingMode::Normal => {
            let enc = base_pipeline.then(EncodeData {});
            if base_compression {
                enc.then(DeltaEncodeBaseTableFixed {}).process(bit_data)?
            } else {
                enc.process(bit_data)?
            }
        }
        EncodingMode::Rle => {
            let enc = base_pipeline.then(EncodeDataOffsetRLE {});
            if base_compression {
                enc.then(DeltaEncodeBaseTableFixed {}).process(bit_data)?
            } else {
                enc.process(bit_data)?
            }
        }
        EncodingMode::Huffman => {
            let enc = base_pipeline.then(EncodeDataHuffman {});
            if base_compression {
                enc.then(DeltaEncodeBaseTableFixed {}).process(bit_data)?
            } else {
                enc.process(bit_data)?
            }
        }
    };
    let compress_time = compress_start.elapsed();

    let save_start = Instant::now();
    SaveTgdFile {
        output_path: output.to_path_buf(),
    }
    .process(compressed.clone())?;
    let save_time = save_start.elapsed();

    let original_size = std::fs::metadata(input)?.len();
    let compressed_size = std::fs::metadata(output)?.len();
    let total_time = total_start.elapsed();
    eprintln!("done");

    println!("Compress  {} → {}", input.display(), output.display());
    println!(
        "  Original   {}  ({} bytes)",
        format_size(original_size),
        original_size
    );
    println!(
        "  Compressed {}  ({} bytes)",
        format_size(compressed_size),
        compressed_size
    );
    if original_size > 0 {
        let ratio = original_size as f64 / compressed_size as f64;
        let pct = 100.0 * compressed_size as f64 / original_size as f64;
        println!("  Ratio      {:.2}x  ({:.1}%)", ratio, pct);
    }
    println!("  Time       {}", format_duration(total_time));

    if verbose {
        let num_features = compressed.metadata.num_features();
        let num_base_bits = compressed.layout.selected_base_bit_positions().len();
        let base_table_entries = compressed.base_table.len();
        let encoded_bits = compressed.encoded_data.get_encoded_size();
        let original_bits = compressed.metadata.original_size_bits();

        println!();
        println!(
            "  Bit representation: {} bits, {} rows, {} features",
            original_bits, original_rows, num_features
        );
        println!("  ─────────────────────────────");
        println!("  Base bits selected: {}", num_base_bits);
        println!("  Base table entries: {}", base_table_entries);
        println!("  Encoded stream:     {} bits", encoded_bits);
        println!("  ─────────────────────────────");
        println!("  Timing:");
        println!("    Load      {}", format_duration(load_time));
        println!("    Compress  {}", format_duration(compress_time));
        println!("    Save      {}", format_duration(save_time));
        println!("    Total     {}", format_duration(total_time));
    }

    Ok(())
}

fn compress_image(
    input: &Path,
    output: &Path,
    pixel_grouping: PixelGrouping,
    base_impl: BaseBitImpl,
    encoding: EncodingMode,
    base_compression: bool,
    verbose: bool,
) -> Result<(), EntroGdError> {
    let total_start = Instant::now();

    eprint!("Loading... ");
    io::stderr().flush().unwrap();
    let load_start = Instant::now();
    let image = OpenImage {}.process(input.to_path_buf())?;
    let load_time = load_start.elapsed();
    let (img_w, img_h) = (image.width(), image.height());
    let raw_pixel_bytes = img_w as u64 * img_h as u64 * image.color().bytes_per_pixel() as u64;
    eprintln!("done ({}x{})", img_w, img_h);

    eprint!("Compressing... ");
    io::stderr().flush().unwrap();
    let compress_start = Instant::now();

    let base_pipeline = BuildImageBitDataSet {
        colorspace: ImageColorSpace::SrgbWithLinearAlpha,
        color_model: ImageColorModel::YCoCgR,
        pixel_grouping,
        grouping_transform: ImageGroupingTransform::ForFirstPixel,
        pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
    }
    .then(Entropy {})
    .then(SelectBasesAdaptive {
        width_decay: 0.45,
        patience: 5,
        base_bit_impl: base_impl,
    })
    .then(BuildSortedBaseTable {});

    let compressed = match encoding {
        EncodingMode::Normal => {
            let enc = base_pipeline.then(EncodeData {});
            if base_compression {
                enc.then(DeltaEncodeBaseTableFixed {}).process(image)?
            } else {
                enc.process(image)?
            }
        }
        EncodingMode::Rle => {
            let enc = base_pipeline.then(EncodeDataOffsetRLE {});
            if base_compression {
                enc.then(DeltaEncodeBaseTableFixed {}).process(image)?
            } else {
                enc.process(image)?
            }
        }
        EncodingMode::Huffman => {
            let enc = base_pipeline.then(EncodeDataHuffman {});
            if base_compression {
                enc.then(DeltaEncodeBaseTableFixed {}).process(image)?
            } else {
                enc.process(image)?
            }
        }
    };
    let compress_time = compress_start.elapsed();

    let save_start = Instant::now();
    SaveIgdFile {
        output_path: output.to_path_buf(),
    }
    .process(compressed.clone())?;
    let save_time = save_start.elapsed();

    let original_size = raw_pixel_bytes;
    let input_file_size = std::fs::metadata(input)?.len();
    let compressed_size = std::fs::metadata(output)?.len();
    let total_time = total_start.elapsed();
    eprintln!("done");

    println!("Compress  {} → {}", input.display(), output.display());
    println!(
        "  Raw        {}  ({} bytes)",
        format_size(original_size),
        original_size
    );
    println!(
        "  Compressed {}  ({} bytes)",
        format_size(compressed_size),
        compressed_size
    );
    if original_size > 0 {
        let ratio = original_size as f64 / compressed_size as f64;
        let pct = 100.0 * compressed_size as f64 / original_size as f64;
        println!("  Ratio      {:.2}x  ({:.1}%)", ratio, pct);
    }
    println!("  Time       {}", format_duration(total_time));

    if verbose {
        let num_features = compressed.metadata.num_features();
        let num_base_bits = compressed.layout.selected_base_bit_positions().len();
        let base_table_entries = compressed.base_table.len();
        let encoded_bits = compressed.encoded_data.get_encoded_size();
        let original_bits = compressed.metadata.original_size_bits();

        println!();
        println!(
            "  Grouping: {}x{} ({} pixels/block)",
            pixel_grouping.width(),
            pixel_grouping.height(),
            pixel_grouping.total_pixels()
        );
        println!(
            "  Bit representation: {} bits, {} features",
            original_bits, num_features
        );
        println!(
            "  Raw pixel data: {}  ({} bytes)",
            format_size(original_size),
            original_size
        );
        println!(
            "  Input file ({}): {}  ({} bytes)",
            input.extension().and_then(|e| e.to_str()).unwrap_or("?"),
            format_size(input_file_size),
            input_file_size
        );
        println!("  ─────────────────────────────");
        println!("  Base bits selected: {}", num_base_bits);
        println!("  Base table entries: {}", base_table_entries);
        println!("  Encoded stream:     {} bits", encoded_bits);
        println!("  ─────────────────────────────");
        println!("  Timing:");
        println!("    Load      {}", format_duration(load_time));
        println!("    Compress  {}", format_duration(compress_time));
        println!("    Save      {}", format_duration(save_time));
        println!("    Total     {}", format_duration(total_time));
    }

    Ok(())
}

fn decompress_tabular(
    input: &Path,
    output: &Path,
    rows: Option<Vec<usize>>,
    verbose: bool,
) -> Result<(), EntroGdError> {
    let total_start = Instant::now();

    eprint!("Loading... ");
    io::stderr().flush().unwrap();
    let load_start = Instant::now();
    let compressed_data = LoadTgdFile {}.process(input.to_path_buf())?;
    let load_time = load_start.elapsed();
    eprintln!("done");

    eprint!("Decompressing... ");
    io::stderr().flush().unwrap();
    let decompress_start = Instant::now();

    let bit_data = if let Some(ref indices) = rows {
        if indices.is_empty() {
            return Err(EntroGdError::DataLoad {
                message: "no rows to decompress".to_string(),
            });
        }
        let handle = DecompressRandomAccessHandle::new(compressed_data.clone())?;
        handle.decompress_samples(indices)?
    } else {
        DecompressFileData {}.process(compressed_data.clone())?
    };
    let decompress_time = decompress_start.elapsed();

    eprint!("Writing... ");
    io::stderr().flush().unwrap();
    let mut df = reconstruct_to_dataframe(&bit_data)?;
    let mut f = std::fs::File::create(output)?;
    CsvWriter::new(&mut f).finish(&mut df)?;
    eprintln!("done");

    let original_size = std::fs::metadata(input)?.len();
    let decompressed_size = std::fs::metadata(output)?.len();
    let total_time = total_start.elapsed();

    let num_rows = bit_data.data.num_rows;
    let num_features = bit_data.info.num_features();

    println!("Decompress  {} → {}", input.display(), output.display());
    println!("  Output   {} rows, {} features", num_rows, num_features);
    if rows.is_some() {
        println!("  Random access: {} row(s) extracted", num_rows);
    }
    println!(
        "  Size     {}  ({} bytes)",
        format_size(decompressed_size),
        decompressed_size
    );
    println!("  Time     {}", format_duration(total_time));

    if verbose {
        println!();
        println!(
            "  Compressed file: {}  ({} bytes)",
            format_size(original_size),
            original_size
        );
        println!("  ─────────────────────────────");
        println!("  Timing:");
        println!("    Load        {}", format_duration(load_time));
        println!("    Decompress  {}", format_duration(decompress_time));
        println!("    Total       {}", format_duration(total_time));
    }

    Ok(())
}

fn decompress_image(
    input: &Path,
    output: &Path,
    verbose: bool,
    crop: Option<(u32, u32, u32, u32)>,
) -> Result<(), EntroGdError> {
    let total_start = Instant::now();

    eprint!("Loading... ");
    io::stderr().flush().unwrap();
    let load_start = Instant::now();
    let compressed_data = LoadIgdFile {}.process(input.to_path_buf())?;
    let load_time = load_start.elapsed();
    eprintln!("done");

    let image_info = match &compressed_data.metadata.reconstruction {
        BitDataReconstructionInfo::Image(info) => *info,
        _ => {
            return Err(EntroGdError::InvalidMetadata {
                message: "expected image reconstruction info".to_string(),
            });
        }
    };

    let (decompress_time, num_rows_decompressed, crop_dims) = match crop {
        Some((x1, y1, x2, y2)) => {
            let group_w = image_info.pixel_grouping.width() as u32;
            let group_h = image_info.pixel_grouping.height() as u32;
            let grouped_width = image_info.width.div_ceil(group_w) as usize;
            let grouped_height = image_info.height.div_ceil(group_h) as usize;

            let gx_min = (x1 / group_w) as usize;
            let gx_max = (x2 / group_w) as usize;
            let gy_min = (y1 / group_h) as usize;
            let gy_max = (y2 / group_h) as usize;

            let mut indices: Vec<usize> = (gy_min..=gy_max)
                .flat_map(|gy| (gx_min..=gx_max).map(move |gx| gy * grouped_width + gx))
                .filter(|&i| i < grouped_width * grouped_height)
                .collect();
            indices.sort_unstable();
            indices.dedup();

            eprint!("Decompressing... ");
            io::stderr().flush().unwrap();
            let decompress_start = Instant::now();
            let handle = DecompressRandomAccessHandle::new(compressed_data.clone())?;
            let bit_data = handle.decompress_samples(&indices)?;
            let decompress_time = decompress_start.elapsed();

            eprint!("Writing... ");
            io::stderr().flush().unwrap();
            write_cropped_bitdata_as_image(&bit_data, &indices, output, x1, y1, x2, y2)?;
            eprintln!("done");

            let crop_w = (x2 - x1 + 1) as u32;
            let crop_h = (y2 - y1 + 1) as u32;
            (decompress_time, indices.len() as usize, Some((crop_w, crop_h)))
        }
        None => {
            eprint!("Decompressing... ");
            io::stderr().flush().unwrap();
            let decompress_start = Instant::now();
            let bit_data = DecompressFileData {}.process(compressed_data.clone())?;
            let decompress_time = decompress_start.elapsed();

            eprint!("Writing... ");
            io::stderr().flush().unwrap();
            write_bitdata_as_image(&bit_data, output)?;
            eprintln!("done");

            (decompress_time, bit_data.num_rows(), None)
        }
    };

    let original_size = std::fs::metadata(input)?.len();
    let decompressed_size = std::fs::metadata(output)?.len();
    let total_time = total_start.elapsed();

    println!("Decompress  {} → {}", input.display(), output.display());
    if let Some((w, h)) = crop_dims {
        println!("  Crop     {}x{} ({} grouped rows)", w, h, num_rows_decompressed);
    } else {
        println!(
            "  Size     {}  ({} bytes)",
            format_size(decompressed_size),
            decompressed_size
        );
    }
    println!("  Time     {}", format_duration(total_time));

    if verbose {
        println!();
        println!(
            "  Compressed file: {}  ({} bytes)",
            format_size(original_size),
            original_size
        );
        println!(
            "  Image: {}x{}, grouping {}x{}",
            image_info.width,
            image_info.height,
            image_info.pixel_grouping.width(),
            image_info.pixel_grouping.height()
        );
        println!("  ─────────────────────────────");
        println!("  Timing:");
        println!("    Load        {}", format_duration(load_time));
        println!("    Decompress  {}", format_duration(decompress_time));
        println!("    Total       {}", format_duration(total_time));
    }

    Ok(())
}

fn write_analytics_csv(
    path: &Path,
    metadata: &BitDataInfo,
    samples: &CondensedSamples,
) -> Result<(), EntroGdError> {
    let mut file = std::fs::File::create(path)?;
    let num_features = metadata.num_features();

    write!(file, "weight")?;
    let column_names: Vec<String> = match &metadata.reconstruction {
        BitDataReconstructionInfo::Tabular { column_names, .. } => column_names.clone(),
        BitDataReconstructionInfo::Image(_) => (0..num_features)
            .map(|i| format!("feature_{}", i))
            .collect(),
    };
    for name in &column_names {
        write!(file, ",{}", name)?;
    }
    writeln!(file)?;

    for (sample, &weight) in samples.samples.iter().zip(samples.weights.iter()) {
        write!(file, "{}", weight)?;
        for fi in 0..num_features {
            let feature_start = metadata.feature_offset(fi);
            let feature_end = feature_start + metadata.feature_bits(fi);
            if feature_end > sample.len() {
                write!(file, ",")?;
                continue;
            }
            let feature_bits = &sample[feature_start..feature_end];
            let value = reconstruct_feature_value(feature_bits, metadata.feature_spec(fi));
            write!(file, ",{}", value)?;
        }
        writeln!(file)?;
    }

    Ok(())
}

fn show_analytics(input: &Path, output: &Path, verbose: bool) -> Result<(), EntroGdError> {
    let ext = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    eprint!("Loading... ");
    io::stderr().flush().unwrap();
    let load_start = Instant::now();

    let compressed_data: CompressedData = match ext.as_str() {
        "tgd" => LoadTgdFile {}.process(input.to_path_buf())?,
        "igd" => LoadIgdFile {}.process(input.to_path_buf())?,
        "egd" => LoadEgdFile {}.process(input.to_path_buf())?,
        _ => unreachable!(),
    };
    let load_time = load_start.elapsed();

    eprint!("Extracting... ");
    io::stderr().flush().unwrap();
    let analytics_start = Instant::now();
    let analytics = DecompressAnalytics {}.process(compressed_data.clone())?;
    let analytics_time = analytics_start.elapsed();
    eprintln!("done");

    let total_time = load_time + analytics_time;

    match analytics {
        None => {
            println!("Analytics  {}", input.display());
            println!("  No condensed samples found in this file.");
            println!("  Time  {}", format_duration(total_time));
        }
        Some(samples) => {
            eprint!("Writing analytics... ");
            io::stderr().flush().unwrap();
            write_analytics_csv(output, &compressed_data.metadata, &samples)?;
            eprintln!("done ({} samples)", samples.samples.len());

            let num_features = compressed_data.metadata.num_features();
            let num_bases = compressed_data.base_table.len();

            println!("Analytics  {}", input.display());
            println!(
                "  {} condensed sample(s), {} feature(s), {} base table entr{}",
                samples.samples.len(),
                num_features,
                num_bases,
                if num_bases == 1 { "y" } else { "ies" }
            );

            if verbose {
                println!();
                let num_show = samples
                    .samples
                    .len()
                    .min(if num_features > 0 { 20 } else { 10 });
                println!("  Samples (showing {}):", num_show);
                for (i, (sample, weight)) in samples
                    .samples
                    .iter()
                    .zip(samples.weights.iter())
                    .take(num_show)
                    .enumerate()
                {
                    print!("    Sample {:>3}  weight={:<6}  [", i, weight);
                    for fi in 0..num_features {
                        let feature_start = compressed_data.metadata.feature_offset(fi);
                        let feature_end = feature_start + compressed_data.metadata.feature_bits(fi);
                        if feature_end > sample.len() {
                            print!("<?>");
                            continue;
                        }
                        let feature_bits = &sample[feature_start..feature_end];
                        let spec = compressed_data.metadata.feature_spec(fi);
                        let formatted = reconstruct_feature_value(feature_bits, spec).to_string();
                        if fi > 0 {
                            print!(", ");
                        }
                        print!("{}", formatted);
                    }
                    println!("]");
                }
                if samples.samples.len() > num_show {
                    println!("    ... and {} more", samples.samples.len() - num_show);
                }
            }

            if verbose {
                println!();
                println!("  ─────────────────────────────");
                println!("  Timing:");
                println!("    Load     {}", format_duration(load_time));
                println!("    Extract  {}", format_duration(analytics_time));
                println!("    Total    {}", format_duration(total_time));
            }
        }
    }

    Ok(())
}

fn main() {
    let args: Args = argh::from_env();

    let action = match detect_action(&args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let result = match action {
        Action::CompressTabular {
            input,
            output,
            float_scaling,
            float_precision,
            base_impl,
            encoding,
            base_compression,
        } => compress_tabular(
            &input,
            &output,
            float_scaling,
            float_precision,
            base_impl,
            encoding,
            base_compression,
            args.verbose,
        ),

        Action::CompressImage {
            input,
            output,
            pixel_grouping,
            base_impl,
            encoding,
            base_compression,
        } => compress_image(
            &input,
            &output,
            pixel_grouping,
            base_impl,
            encoding,
            base_compression,
            args.verbose,
        ),

        Action::DecompressTabular {
            input,
            output,
            rows,
        } => decompress_tabular(&input, &output, rows, args.verbose),

        Action::DecompressImage { input, output, crop } => {
            decompress_image(&input, &output, args.verbose, crop)
        }

        Action::ShowAnalytics { input, output } => show_analytics(&input, &output, args.verbose),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
