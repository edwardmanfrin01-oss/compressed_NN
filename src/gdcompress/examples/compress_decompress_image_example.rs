use gdcompress::ImageColorModel;
use gdcompress::ScopedTimer;

use gdcompress::compression::data::DEFAULT_ALIGN_ROWS_TO_WORD;
use gdcompress::prelude::*;
use gdcompress::{EntroGdError, init_logging, write_bitdata_as_image};
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::Path;

fn main() -> Result<(), EntroGdError> {
    unsafe {
        env::set_var("ENTRO_GD_LOG_TO_STDERR", "1");
        env::set_var("RUST_LOG", "info");
    }
    let _log_handle = init_logging();
    let _timer =
        ScopedTimer::info("Total processing time for compressing and decompressing image(s)");
    let args: Vec<String> = env::args().collect();
    let input_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "data/images/rustacean.png".to_string()
    };

    let input = Path::new(&input_path);

    let files_to_process = if input.is_dir() {
        // If it's a directory, collect all image files
        let mut files = Vec::new();
        for entry in fs::read_dir(input)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file()
                && let Some(ext) = path.extension()
            {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if matches!(ext_str.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "gif") {
                    files.push(path);
                }
            }
        }
        files.sort();
        files
    } else {
        // If it's a file, process just that file
        vec![input.to_path_buf()]
    };

    if files_to_process.is_empty() {
        tracing::warn!("No image files found to process");
        return Ok(());
    }
    let image_opener = OpenImage {};
    let images = files_to_process
        .iter()
        .map(|file| {
            tracing::info!("Loading image: {}", file.display());
            image_opener.process(file.to_path_buf())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let data_folder = Path::new("data");
    let compressed_folder = data_folder.join("compressed");
    let decompressed_folder = data_folder.join("decompressed");
    let base_debug_folder = data_folder.join("base_selection_debug");
    let _base_selection_debug_csv_paths = files_to_process.iter().map(|file| {
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image_dataset");
        base_debug_folder.join(format!("{}_base_selection_debug.csv", stem))
    });

    fs::create_dir_all(&compressed_folder)?;
    fs::create_dir_all(&decompressed_folder)?;

    let compression_pipeline = BuildImageBitDataSet {
        colorspace: ImageColorSpace::SrgbWithLinearAlpha,
        color_model: ImageColorModel::YCoCgR,
        pixel_grouping: PixelGrouping::new(2, 2),
        grouping_transform: ImageGroupingTransform::ForFirstPixel,
        pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
    }
    .then(Entropy {})
    .then(SelectBasesAdaptive {
        width_decay: 0.45,
        patience: 5,
        base_bit_impl: BaseBitImpl::HyperLogLogCount,
    })
    .then(BuildSortedBaseTable {})
    .then(EncodeDataHuffman {})
    .then(DeltaEncodeBaseTableFixed {});

    let load_compressed_data = LoadIgdFile {};

    for (image_file, image) in files_to_process.into_iter().zip(images.into_iter()) {
        let stem = image_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image_dataset");

        let igd_path = compressed_folder.join(format!("{}.igd", stem));
        let decompressed_path = decompressed_folder.join(format!("{}-decompressed.png", stem));

        tracing::info!("IGD output: {}", igd_path.display());
        tracing::info!("Decoded image output: {}", decompressed_path.display());

        let compressed = compression_pipeline.process(image.clone())?;
        let compressed_path = SaveIgdFile {
            output_path: igd_path.clone(),
        }
        .process(compressed.clone())?;

        let compressed_data = load_compressed_data.process(compressed_path.clone())?;
        let bit_data = DecompressFileData {}
            .process(compressed_data.clone())
            .unwrap();
        write_bitdata_as_image(&bit_data, &decompressed_path)?;
        let cloned_compressed_data = compressed_data.clone();
        let _timer = ScopedTimer::info("Decompression time for random access handle");
        let decompress_handle = DecompressRandomAccessHandle::new(cloned_compressed_data)?;
        let _decompressed_data = decompress_handle.decompress_samples(&vec![
            compressed_data.encoded_data.get_num_samples() / 2 as usize,
        ])?;
        black_box(decompress_handle);
        drop(_timer);

        tracing::info!(
            "Compression done: original={} bits, encoded={} bits",
            compressed.metadata.original_size_bits(),
            compressed.encoded_data.get_encoded_size()
        );
        let igd_size_bytes = fs::metadata(&compressed_path)?.len();
        let igd_size_bits = igd_size_bytes.saturating_mul(8);
        let original_size_bits = compressed.metadata.original_size_bits();

        if original_size_bits > 0 {
            tracing::info!(
                "Compression ratio (stored .igd size): {:.2}% ({} bytes)",
                100.0 * igd_size_bits as f64 / original_size_bits as f64,
                igd_size_bytes
            );
        } else {
            tracing::info!(
                "Compression ratio (stored .igd size): n/a (original size is 0 bits, .igd={} bytes)",
                igd_size_bytes
            );
        }
        tracing::info!("Completed processing: {}", image_file.display());
    }

    tracing::info!("Done.");
    Ok(())
}
