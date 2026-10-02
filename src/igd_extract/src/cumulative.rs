use super::*;

/// Exact integer rank plus cumulative quantized gaps, in row-major spatial order.
pub struct CumulativeU16 {
    pub width: usize,
    pub height: usize,
    pub z_u16: Vec<u16>,
    pub metadata_json: Vec<u8>,
}

fn cumulative_values(normalized: &[f64], scale: u16) -> Result<Vec<u16>> {
    let mut sum = 0u64;
    normalized.iter().enumerate().map(|(rank, &a)| {
        if !a.is_finite() || !(0.0..=1.0).contains(&a) {
            return Err(invalid("Invalid normalized delta."));
        }
        sum += (f64::from(scale) * a).round() as u64;
        let z = rank as u64 + sum;
        u16::try_from(z).map_err(|_| invalid(format!(
            "uint16 overflow: S={scale}, rank={rank}, cumulative={sum}, z={z} > 65535. Reduce --scale."
        )))
    }).collect()
}

pub fn extract_cumulative_u16(
    bytes: Vec<u8>,
    source: &str,
    options: Options,
    scale: u16,
) -> Result<CumulativeU16> {
    // Reuse the exact ordering and bit-layout interpretation of the baseline.
    // Intermediate data live only in memory, one image at a time.
    let document = extract_network_features(bytes, source, options)?;
    if document["image"]["pixel_group_width"] != 1 || document["image"]["pixel_group_height"] != 1 {
        return Err(invalid("cumulative-rank requires pixel grouping 1x1."));
    }
    let width = document["image"]["grid_width"]
        .as_u64()
        .ok_or_else(|| invalid("Missing width"))? as usize;
    let height = document["image"]["grid_height"]
        .as_u64()
        .ok_or_else(|| invalid("Missing height"))? as usize;
    let dictionary = document["sorted_dictionary"]
        .as_array()
        .ok_or_else(|| invalid("Missing dictionary"))?;
    let max_delta = document["max_delta_bits"]
        .as_str()
        .ok_or_else(|| invalid("Missing max delta"))?;
    let denominator = binary_log2(&add_one_binary(max_delta)?)?;
    let normalized: Vec<f64> = dictionary
        .iter()
        .map(|row| {
            let bits = row["delta_bits"]
                .as_str()
                .ok_or_else(|| invalid("Missing delta"))?;
            Ok(if denominator == 0.0 {
                0.0
            } else {
                binary_log2(&add_one_binary(bits)?)? / denominator
            })
        })
        .collect::<Result<_>>()?;
    let lookup = cumulative_values(&normalized, scale)?;
    let rows = document["spatial_rank_ids"]
        .as_array()
        .ok_or_else(|| invalid("Missing spatial ranks"))?;
    let mut z_u16 = Vec::with_capacity(width * height);
    for row in rows {
        for rank in row
            .as_array()
            .ok_or_else(|| invalid("Invalid spatial row"))?
        {
            let rank = rank.as_u64().ok_or_else(|| invalid("Invalid rank"))? as usize;
            z_u16.push(
                *lookup
                    .get(rank)
                    .ok_or_else(|| invalid("Rank outside dictionary"))?,
            );
        }
    }
    let z_max = *lookup.last().ok_or_else(|| invalid("Empty dictionary"))?;
    let metadata = json!({
        "schema": "igd-cumulative-rank-npz-v1",
        "extractor_version": env!("CARGO_PKG_VERSION"),
        "source": document["source"], "image": document["image"],
        "verification": document["verification"],
        "representation": {
            "name": "cumulative-rank-u16-v1", "scale": scale,
            "channel_order": ["z_u16"], "dtype": "uint16", "array_order": "C row-major; (height, width)",
            "base_order": "Ascending selected chunk positions, first bit treated as MSB; ascending numeric bases",
            "formula": "a_r=log2(1+d_r)/log2(1+d_max); q_r=round(S*a_r); c_r=sum(q_0..q_r); z_r=r+c_r",
            "rounding": "nearest integer, ties upward; before cumulative sum",
            "first_delta": 0, "zero_max_delta": "all normalized deltas are zero",
            "overflow": "reject; never clip or wrap", "training_suggestion": "float32(z_u16) / 65535; not applied in file"
        },
        "num_bases": dictionary.len(), "num_samples": z_u16.len(),
        "z_max": z_max, "cumulative_last": u64::from(z_max) - (dictionary.len() as u64 - 1),
        "base_min_bits": document["base_min_bits"], "base_max_bits": document["base_max_bits"],
        "max_delta_bits": document["max_delta_bits"]
    });
    Ok(CumulativeU16 {
        width,
        height,
        z_u16,
        metadata_json: serde_json::to_vec_pretty(&metadata)?,
    })
}

pub fn write_cumulative_npz<P: AsRef<Path>>(
    bytes: Vec<u8>,
    source: &str,
    options: Options,
    scale: u16,
    output_path: P,
) -> Result<NetworkNpzSummary> {
    // Validate all values before opening any output file.
    let data = extract_cumulative_u16(bytes, source, options, scale)?;
    let raw: Vec<u8> = data.z_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = npy_2d(data.width, data.height, &raw, "<u2", 2)?;
    let path = output_path.as_ref();
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    write_stored_npz(
        &mut writer,
        vec![
            ZipEntry {
                name: "z_u16.npy",
                bytes: npy,
            },
            ZipEntry {
                name: "metadata.json",
                bytes: data.metadata_json,
            },
        ],
    )?;
    writer.flush()?;
    Ok(NetworkNpzSummary {
        width: data.width,
        height: data.height,
        tensor_bytes: raw.len(),
        file_bytes: std::fs::metadata(path)?.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_cumulative_and_boundaries() {
        assert_eq!(
            cumulative_values(&[0.0, 1.0, 0.2, 1.0], 10).unwrap(),
            [0, 11, 14, 25]
        );
        assert_eq!(
            cumulative_values(&[0.0, 0.49, 0.5, 1.0], 1).unwrap(),
            [0, 1, 3, 5]
        );
        assert_eq!(cumulative_values(&[0.0], 65535).unwrap(), [0]);
        assert_eq!(cumulative_values(&[0.0, 1.0], 65534).unwrap(), [0, 65535]);
        assert!(cumulative_values(&[0.0, 1.0], 65535).is_err());
        assert_eq!(cumulative_values(&[0.0, 0.5, 1.0], 0).unwrap(), [0, 1, 2]);
    }
}
