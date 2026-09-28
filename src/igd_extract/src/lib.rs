//! Export IGD dictionaries and spatial base IDs without reconstructing RGB pixels.
//! Bit positions always refer to the transformed chunk, not the RGB byte stream.

use bitvec::prelude::*;
use gdcompress::compression::base_table::BaseBitLayoutState;
use gdcompress::compression::decompression::decompress_file;
use gdcompress::compression::encoding::{BaseTable, EncodedData};
use gdcompress::{BitDataReconstructionInfo, FeatureTransform, IgdFile};
use serde_json::{Value, json};
use std::error::Error;
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Default, Clone, Copy)]
pub struct Options {
    pub include_deviations: bool,
    /// Reconstruct transformed chunks from the exported representation and compare
    /// every bit with gdcompress's full bit-data decoder before writing the JSON.
    pub verify: bool,
}

/// The two uint8 channels consumed by the first compressed-domain ResNet.
/// The data are row-major, with index `y * width + x`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkU8 {
    pub width: usize,
    pub height: usize,
    pub rank_u8: Vec<u8>,
    pub delta_u8: Vec<u8>,
    pub metadata_json: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetworkNpzSummary {
    pub width: usize,
    pub height: usize,
    pub tensor_bytes: usize,
    pub file_bytes: u64,
}

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

fn bit_string(bits: &BitSlice<usize, Lsb0>) -> String {
    bits.iter().map(|b| if *b { '1' } else { '0' }).collect()
}

/// Compare equal-width binary strings as unsigned integers. The first character
/// is the most-significant bit of the experimental base-value convention.
fn compare_binary(left: &str, right: &str) -> std::cmp::Ordering {
    debug_assert_eq!(left.len(), right.len());
    left.as_bytes().cmp(right.as_bytes())
}

/// `left - right`, for equal-width unsigned binary strings where `left >= right`.
/// Leading zeroes are kept so a delta has the same bit width as its base.
fn subtract_binary(left: &str, right: &str) -> Result<String> {
    if left.len() != right.len() || compare_binary(left, right).is_lt() {
        return Err(invalid(
            "Sottrazione binaria non valida per le basi ordinate.",
        ));
    }
    let mut result = vec![b'0'; left.len()];
    let mut borrow = 0u8;
    for index in (0..left.len()).rev() {
        let lhs = left.as_bytes()[index]
            .checked_sub(b'0')
            .filter(|bit| *bit <= 1)
            .ok_or_else(|| invalid("Bit non binario in una base."))?;
        let rhs = right.as_bytes()[index]
            .checked_sub(b'0')
            .filter(|bit| *bit <= 1)
            .ok_or_else(|| invalid("Bit non binario in una base."))?;
        let subtrahend = rhs + borrow;
        if lhs >= subtrahend {
            result[index] = b'0' + (lhs - subtrahend);
            borrow = 0;
        } else {
            result[index] = b'0' + (lhs + 2 - subtrahend);
            borrow = 1;
        }
    }
    if borrow != 0 {
        return Err(invalid("Prestito residuo nella sottrazione binaria."));
    }
    String::from_utf8(result).map_err(Into::into)
}

fn add_one_binary(bits: &str) -> Result<String> {
    let mut result = bits.as_bytes().to_vec();
    let mut carry = 1u8;
    for value in result.iter_mut().rev() {
        let bit = value
            .checked_sub(b'0')
            .filter(|bit| *bit <= 1)
            .ok_or_else(|| invalid("Bit non binario in un delta."))?;
        let sum = bit + carry;
        *value = b'0' + (sum & 1);
        carry = sum >> 1;
        if carry == 0 {
            break;
        }
    }
    if carry == 1 {
        result.insert(0, b'1');
    }
    String::from_utf8(result).map_err(Into::into)
}

/// Approximate `log2(value)` from an arbitrarily long unsigned binary string.
/// The original integer remains in the JSON as `delta_bits`; this is only the
/// bounded f64 channel supplied to a neural network.
fn binary_log2(bits: &str) -> Result<f64> {
    let significant = bits.trim_start_matches('0');
    if significant.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }
    let prefix_len = significant.len().min(53);
    let mut prefix = 0u64;
    for byte in significant.bytes().take(prefix_len) {
        if !matches!(byte, b'0' | b'1') {
            return Err(invalid("Bit non binario in un delta."));
        }
        prefix = (prefix << 1) | u64::from(byte == b'1');
    }
    let mantissa = prefix as f64 / 2f64.powi((prefix_len - 1) as i32);
    Ok((significant.len() - 1) as f64 + mantissa.log2())
}

fn matrix<T: Clone>(values: &[T], width: usize, height: usize) -> Result<Vec<Vec<T>>> {
    if values.len() != width.saturating_mul(height) {
        return Err(invalid(
            "Lunghezza della mappa diversa dalla griglia dell'immagine.",
        ));
    }
    Ok(values.chunks(width).map(|row| row.to_vec()).collect())
}

/// Quantize a value already normalized in [0, 1] to all 256 uint8 levels.
/// The inverse used by the training loader is approximately `byte / 255.0`.
fn quantize_unit(value: f64) -> Result<u8> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(invalid("Valore normalizzato fuori dall'intervallo [0, 1]."));
    }
    Ok((value * 255.0).round() as u8)
}

fn u8_matrix(document: &Value, key: &str, width: usize, height: usize) -> Result<Vec<u8>> {
    let rows = document[key]
        .as_array()
        .ok_or_else(|| invalid(format!("Mappa {key} mancante.")))?;
    if rows.len() != height {
        return Err(invalid(format!("Altezza non valida per la mappa {key}.")));
    }
    let mut values = Vec::with_capacity(width * height);
    for row in rows {
        let row = row
            .as_array()
            .ok_or_else(|| invalid(format!("Riga non valida per la mappa {key}.")))?;
        if row.len() != width {
            return Err(invalid(format!("Larghezza non valida per la mappa {key}.")));
        }
        for value in row {
            values.push(quantize_unit(value.as_f64().ok_or_else(|| {
                invalid(format!("Valore non numerico nella mappa {key}."))
            })?)?);
        }
    }
    Ok(values)
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn npy_u8_2d(width: usize, height: usize, values: &[u8]) -> Result<Vec<u8>> {
    if values.len()
        != width
            .checked_mul(height)
            .ok_or_else(|| invalid("Dimensioni NPY troppo grandi."))?
    {
        return Err(invalid(
            "Lunghezza del canale diversa dalle dimensioni NPY.",
        ));
    }
    let mut header =
        format!("{{'descr': '|u1', 'fortran_order': False, 'shape': ({height}, {width}), }}")
            .into_bytes();
    // NPY v1.0 header: magic(6) + version(2) + length(2) + header must be a multiple of 16.
    let padding = (16 - ((10 + header.len() + 1) % 16)) % 16;
    header.extend(std::iter::repeat_n(b' ', padding));
    header.push(b'\n');
    let header_len =
        u16::try_from(header.len()).map_err(|_| invalid("Header NPY troppo lungo."))?;
    let mut output = Vec::with_capacity(10 + header.len() + values.len());
    output.extend_from_slice(b"\x93NUMPY");
    output.extend_from_slice(&[1, 0]);
    output.extend_from_slice(&header_len.to_le_bytes());
    output.extend_from_slice(&header);
    output.extend_from_slice(values);
    Ok(output)
}

struct ZipEntry {
    name: &'static str,
    bytes: Vec<u8>,
}

/// Minimal ZIP writer for a standard, uncompressed NumPy `.npz` archive.
/// NumPy accepts stored ZIP entries, so this adds no dependency or hidden data
/// conversion. The tensors themselves remain exact uint8 bytes.
fn write_stored_npz(mut writer: impl Write, entries: Vec<ZipEntry>) -> Result<()> {
    let mut central = Vec::new();
    let mut offset = 0u32;
    let entry_count = u16::try_from(entries.len()).map_err(|_| invalid("Troppe entry NPZ."))?;
    for entry in entries {
        let name = entry.name.as_bytes();
        let size =
            u32::try_from(entry.bytes.len()).map_err(|_| invalid("Entry NPZ troppo grande."))?;
        let name_len =
            u16::try_from(name.len()).map_err(|_| invalid("Nome entry NPZ troppo lungo."))?;
        let crc = crc32(&entry.bytes);
        writer.write_all(&0x0403_4B50u32.to_le_bytes())?;
        writer.write_all(&20u16.to_le_bytes())?;
        writer.write_all(&0u16.to_le_bytes())?;
        writer.write_all(&0u16.to_le_bytes())?;
        writer.write_all(&0u16.to_le_bytes())?;
        writer.write_all(&0u16.to_le_bytes())?;
        writer.write_all(&crc.to_le_bytes())?;
        writer.write_all(&size.to_le_bytes())?;
        writer.write_all(&size.to_le_bytes())?;
        writer.write_all(&name_len.to_le_bytes())?;
        writer.write_all(&0u16.to_le_bytes())?;
        writer.write_all(name)?;
        writer.write_all(&entry.bytes)?;

        central.extend_from_slice(&0x0201_4B50u32.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&size.to_le_bytes());
        central.extend_from_slice(&size.to_le_bytes());
        central.extend_from_slice(&name_len.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u32.to_le_bytes());
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name);

        offset = offset
            .checked_add(30 + u32::from(name_len) + size)
            .ok_or_else(|| invalid("Overflow dimensione archivio NPZ."))?;
    }
    let central_offset = offset;
    let central_size =
        u32::try_from(central.len()).map_err(|_| invalid("Indice NPZ troppo grande."))?;
    writer.write_all(&central)?;
    writer.write_all(&0x0605_4B50u32.to_le_bytes())?;
    writer.write_all(&0u16.to_le_bytes())?;
    writer.write_all(&0u16.to_le_bytes())?;
    writer.write_all(&entry_count.to_le_bytes())?;
    writer.write_all(&entry_count.to_le_bytes())?;
    writer.write_all(&central_size.to_le_bytes())?;
    writer.write_all(&central_offset.to_le_bytes())?;
    writer.write_all(&0u16.to_le_bytes())?;
    Ok(())
}

fn feature_transform(transform: FeatureTransform) -> Value {
    match transform {
        FeatureTransform::None => json!({"kind": "none"}),
        FeatureTransform::ScaledSignedInt { decimal_scale } => {
            json!({"kind": "scaled_signed_int", "decimal_scale": decimal_scale})
        }
        FeatureTransform::OffsetSignedInt { min_value } => {
            json!({"kind": "offset_signed_int", "min_value": min_value})
        }
        FeatureTransform::OffsetUnsignedInt { min_value } => {
            json!({"kind": "offset_unsigned_int", "min_value": min_value})
        }
        FeatureTransform::ScaledOffsetSignedInt {
            decimal_scale,
            min_value,
        } => json!({"kind": "scaled_offset_signed_int", "decimal_scale": decimal_scale,
                "min_value": min_value}),
    }
}

/// Load the existing IGD/EGD codecs, restore the dictionary's column order,
/// and export the final positional mapping. No selection history is invented.
pub fn extract(bytes: Vec<u8>, source: &str, options: Options) -> Result<Value> {
    if bytes.len() < 32 || &bytes[..3] != b"IGD" {
        return Err(invalid(
            "File non IGD o header incompleto (minimo 32 byte).",
        ));
    }
    let payload_len = u64::from_le_bytes(bytes[24..32].try_into()?);
    if payload_len != (bytes.len() - 32) as u64 || payload_len < 28 {
        return Err(invalid(
            "Lunghezza IGD non valida: possibile file troncato.",
        ));
    }
    let igd_version = bytes[3];
    let egd_version = bytes[35];
    let source_bytes = bytes.len();
    let mut compressed = IgdFile::from_bytes(bytes).to_compressed_data()?;
    let info = match &compressed.metadata.reconstruction {
        BitDataReconstructionInfo::Image(info) => *info,
        _ => return Err(invalid("Metadati immagine mancanti.")),
    };
    // Condensation stores unique samples and weights, not their original spatial
    // ordering. Never label that list as an image's spatial ID map.
    if compressed.condensed_sample_weights.is_some() {
        return Err(invalid(
            "Campioni condensati: non e' disponibile una mappa spaziale affidabile.",
        ));
    }
    let base_encoding = match &compressed.base_table {
        BaseTable::Raw(_) => "raw",
        BaseTable::Delta(delta) => match delta.codec_id {
            1 => "delta_unary",
            2 => "delta_fixed",
            _ => return Err(invalid("Codec del dizionario non supportato.")),
        },
    };
    if let BaseTable::Delta(delta) = &compressed.base_table {
        compressed.base_table = BaseTable::Raw(delta.decode_rows()?);
    }
    let sample_encoding = match &compressed.encoded_data {
        EncodedData::Normal(_) => "normal",
        EncodedData::RleOffset(_) => "rle_offset",
        EncodedData::Huffman(_) => "huffman",
    };
    // Decode the stream once. Repeated get_sample on the compressed stream may
    // repeatedly scan a row for RLE/Huffman. These public conversions return errors.
    let samples = match &compressed.encoded_data {
        EncodedData::Normal(data) => data.clone(),
        EncodedData::RleOffset(data) => data.to_deviation_data()?,
        EncodedData::Huffman(data) => data.to_deviation_data()?,
    };
    let chunk_bits = compressed.metadata.chunk_size();
    let base_positions = compressed.layout.selected_base_bit_positions();
    let variable_positions = compressed.layout.variable_base_bit_positions();
    let deviation_positions = compressed.layout.deviation_bit_positions();
    let zero_positions = compressed.layout.constant_zero_bit_positions();
    let one_positions = compressed.layout.constant_one_bit_positions();
    let grid_width = info.width.div_ceil(info.pixel_grouping.width()) as usize;
    let grid_height = info.height.div_ceil(info.pixel_grouping.height()) as usize;
    let count = grid_width
        .checked_mul(grid_height)
        .ok_or_else(|| invalid("Dimensioni della griglia troppo grandi."))?;
    if samples.get_num_samples() != count || compressed.metadata.n_data_samples() != count {
        return Err(invalid(
            "Numero di campioni diverso dalla griglia dei gruppi di pixel.",
        ));
    }
    if compressed.layout.chunk_size() != chunk_bits
        || samples.get_num_deviation_bits() != deviation_positions.len()
        || base_positions.len() + deviation_positions.len() != chunk_bits
    {
        return Err(invalid("Layout dei bit incoerente con i campioni."));
    }

    // Rows contain ONLY variable bits, already restored to ascending chunk-position
    // order by decode_rows(). Add constants explicitly to each exported full base.
    let table = compressed.base_table.as_raw();
    let mut bases = Vec::with_capacity(table.len());
    for (id, (variable_bits, _)) in table.iter().enumerate() {
        if variable_bits.len() != variable_positions.len() {
            return Err(invalid(format!("Lunghezza non valida per la base {id}.")));
        }
        let mut variable_index = 0;
        let mut full_bits = String::with_capacity(base_positions.len());
        for &position in &base_positions {
            let bit = match compressed.layout.state_at(position) {
                BaseBitLayoutState::Variable => {
                    let value = variable_bits[variable_index];
                    variable_index += 1;
                    value
                }
                BaseBitLayoutState::ConstantZero => false,
                BaseBitLayoutState::ConstantOne => true,
                BaseBitLayoutState::Deviation => {
                    return Err(invalid("Base contenente una deviazione."));
                }
            };
            full_bits.push(if bit { '1' } else { '0' });
        }
        bases.push(json!({
            "id": id,
            "bits": full_bits,
            "variable_bits": bit_string(variable_bits),
            "frequency": 0
        }));
    }
    let mut ids = Vec::with_capacity(count);
    let mut frequencies = vec![0usize; bases.len()];
    let keep_deviations = options.include_deviations || options.verify;
    let mut deviations = Vec::new();
    if keep_deviations {
        deviations.reserve(count);
    }
    for i in 0..count {
        let sample = samples
            .get_sample(i)
            .ok_or_else(|| invalid(format!("Campione {i} non decodificabile.")))?;
        if sample.id.len() > usize::BITS as usize {
            return Err(invalid("ID troppo lungo per questa piattaforma."));
        }
        let id = if sample.id.is_empty() {
            0
        } else {
            sample.id.load_le::<usize>()
        };
        let frequency = frequencies
            .get_mut(id)
            .ok_or_else(|| invalid(format!("ID {id} fuori dal dizionario, campione {i}.")))?;
        *frequency += 1;
        ids.push(id);
        if keep_deviations {
            deviations.push(bit_string(&sample.deviation));
        }
    }
    // Recount IDs: delta-decoded rows do not carry reliable stored frequencies.
    for (base, frequency) in bases.iter_mut().zip(frequencies) {
        base["frequency"] = json!(frequency);
    }
    let mut offset = 0usize;
    let features: Vec<Value> = compressed
        .metadata
        .features()
        .iter()
        .enumerate()
        .map(|(index, feature)| {
            let width = feature.data_type.bits();
            let result = json!({
                "index": index, "chunk_bit_offset": offset, "bit_count": width,
                "data_type": format!("{:?}", feature.data_type),
                "transform": feature_transform(feature.transform)
            });
            offset += width;
            result
        })
        .collect();
    if offset != chunk_bits {
        return Err(invalid("Dimensioni delle feature incoerenti."));
    }
    let mut document = json!({
        "schema": "igd-extract-v1",
        "source": {"path": source, "size_bytes": source_bytes,
            "igd_version": igd_version, "egd_version": egd_version,
            "base_table_encoding": base_encoding, "sample_encoding": sample_encoding},
        "conventions": {
            "index_origin": 0,
            "bit_strings": "Character k is bit k of the corresponding positions list; NOT an MSB-first integer.",
            "base_mapping": "bases[id].bits[k] belongs at transformed_chunk[base_bit_positions[k]].",
            "variable_mapping": "bases[id].variable_bits[k] belongs at transformed_chunk[variable_base_bit_positions[k]].",
            "deviation_mapping": "sample_deviation_bits[i][k] belongs at transformed_chunk[deviation_bit_positions[k]].",
            "sample_order": "Row-major pixel groups: i = group_y * grid_width + group_x.",
            "feature_order": "Channels in metadata order; grouped values inside each channel; field bits least-significant first.",
            "selection_history": "Not stored in IGD. The exported positions are ascending spatial bit positions, not chronological choices."
        },
        "image": {
            "width": info.width, "height": info.height, "channels": info.channels,
            "color_model": format!("{:?}", info.color_model),
            "color_model_code": info.color_model.as_u8(),
            "colorspace_code": info.colorspace,
            "grouping_transform": format!("{:?}", info.grouping_transform),
            "grouping_transform_code": info.grouping_transform.as_u8(),
            "pixel_group_width": info.pixel_grouping.width(),
            "pixel_group_height": info.pixel_grouping.height(),
            "grid_width": grid_width, "grid_height": grid_height
        },
        "chunk_bits": chunk_bits,
        "features": features,
        "num_samples": count,
        "num_bases": bases.len(),
        "base_bits": base_positions.len(),
        "variable_base_bits": variable_positions.len(),
        "deviation_bits": deviation_positions.len(),
        "base_bit_positions": base_positions,
        "variable_base_bit_positions": variable_positions,
        "constant_zero_bit_positions": zero_positions,
        "constant_one_bit_positions": one_positions,
        "deviation_bit_positions": deviation_positions,
        "selection_order": null,
        "bases": bases,
        "sample_base_ids": ids,
        "includes_deviations": options.include_deviations,
        "sample_deviation_bits": if keep_deviations { json!(deviations) } else { Value::Null },
        "verification": {"performed": false, "method": "full_transformed_chunks_against_gdcompress"}
    });
    if options.verify {
        let reference = decompress_file(compressed)?;
        for i in 0..count {
            let rebuilt = reconstruct_chunk(&document, i)?;
            let start = i * reference.data.stride;
            if rebuilt.as_bitslice() != &reference.data.data[start..start + chunk_bits] {
                return Err(invalid(format!(
                    "Verifica fallita: bit differenti nel campione {i}."
                )));
            }
        }
        document["verification"] = json!({
            "performed": true, "passed": true, "samples_checked": count,
            "method": "full_transformed_chunks_against_gdcompress"
        });
    }
    if !options.include_deviations {
        document["sample_deviation_bits"] = Value::Null;
    }
    Ok(document)
}

/// Produce the first neural-network representation directly from an IGD file.
///
/// The dictionary is sorted by its fixed-width base-bit string. This defines an
/// experimental unsigned-integer ordering without converting long bases to a
/// lossy machine integer. Every spatial sample then receives its new rank and
/// the delta attached to that rank.
pub fn extract_network_features(bytes: Vec<u8>, source: &str, options: Options) -> Result<Value> {
    // This is intentionally an in-memory conversion: no generic JSON is written
    // to disk before the smaller, experiment-specific document is produced.
    let extracted = extract(bytes, source, options)?;
    let base_bits = extracted["base_bits"]
        .as_u64()
        .ok_or_else(|| invalid("base_bits mancante."))? as usize;
    let num_bases = extracted["num_bases"]
        .as_u64()
        .ok_or_else(|| invalid("num_bases mancante."))? as usize;
    let grid_width = extracted["image"]["grid_width"]
        .as_u64()
        .ok_or_else(|| invalid("grid_width mancante."))? as usize;
    let grid_height = extracted["image"]["grid_height"]
        .as_u64()
        .ok_or_else(|| invalid("grid_height mancante."))? as usize;
    let raw_bases = extracted["bases"]
        .as_array()
        .ok_or_else(|| invalid("Dizionario delle basi mancante."))?;
    if raw_bases.len() != num_bases || num_bases == 0 {
        return Err(invalid("Numero di basi non valido."));
    }

    let mut ordered: Vec<(usize, String, usize)> = Vec::with_capacity(num_bases);
    for base in raw_bases {
        let old_id = base["id"]
            .as_u64()
            .ok_or_else(|| invalid("ID originale mancante."))? as usize;
        let bits = base["bits"]
            .as_str()
            .ok_or_else(|| invalid("Bit della base mancanti."))?
            .to_owned();
        let frequency = base["frequency"]
            .as_u64()
            .ok_or_else(|| invalid("Frequenza della base mancante."))?
            as usize;
        if old_id >= num_bases
            || bits.len() != base_bits
            || !bits.bytes().all(|b| matches!(b, b'0' | b'1'))
        {
            return Err(invalid("Base non valida nel dizionario."));
        }
        ordered.push((old_id, bits, frequency));
    }
    ordered.sort_by(|left, right| compare_binary(&left.1, &right.1));
    if ordered.windows(2).any(|pair| pair[0].1 == pair[1].1) {
        return Err(invalid("Il dizionario contiene basi duplicate."));
    }

    let mut old_to_rank = vec![None; num_bases];
    let mut deltas = Vec::with_capacity(num_bases);
    let mut sorted_dictionary = Vec::with_capacity(num_bases);
    for (rank, (old_id, bits, frequency)) in ordered.iter().enumerate() {
        let slot = old_to_rank
            .get_mut(*old_id)
            .ok_or_else(|| invalid("ID originale fuori intervallo."))?;
        if slot.replace(rank).is_some() {
            return Err(invalid("ID originale duplicato."));
        }
        let delta = if rank == 0 {
            "0".repeat(base_bits)
        } else {
            subtract_binary(bits, &ordered[rank - 1].1)?
        };
        deltas.push(delta.clone());
        sorted_dictionary.push(json!({
            "rank": rank,
            "original_id": old_id,
            "base_bits": bits,
            "delta_bits": delta,
            "frequency": frequency
        }));
    }
    let old_ids = extracted["sample_base_ids"]
        .as_array()
        .ok_or_else(|| invalid("ID spaziali mancanti."))?;
    let mut ranks = Vec::with_capacity(old_ids.len());
    let mut spatial_delta_bits = Vec::with_capacity(old_ids.len());
    for old_id in old_ids {
        let old_id = old_id
            .as_u64()
            .ok_or_else(|| invalid("ID spaziale non intero."))? as usize;
        let rank = old_to_rank
            .get(old_id)
            .and_then(|value| *value)
            .ok_or_else(|| invalid("ID spaziale fuori dal dizionario."))?;
        ranks.push(rank);
        spatial_delta_bits.push(deltas[rank].clone());
    }
    let max_delta = deltas
        .iter()
        .max_by(|left, right| compare_binary(left, right))
        .ok_or_else(|| invalid("Delta mancanti."))?;
    // `log2(delta + 1)` avoids -infinity for rank zero and preserves a bounded
    // numeric feature even when bases are wider than f64 can represent exactly.
    let denominator = binary_log2(&add_one_binary(max_delta)?)?;
    let delta_normalized: Vec<f64> = spatial_delta_bits
        .iter()
        .map(|delta| {
            let numerator = binary_log2(&add_one_binary(delta)?)?;
            Ok(if denominator == 0.0 {
                0.0
            } else {
                numerator / denominator
            })
        })
        .collect::<Result<_>>()?;
    let rank_normalized: Vec<f64> = ranks
        .iter()
        .map(|rank| {
            if num_bases == 1 {
                0.0
            } else {
                *rank as f64 / (num_bases - 1) as f64
            }
        })
        .collect();
    let frequencies: usize = ordered.iter().map(|(_, _, frequency)| frequency).sum();
    if frequencies != ranks.len() {
        return Err(invalid(
            "Le frequenze del dizionario non coincidono con gli ID spaziali.",
        ));
    }

    Ok(json!({
        "schema": "igd-network-features-v1",
        "source": extracted["source"].clone(),
        "image": extracted["image"].clone(),
        "verification": extracted["verification"].clone(),
        "representation": {
            "name": "sorted-base-rank-and-delta-v1",
            "index_origin": 0,
            "base_value_bit_width": base_bits,
            "base_bit_positions": extracted["base_bit_positions"].clone(),
            "base_order": "Ascending lexicographic order of equal-width binary strings; the leftmost character of base_bits is treated as the most-significant bit.",
            "delta_order": "delta_bits[0] is zero; delta_bits[r] = base_bits[r] - base_bits[r - 1]. Deltas are tied to dictionary rank, not to adjacent pixels.",
            "spatial_order": "Row-major: sample i has coordinates y = i // grid_width, x = i % grid_width.",
            "neural_channels": {
                "rank_normalized": "spatial_rank_ids / (num_bases - 1); all zero when num_bases is 1.",
                "delta_log2_normalized": "log2(delta + 1) / log2(max_delta + 1); all zero when max_delta is zero. Exact deltas remain in spatial_delta_bits."
            }
        },
        "num_bases": num_bases,
        "num_samples": ranks.len(),
        "base_min_bits": ordered.first().map(|base| base.1.clone()),
        "base_max_bits": ordered.last().map(|base| base.1.clone()),
        "max_delta_bits": max_delta,
        "sorted_dictionary": sorted_dictionary,
        "spatial_rank_ids": matrix(&ranks, grid_width, grid_height)?,
        "spatial_delta_bits": matrix(&spatial_delta_bits, grid_width, grid_height)?,
        "spatial_rank_normalized": matrix(&rank_normalized, grid_width, grid_height)?,
        "spatial_delta_log2_normalized": matrix(&delta_normalized, grid_width, grid_height)?
    }))
}

/// Convert the two normalized network maps to compact uint8 tensors.
/// No JSON is written to disk; the JSON value exists only while this function
/// derives the two arrays and their small metadata record.
pub fn extract_network_u8(bytes: Vec<u8>, source: &str, options: Options) -> Result<NetworkU8> {
    let document = extract_network_features(bytes, source, options)?;
    let width = document["image"]["grid_width"]
        .as_u64()
        .ok_or_else(|| invalid("grid_width mancante."))? as usize;
    let height = document["image"]["grid_height"]
        .as_u64()
        .ok_or_else(|| invalid("grid_height mancante."))? as usize;
    let rank_u8 = u8_matrix(&document, "spatial_rank_normalized", width, height)?;
    let delta_u8 = u8_matrix(&document, "spatial_delta_log2_normalized", width, height)?;
    let metadata = json!({
        "schema": "igd-network-npz-v1",
        "source": document["source"].clone(),
        "image": document["image"].clone(),
        "verification": document["verification"].clone(),
        "representation": {
            "name": "sorted-base-rank-and-delta-u8-v1",
            "array_order": "C row-major; shape is (height, width).",
            "rank_u8": "round(255 * rank / (num_bases - 1)); zero if num_bases is 1. Training restores approximately with rank_u8 / 255.",
            "delta_u8": "round(255 * log2(delta + 1) / log2(max_delta + 1)); zero if max_delta is zero. Training restores approximately with delta_u8 / 255.",
            "channel_order": ["rank_u8", "delta_u8"]
        },
        "num_bases": document["num_bases"].clone(),
        "num_samples": document["num_samples"].clone(),
        "base_min_bits": document["base_min_bits"].clone(),
        "base_max_bits": document["base_max_bits"].clone(),
        "max_delta_bits": document["max_delta_bits"].clone()
    });
    Ok(NetworkU8 {
        width,
        height,
        rank_u8,
        delta_u8,
        metadata_json: serde_json::to_vec_pretty(&metadata)?,
    })
}

/// Write a standard NumPy `.npz` archive with `rank_u8` and `delta_u8` arrays.
/// The third ZIP entry, `metadata.json`, is deliberately not a NumPy array and
/// therefore is ignored by `numpy.load(...).files` while remaining inspectable.
pub fn write_network_npz<P: AsRef<Path>>(
    bytes: Vec<u8>,
    source: &str,
    options: Options,
    output_path: P,
) -> Result<NetworkNpzSummary> {
    let network = extract_network_u8(bytes, source, options)?;
    let rank_npy = npy_u8_2d(network.width, network.height, &network.rank_u8)?;
    let delta_npy = npy_u8_2d(network.width, network.height, &network.delta_u8)?;
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)?;
    let mut writer = BufWriter::new(file);
    write_stored_npz(
        &mut writer,
        vec![
            ZipEntry {
                name: "rank_u8.npy",
                bytes: rank_npy,
            },
            ZipEntry {
                name: "delta_u8.npy",
                bytes: delta_npy,
            },
            ZipEntry {
                name: "metadata.json",
                bytes: network.metadata_json,
            },
        ],
    )?;
    writer.flush()?;
    let file_bytes = std::fs::metadata(output_path)?.len();
    Ok(NetworkNpzSummary {
        width: network.width,
        height: network.height,
        tensor_bytes: network.rank_u8.len() + network.delta_u8.len(),
        file_bytes,
    })
}

/// Reassemble one transformed chunk using only the JSON representation.
/// Requires the optional deviations; this does not invert color/group transforms.
pub fn reconstruct_chunk(document: &Value, sample_index: usize) -> Result<BitVec<usize, Lsb0>> {
    let chunk_bits = document["chunk_bits"]
        .as_u64()
        .ok_or_else(|| invalid("chunk_bits mancante."))? as usize;
    let id = document["sample_base_ids"][sample_index]
        .as_u64()
        .ok_or_else(|| invalid("ID campione mancante."))? as usize;
    let base = document["bases"][id]["bits"]
        .as_str()
        .ok_or_else(|| invalid("Base mancante."))?;
    let deviation = document["sample_deviation_bits"][sample_index]
        .as_str()
        .ok_or_else(|| invalid("Deviazioni mancanti: usare --include-deviations."))?;
    let mut result = bitvec![usize, Lsb0; 0; chunk_bits];
    let mut assigned = vec![false; chunk_bits];
    for (key, bits) in [
        ("base_bit_positions", base),
        ("deviation_bit_positions", deviation),
    ] {
        let positions = document[key]
            .as_array()
            .ok_or_else(|| invalid(format!("{key} mancante.")))?;
        if positions.len() != bits.len() {
            return Err(invalid(
                "Numero di posizioni diverso dalla lunghezza della stringa di bit.",
            ));
        }
        for (position, bit) in positions.iter().zip(bits.bytes()) {
            let position = position
                .as_u64()
                .ok_or_else(|| invalid("Posizione non intera."))?
                as usize;
            if position >= chunk_bits || assigned[position] || !matches!(bit, b'0' | b'1') {
                return Err(invalid(
                    "Posizione duplicata/fuori intervallo o valore diverso da 0/1.",
                ));
            }
            result.set(position, bit == b'1');
            assigned[position] = true;
        }
    }
    if assigned.iter().any(|&value| !value) {
        return Err(invalid("Alcuni bit del blocco non sono stati assegnati."));
    }
    Ok(result)
}
