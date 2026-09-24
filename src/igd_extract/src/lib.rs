//! Export IGD dictionaries and spatial base IDs without reconstructing RGB pixels.
//! Bit positions always refer to the transformed chunk, not the RGB byte stream.

use bitvec::prelude::*;
use gdcompress::compression::base_table::BaseBitLayoutState;
use gdcompress::compression::decompression::decompress_file;
use gdcompress::compression::encoding::{BaseTable, EncodedData};
use gdcompress::{BitDataReconstructionInfo, FeatureTransform, IgdFile};
use serde_json::{Value, json};
use std::error::Error;
use std::io;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Default, Clone, Copy)]
pub struct Options {
    pub include_deviations: bool,
    /// Reconstruct transformed chunks from the exported representation and compare
    /// every bit with gdcompress's full bit-data decoder before writing the JSON.
    pub verify: bool,
}

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

fn bit_string(bits: &BitSlice<usize, Lsb0>) -> String {
    bits.iter().map(|b| if *b { '1' } else { '0' }).collect()
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
