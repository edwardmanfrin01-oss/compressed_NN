//! Read dictionaries only; never reconstruct spatial maps or modify inputs.
use gdcompress::compression::base_table::BaseBitLayoutState;
use gdcompress::compression::encoding::BaseTable;
use gdcompress::{BitDataReconstructionInfo, IgdFile};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

fn zmax(a: &[f64], s: u64) -> u64 {
    a.len() as u64 + a.iter().map(|x| (x * s as f64).round() as u64).sum::<u64>()
}

fn largest_s(a: &[f64], limit: u64) -> Option<u64> {
    if a.len() as u64 > limit {
        return None;
    }
    // Single-base dictionaries have no finite upper bound; handled separately.
    if a.is_empty() {
        return None;
    }
    let (mut lo, mut hi) = (0, limit + 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if zmax(a, mid) <= limit {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

fn analyze(path: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 60
        || &bytes[..3] != b"IGD"
        || u64::from_le_bytes(bytes[24..32].try_into()?) != (bytes.len() - 32) as u64
    {
        return Err("Invalid or truncated IGD".into());
    }
    let mut data = IgdFile::from_bytes(bytes).to_compressed_data()?;
    let info = match &data.metadata.reconstruction {
        BitDataReconstructionInfo::Image(info) => *info,
        _ => return Err("Not an image".into()),
    };
    if info.pixel_grouping.width() != 1 || info.pixel_grouping.height() != 1 {
        return Err("Analysis requires grouping 1x1".into());
    }
    if data.condensed_sample_weights.is_some() {
        return Err("Condensed samples unsupported".into());
    }
    if let BaseTable::Delta(delta) = &data.base_table {
        data.base_table = BaseTable::Raw(delta.decode_rows()?);
    }
    let positions = data.layout.selected_base_bit_positions();
    if positions.len() > 32 {
        return Err("Analysis supports at most 32 base bits".into());
    }
    let mut values = Vec::new();
    for (bits, _) in data.base_table.as_raw().iter() {
        let mut index = 0;
        let mut value = 0u64;
        for &p in &positions {
            let bit = match data.layout.state_at(p) {
                BaseBitLayoutState::Variable => {
                    let b = *bits.get(index).ok_or("Short base row")?;
                    index += 1;
                    b
                }
                BaseBitLayoutState::ConstantZero => false,
                BaseBitLayoutState::ConstantOne => true,
                _ => return Err("Invalid base position".into()),
            };
            value = (value << 1) | u64::from(bit);
        }
        if index != bits.len() {
            return Err("Long base row".into());
        }
        values.push(value);
    }
    values.sort_unstable();
    if values.is_empty() || values.windows(2).any(|p| p[0] == p[1]) {
        return Err("Empty or duplicate dictionary".into());
    }
    let deltas: Vec<_> = values.windows(2).map(|p| p[1] - p[0]).collect();
    let max = *deltas.iter().max().unwrap_or(&0);
    let denom = ((max + 1) as f64).log2();
    let a: Vec<_> = deltas
        .iter()
        .map(|d| ((*d + 1) as f64).log2() / denom)
        .collect();
    Ok(
        json!({"path":path, "width":info.width, "height":info.height,
        "base_bits":positions.len(), "num_bases":values.len(),
        "delta_min":deltas.iter().min(), "delta_max":max,
        "normalized_delta_mean":if a.is_empty() {0.0} else {a.iter().sum::<f64>()/a.len() as f64},
        "all_gaps_equal":!deltas.is_empty() && deltas.iter().all(|d| *d == max),
        "zmax_s0_to_16":(0..=16).map(|s| zmax(&a,s)).collect::<Vec<_>>(),
        "max_s_uint8":largest_s(&a,255), "max_s_uint16":largest_s(&a,65535),
        "s_unbounded":a.is_empty()}),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    // One path per stdin line, one compact result per stdout line.
    for line in io::stdin().lock().lines() {
        let path = line?;
        let result = std::panic::catch_unwind(|| analyze(&path));
        let doc = match result {
            Ok(Ok(doc)) => doc,
            Ok(Err(e)) => json!({"path":path,"error":e.to_string()}),
            Err(_) => json!({"path":path,"error":"Decoder panicked"}),
        };
        serde_json::to_writer(&mut out, &doc)?;
        writeln!(out)?;
        out.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_limits_and_rounding() {
        assert_eq!(zmax(&[1.0, 1.0, 1.0], 255), 768);
        assert_eq!(largest_s(&[1.0, 1.0, 1.0], 255), Some(84));
        assert_eq!(zmax(&[0.49, 0.5, 1.0], 1), 5);
        assert_eq!(largest_s(&vec![1.0; 256], 255), None);
        assert_eq!(zmax(&[], 100), 0);
        assert_eq!(largest_s(&[], 65535), None);
    }
}
