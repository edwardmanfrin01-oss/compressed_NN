//! Streaming dictionary metadata for selective recompression; no pixel decoding.
use gdcompress::{BitDataReconstructionInfo, IgdFile};
use std::io::{self, BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let path = line?;
        let bytes = std::fs::read(&path)?;
        if bytes.len() < 32 || u64::from_le_bytes(bytes[24..32].try_into()?) != (bytes.len() - 32) as u64 {
            return Err(format!("Invalid IGD payload length: {path}").into());
        }
        let data = IgdFile::from_bytes(bytes).to_compressed_data()?;
        let BitDataReconstructionInfo::Image(info) = data.metadata.reconstruction else {
            return Err(format!("Not an image: {path}").into());
        };
        writeln!(output, "{}\t{}\t{}\t{}\t{}", data.base_table.len(), info.width, info.height,
                 info.pixel_grouping.width(), info.pixel_grouping.height())?;
        output.flush()?;
    }
    Ok(())
}
