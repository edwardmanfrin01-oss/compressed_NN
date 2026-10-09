use igd_extract::{
    Options, Result, extract, extract_network_features, write_adaptive_cumulative_npz,
    write_cumulative_npz, write_network_npz,
};
use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

const HELP: &str = "USE: igd_extract INPUT.igd [-o OUTPUT] [--network-features | --network-npz] [--include-deviations] [--verify]

Esporta basi, ID spaziali e mappa delle posizioni dei bit in JSON leggibile.
  -o, --output PATH       Default: INPUT.decoded.json, INPUT.network.json, or INPUT.network.npz
  --network-features      Esporta rango e delta ordinati per il primo esperimento di rete
  --network-npz           Esporta rank_u8 e delta_u8 in NPZ, senza JSON intermedio
  --representation cumulative-rank  Esporta un canale z_u16 in NPZ (gruppi 1x1)
  --representation cumulative-rank-adaptive  V3: S_i automatico, floor, uint16
  --scale S               Intero 0..65535, default 1; solo con cumulative-rank
  --include-deviations   Include anche i bit necessari a ricostruire ogni blocco
  --verify               Confronta tutti i blocchi con il decoder gdcompress
  -h, --help             Mostra questo messaggio

Gli indici partono da 0. L'ordine cronologico di selezione non e' salvato nell'IGD.
Un file di output esistente non viene sovrascritto.";

fn argument_error(message: &str) -> Box<dyn std::error::Error> {
    Box::new(io::Error::new(
        io::ErrorKind::InvalidInput,
        message.to_string(),
    ))
}

fn run() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut input = None;
    let mut output = None;
    let mut options = Options::default();
    let mut network_features = false;
    let mut network_npz = false;
    let mut cumulative = false;
    let mut adaptive = false;
    let mut scale = None;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("-h" | "--help") => {
                println!("{HELP}");
                return Ok(());
            }
            Some("-o" | "--output") => {
                output =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        argument_error("Percorso mancante dopo --output.")
                    })?));
            }
            Some("--network-features") => network_features = true,
            Some("--network-npz") => network_npz = true,
            Some("--representation") => {
                let value = args
                    .next()
                    .ok_or_else(|| argument_error("Missing representation"))?;
                if value != "cumulative-rank" && value != "cumulative-rank-adaptive" {
                    return Err(argument_error(
                        "Supported representations: cumulative-rank, cumulative-rank-adaptive",
                    ));
                }
                adaptive = value == "cumulative-rank-adaptive";
                cumulative = true;
            }
            Some("--scale") => {
                let value = args.next().ok_or_else(|| argument_error("Missing scale"))?;
                scale = Some(
                    value
                        .to_str()
                        .and_then(|v| v.parse::<u16>().ok())
                        .ok_or_else(|| argument_error("Scale must be an integer in 0..65535"))?,
                );
            }
            Some("--include-deviations") => options.include_deviations = true,
            Some("--verify") => options.verify = true,
            Some(flag) if flag.starts_with('-') => {
                return Err(argument_error("Unknown option. Use --help."));
            }
            _ => {
                if input.is_some() {
                    return Err(argument_error("Specify only one IGD file."));
                }
                input = Some(PathBuf::from(arg));
            }
        }
    }
    if u8::from(network_features) + u8::from(network_npz) + u8::from(cumulative) > 1 {
        return Err(argument_error(
            "Choose only one output mode: --network-features, --network-npz, or --representation cumulative-rank.",
        ));
    }
    if scale.is_some() && (!cumulative || adaptive) {
        return Err(argument_error(
            "--scale requires --representation cumulative-rank",
        ));
    }
    if cumulative && options.include_deviations {
        return Err(argument_error(
            "Cumulative NPZ does not include deviations; use --verify for verification",
        ));
    }
    let scale = scale.unwrap_or(1);
    let input = input.ok_or_else(|| argument_error(HELP))?;
    let output = output.unwrap_or_else(|| {
        if adaptive {
            return input.with_extension("cumulative-v3.npz");
        }
        if cumulative {
            return input.with_extension(format!("cumulative-s{scale}.npz"));
        }
        input.with_extension(if network_npz {
            "network.npz"
        } else if network_features {
            "network.json"
        } else {
            "decoded.json"
        })
    });
    if output.exists() {
        return Err(argument_error("This output already exists"));
    }
    if cumulative {
        let summary = if adaptive {
            write_adaptive_cumulative_npz(
                fs::read(&input)?,
                &input.to_string_lossy(),
                options,
                &output,
            )?
        } else {
            write_cumulative_npz(
                fs::read(&input)?,
                &input.to_string_lossy(),
                options,
                scale,
                &output,
            )?
        };
        println!(
            "Saved: {}\n{}; {}x{}; one uint16 channel: {} bytes; NPZ: {} bytes.",
            output.display(),
            if adaptive {
                "S_i automatic (see metadata)".to_string()
            } else {
                format!("S={scale}")
            },
            summary.width,
            summary.height,
            summary.tensor_bytes,
            summary.file_bytes
        );
        if options.verify {
            println!("Full transformed-chunk verification: OK.");
        }
        return Ok(());
    }
    if network_npz {
        let summary = write_network_npz(
            fs::read(&input)?,
            &input.to_string_lossy(),
            options,
            &output,
        )?;
        println!("Saved: {}", output.display());
        println!(
            "{}x{}; two uint8 channels: {} bytes before NPZ metadata.",
            summary.width, summary.height, summary.tensor_bytes
        );
        println!("NPZ file size: {} bytes.", summary.file_bytes);
        if options.verify {
            println!("Full transformed-chunk verification: OK.");
        }
        return Ok(());
    }
    let document = if network_features {
        extract_network_features(fs::read(&input)?, &input.to_string_lossy(), options)?
    } else {
        extract(fs::read(&input)?, &input.to_string_lossy(), options)?
    };
    let mut json = serde_json::to_vec_pretty(&document)?;
    json.push(b'\n');
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    // create_new prevents overwriting the input or any pre-existing analysis.
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&json)?;
    writer.flush()?;
    println!("Salvato: {}", output.display());
    if network_features {
        println!(
            "{} campioni, {} basi, {} bit/base. Mappe: rango e delta normalizzato.",
            document["num_samples"],
            document["num_bases"],
            document["representation"]["base_value_bit_width"]
        );
    } else {
        println!(
            "{} campioni, {} basi, {} bit/base ({} variabili), {} bit/blocco.",
            document["num_samples"],
            document["num_bases"],
            document["base_bits"],
            document["variable_base_bits"],
            document["chunk_bits"]
        );
    }
    if options.verify {
        println!("Verifica completa dei blocchi: OK.");
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Errore: {error}");
        std::process::exit(1);
    }
}
