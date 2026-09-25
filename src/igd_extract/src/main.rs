use igd_extract::{Options, Result, extract, extract_network_features};
use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

const HELP: &str = "USE: igd_extract INPUT.igd [-o OUTPUT.json] [--network-features] [--include-deviations] [--verify]

Esporta basi, ID spaziali e mappa delle posizioni dei bit in JSON leggibile.
  -o, --output PATH       Default: INPUT.decoded.json (or INPUT.network.json)
  --network-features      Esporta rango e delta ordinati per il primo esperimento di rete
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
            Some("--include-deviations") => options.include_deviations = true,
            Some("--verify") => options.verify = true,
            Some(flag) if flag.starts_with('-') => {
                return Err(argument_error("Opzione sconosciuta. Usare --help."));
            }
            _ => {
                if input.is_some() {
                    return Err(argument_error("Specificare un solo file IGD."));
                }
                input = Some(PathBuf::from(arg));
            }
        }
    }
    let input = input.ok_or_else(|| argument_error(HELP))?;
    let output = output.unwrap_or_else(|| {
        input.with_extension(if network_features {
            "network.json"
        } else {
            "decoded.json"
        })
    });
    if output.exists() {
        return Err(argument_error(
            "L'output esiste gia': scegliere un altro nome.",
        ));
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
