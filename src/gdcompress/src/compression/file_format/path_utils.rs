use std::path::{Path, PathBuf};

pub(super) fn ensure_egd_extension(path: &Path) -> PathBuf {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("egd") => path.to_path_buf(),
        _ => {
            let mut out = path.to_path_buf();
            out.set_extension("egd");
            out
        }
    }
}

pub(super) fn ensure_igd_extension(path: &Path) -> PathBuf {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("igd") => path.to_path_buf(),
        _ => {
            let mut out = path.to_path_buf();
            out.set_extension("igd");
            out
        }
    }
}

pub(super) fn ensure_csv_extension(path: &Path) -> PathBuf {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("csv") => path.to_path_buf(),
        _ => {
            let mut out = path.to_path_buf();
            out.set_extension("csv");
            out
        }
    }
}

pub(super) fn ensure_tgd_extension(path: &Path) -> PathBuf {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("tgd") => path.to_path_buf(),
        _ => {
            let mut out = path.to_path_buf();
            out.set_extension("tgd");
            out
        }
    }
}
