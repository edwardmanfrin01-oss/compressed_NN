use std::borrow::Cow;
use std::sync::OnceLock;
use std::time::Instant;
use tracing::Level;

fn parse_bool_env(value: &str) -> bool {
    !matches!(
        value,
        "0" | "false" | "FALSE" | "False" | "no" | "NO" | "off" | "OFF"
    )
}

fn timing_enabled() -> bool {
    static TIMING_ENABLED: OnceLock<bool> = OnceLock::new();
    *TIMING_ENABLED.get_or_init(|| {
        std::env::var("ENTRO_GD_TIMING")
            .map(|v| parse_bool_env(v.as_str()))
            .unwrap_or(true)
    })
}

/// Scope-based timer that logs elapsed time when dropped.
///
/// Behavior:
/// - Respects log level filtering (`RUST_LOG`).
/// - Can be globally disabled with `ENTRO_GD_TIMING=0`.
/// - Has near-zero overhead when disabled.
pub struct ScopedTimer {
    start: Option<Instant>,
    label: Cow<'static, str>,
    level: Level,
}

impl ScopedTimer {
    fn level_enabled(level: Level) -> bool {
        match level {
            Level::ERROR => tracing::enabled!(Level::ERROR),
            Level::WARN => tracing::enabled!(Level::WARN),
            Level::INFO => tracing::enabled!(Level::INFO),
            Level::DEBUG => tracing::enabled!(Level::DEBUG),
            Level::TRACE => tracing::enabled!(Level::TRACE),
        }
    }

    pub fn new(level: Level, label: impl Into<Cow<'static, str>>) -> Self {
        let enabled = timing_enabled() && Self::level_enabled(level);
        ScopedTimer {
            start: enabled.then(Instant::now),
            label: label.into(),
            level,
        }
    }

    pub fn info(label: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Level::INFO, label)
    }

    pub fn debug(label: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Level::DEBUG, label)
    }

    pub fn trace(label: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Level::TRACE, label)
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed = start.elapsed();
            match self.level {
                Level::ERROR => tracing::error!("{} completed in {:.3?}", self.label, elapsed),
                Level::WARN => tracing::warn!("{} completed in {:.3?}", self.label, elapsed),
                Level::INFO => tracing::info!("{} completed in {:.3?}", self.label, elapsed),
                Level::DEBUG => tracing::debug!("{} completed in {:.3?}", self.label, elapsed),
                Level::TRACE => tracing::trace!("{} completed in {:.3?}", self.label, elapsed),
            }
        }
    }
}
