use log::{Level, Metadata, Record};
use std::io::{self, Write};

pub struct Logger {
    level: Level,
}

impl log::Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level
    }
    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            // Write to stderr, which is a common practice for logging.
            let _ = writeln!(&mut io::stderr(), "{}", record.args());
        }
    }
    fn flush(&self) {}
}

impl Logger {
    pub fn init(is_debug: bool) {
        let level = if is_debug { Level::Debug } else { Level::Info };
        log::set_max_level(level.to_level_filter());
        let logger = Box::new(Self { level });
        log::set_logger(Box::leak(logger)).unwrap();
    }
}
