extern crate log;
extern crate nock;

use std::io::{self, BufRead, Write};
use log::{LogRecord, LogLevel, LogMetadata, LogLevelFilter, SetLoggerError};

struct SimpleLogger;

impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &LogMetadata) -> bool {
        metadata.level() <= LogLevel::Info
    }

    fn log(&self, record: &LogRecord) {
        if self.enabled(record.metadata()) {
            println!("{} - {}", record.level(), record.args());
        }
    }
}

pub fn init_log() -> Result<(), SetLoggerError> {
    log::set_logger(|max_log_level| {
        max_log_level.set(LogLevelFilter::Info);
        Box::new(SimpleLogger)
    })
}

fn main() {
    // init_log().unwrap();

    println!("Welcome to nock-rs");
    loop {
        let mut input = String::new();
        print!("> ");
        io::stdout().flush().expect("IO error");
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                match input.parse::<nock::Noun>() {
                    Ok(noun) => {
                        match noun.nock() {
                            Ok(eval) => println!("{}", eval),
                            Err(_) => println!("Eval error"),
                        }
                    }
                    Err(_) => println!("Syntax error"),
                }
            }

            Err(_) => break,
        }
    }
}
