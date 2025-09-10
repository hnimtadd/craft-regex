use clap::Parser;
use log;
use std::env;
use std::fs;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::process;

mod logger;
mod regex;

fn process_folder(dir: &str, nfa: &regex::compiler::NFA) -> io::Result<bool> {
    let mut overall_match: bool = false;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if process_folder(entry.path().to_str().unwrap(), &nfa)? {
                overall_match = true;
            }
        } else {
            if process_file(entry.path().to_str().unwrap(), &nfa, true)? {
                overall_match = true
            }
        };
    }
    Ok(overall_match)
}

fn process_file(path: &str, nfa: &regex::compiler::NFA, print_path: bool) -> io::Result<bool> {
    let f = File::open(path)?;

    let reader = BufReader::new(f);

    let mut is_match = false;
    for line_result in reader.lines() {
        let line = line_result?; // Check each line for an I/O error
        let result = nfa.simulate(&line);
        if result.is_match {
            if print_path {
                println!("{}:{}", path, line);
            } else {
                println!("{}", line);
            }
            if is_match == false {
                is_match = true;
            }
        }
    }
    Ok(is_match)
}

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short = 'E')]
    expression: String,

    /// Number of times to greet
    #[arg(short, default_value_t = false)]
    recursive: bool,

    #[arg(required = true)]
    targets: Vec<String>,
}

// Usage: echo <input_text> | your_program.sh -E <pattern>
fn main() -> io::Result<()> {
    let is_debug = if let Result::Ok(val) = env::var("debug") {
        val.eq_ignore_ascii_case("true") || val == "1" || val.eq_ignore_ascii_case("yes")
    } else {
        false
    };
    let args = Args::parse();
    logger::logger::Logger::init(is_debug);

    let nfa = regex::compiler::compile(&args.expression);
    let mut exit_code = 1;

    if args.targets.is_empty() {
        // handle only std in case.
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let result = nfa.simulate(&input);
        if result.is_match {
            log::debug!("{:?}", result.capture_groups);
            exit_code = 0;
        }
    } else if args.recursive {
        if args.targets.len() != 1 {
            log::error!("-r work with 1 directory only");
            exit_code = 1;
        }
        let dir = args.targets.iter().nth(0).unwrap();
        match process_folder(dir.as_str(), &nfa) {
            Ok(is_match) => {
                if is_match {
                    exit_code = 0;
                }
            }
            Err(err) => log::error!("cannot process the dir: {}", err),
        }
    } else {
        let print_path = args.targets.len() > 1;
        for path in args.targets {
            match process_file(path.as_str(), &nfa, print_path) {
                Ok(matched) => {
                    if matched {
                        exit_code = 0;
                    };
                }
                Err(e) => {
                    log::error!("Failed to process file {}: {}", path, e);
                }
            }
        }
    }

    process::exit(exit_code);
}
