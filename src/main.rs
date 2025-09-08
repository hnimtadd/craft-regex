use log;
use std::env;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::process;

mod logger;
mod regex;

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
// Usage: echo <input_text> | your_program.sh -E <pattern>
fn main() {
    let is_debug = if let Result::Ok(val) = env::var("debug") {
        val.eq_ignore_ascii_case("true") || val == "1" || val.eq_ignore_ascii_case("yes")
    } else {
        false
    };
    logger::logger::Logger::init(is_debug);
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args[1] != "-E" {
        log::error!("Expected first argument to be '-E'");
        process::exit(1);
    }

    let pattern = &args[2];
    let nfa = regex::compiler::compile(&pattern);
    let paths = args.iter().skip(3).collect::<Vec<&String>>();
    let mut exit_code = 1;

    if paths.is_empty() {
        // handle only std in case.
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let result = nfa.simulate(&input);
        if result.is_match {
            log::debug!("{:?}", result.capture_groups);
            exit_code = 0;
        }
    } else {
        let mut is_match = false;
        let print_path = paths.len() > 1;
        for path in paths {
            match process_file(path, &nfa, print_path) {
                Ok(matched) => {
                    if matched {
                        is_match = true;
                        exit_code = 0;
                    }
                }
                Err(e) => {
                    log::error!("Failed to process file {}: {}", path, e);
                    exit_code = 1;
                }
            }
        }
        if is_match {
            exit_code = 0;
        }
    }

    process::exit(exit_code);
}
