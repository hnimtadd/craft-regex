use log;
use std::env;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::process;

mod logger;
mod regex;

// Usage: echo <input_text> | your_program.sh -E <pattern>
fn main() {
    let is_debug = if let Result::Ok(val) = env::var("debug") {
        val.eq_ignore_ascii_case("true") || val == "1" || val.eq_ignore_ascii_case("yes")
    } else {
        false
    };
    logger::logger::Logger::init(is_debug);
    if env::args().nth(1).unwrap() != "-E" {
        log::error!("Expected first argument to be '-E'");
        process::exit(1);
    }
    let pattern = env::args().nth(2).unwrap();
    let nfa = regex::compiler::compile(&pattern);

    if let Some(path) = env::args().nth(3) {
        let f = File::open(path.as_str()).unwrap();
        let r = BufReader::new(f);
        for line_result in r.lines() {
            let line = line_result.unwrap(); // Check each line for an I/O error
            let result = nfa.simulate(&line);
            if result.is_match {
                println!("{}", line);
            }
        }
    } else {
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let result = nfa.simulate(&input);
        if result.is_match {
            log::info!("{:?}", result.capture_groups);
            // for cg in result.capture_groups {
            //     println!(
            //         "group: {}, from: {}, to: {}, match: {}",
            //         cg.idx,
            //         cg.from,
            //         cg.to,
            //         &input_line[cg.from..cg.to],
            //     );
            // }
            process::exit(0);
        } else {
            process::exit(1);
        }
    }
}
