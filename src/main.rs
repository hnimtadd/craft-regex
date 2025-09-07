use std::env;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::process;

mod regex;

// Usage: echo <input_text> | your_program.sh -E <pattern>
fn main() {
    if env::args().nth(1).unwrap() != "-E" {
        println!("Expected first argument to be '-E'");
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
            println!("{:?}", result.capture_groups);
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
