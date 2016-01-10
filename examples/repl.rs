extern crate nock;

use std::io::{self, BufRead, Write};

fn main() {
    println!("Welcome to nock-rs");
    loop {
        let mut input = String::new();
        print!("> ");
        io::stdout().flush().expect("IO error");
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                match input.parse::<nock::Noun>() {
                    Ok(nock::Noun::Cell(s, f)) => {
                        match nock::nock_on(&s, &f) {
                            Ok(eval) => println!("{}", eval),
                            Err(_) => println!("Eval error"),
                        }
                    }
                    _ => println!("Syntax error"),
                }
            }

            Err(_) => break,
        }
    }
}
