extern crate nock;

use std::io::{self, BufRead, Write};
use nock::parse;

fn main() {
    println!("Welcome to nock-rs");
    loop {
        let mut input = String::new();
        print!("> ");
        io::stdout().flush().expect("IO error");
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                match parse(&input) {
                    Ok(noun) =>
                        println!("{}", noun),
                    Err(_) =>
                        println!("Syntax error"),
                }
            }

            Err(_) => break
        }
    }
}
