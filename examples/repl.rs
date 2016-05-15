extern crate nock;

use std::default::Default;
use std::io::{self, Write};
use nock::{Nock, Noun};

struct VM;
impl Nock for VM {}

fn main() {
    let mut subject: Noun = Default::default();
    // Default to identity formula.
    let mut formula = Noun::cell(Noun::from(0u32), Noun::from(1u32));
    let mut vm = VM;

    println!("Welcome to nock-rs");
    println!("Type a formula to nock on the subject");
    loop {
        let mut input = String::new();

        println!("{}", &subject);
        print!("> ");
        io::stdout().flush().expect("IO error");
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                if input == "\n" {
                    // Reuse previous valid formula on empty input.
                    match vm.nock_on(subject.clone(), formula.clone()) {
                        Ok(eval) => {
                            subject = eval;
                        }
                        Err(_) => println!("Nock eval error"),
                    }
                } else {
                    match input.parse::<nock::Noun>() {
                        Ok(f) => {
                            match vm.nock_on(subject.clone(), f.clone()) {
                                Ok(eval) => {
                                    subject = eval;
                                    formula = f;
                                }
                                Err(_) => println!("Nock eval error"),
                            }
                        }
                        _ => println!("Not a Nock formula"),
                    }
                }
            }

            Err(_) => break,
        }
    }
}
