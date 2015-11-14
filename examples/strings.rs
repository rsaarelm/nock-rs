#![feature(box_patterns)]

extern crate nock;

use std::io::{self, Read};

// Parse long values from nock as ascii strings.
// Start out with "!!" (first ASCII char after space, at least two chars) as
// the first potential string. Evals to 0x2121.

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let nock = input.parse::<nock::Noun>().unwrap();
    strings(nock);
}

fn strings(noun: nock::Noun) {
    match noun {
        nock::Noun::Cell(box a, box b) => {
            strings(a);
            strings(b);
        }
        nock::Noun::Atom(a) => {
            if a >= 0x2121 {
                if let Some(s) = nock::cord(a) {
                    println!("{}", s);
                }
            }
        }
    }
}
