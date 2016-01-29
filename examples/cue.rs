extern crate nock;
extern crate num;

use std::env;
use std::io::prelude::*;
use std::fs::File;
use nock::Noun;
use nock::Shape;
use nock::nock_on;
use num::bigint::BigUint;
use num::traits::One;

fn main() {
    let mut file = File::open(env::args()
                                  .nth(1)
                                  .expect("Usage cue [urbit.pill]"))
                       .unwrap();

    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();
    println!("Unpacking pill");
    let noun = nock::unpack_pill(buf).unwrap();
    println!("Unpacked pill");

    let count: BigUint = noun.fold(|x| {
        match x {
            Shape::Cell(p, q) => p + q,
            _ => One::one(),
        }
    });

    println!("Pill has {} atoms", count);

    println!("Nocking pill");
    let noun = nock_on(Noun::from(0u32), noun).unwrap();
    println!("Result: {}", noun);
}
