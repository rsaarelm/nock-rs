#![feature(box_syntax, box_patterns)]

use std::fmt;

// Nock(a)          *a
// [a b c]          [a [b c]]
//
// ?[a b]           0
// ?a               1
// +[a b]           +[a b]
// +a               1 + a
// =[a a]           0
// =[a b]           1
// =a               =a
//
// /[1 a]           a
// /[2 a b]         a
// /[3 a b]         b
// /[(a + a) b]     /[2 /[a b]]
// /[(a + a + 1) b] /[3 /[a b]]
// /a               /a
//
// *[a [b c] d]     [*[a b c] *[a d]]
//
// *[a 0 b]         /[b a]
// *[a 1 b]         b
// *[a 2 b c]       *[*[a b] *[a c]]
// *[a 3 b]         ?*[a b]
// *[a 4 b]         +*[a b]
// *[a 5 b]         =*[a b]
//
// *[a 6 b c d]     *[a 2 [0 1] 2 [1 c d] [1 0] 2 [1 2 3] [1 0] 4 4 b]
// *[a 7 b c]       *[a 2 b 1 c]
// *[a 8 b c]       *[a 7 [[7 [0 1] b] 0 1] c]
// *[a 9 b c]       *[a 7 c 2 [0 1] 0 b]
// *[a 10 [b c] d]  *[a 8 c 7 [0 3] d]
// *[a 10 b c]      *[a c]
//
// *a               *a

#[derive(Clone, PartialEq, Eq)]
pub enum Noun {
    Atom(u64), // TODO: Need bigints?
    Cell(Box<Noun>, Box<Noun>),
}

impl fmt::Display for Noun {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Atom(ref n) => write!(f, "{}", n),
            &Cell(ref a, ref b) => {
                try!(write!(f, "[{} ", a));
                // List pretty-printer.
                let mut cur = b;
                loop {
                    match cur {
                        &box Cell(ref a, ref b) => {
                            try!(write!(f, "{} ", a));
                            cur = &b;
                        }
                        &box Atom(ref n) => {
                            return write!(f, "{}]", n);
                        }
                    }
                }
            }
        }
    }
}

impl fmt::Debug for Noun {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", self) }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Op {
    Wut, // ?
    Lus, // +
    Tis, // =
    Fas, // /
    Tar, // *
}

use Op::*;
use Noun::*;

fn eval(op: Op, noun: Noun) -> Result<Noun, Noun> {
    match (op, noun) {
        (Wut, Cell(_, _)) => Ok(Atom(0)),
        (Wut, Atom(_)) => Ok(Atom(1)),
        (Lus, Atom(a)) => Ok(Atom(1 + a)),
        (Lus, a) => Err(a),
        (Tis, Cell(a, b)) => Ok(Atom(if a == b { 0 } else { 1 })),
        (Tis, a) => Err(a),

        (Fas, Cell(box Atom(1), box a)) => Ok(a),
        (Fas, Cell(box Atom(2), box Cell(box a, _))) => Ok(a),
        (Fas, Cell(box Atom(3), box Cell(_, box b))) => Ok(b),
        (Fas, Cell(box Atom(a), b)) => {
            let x = try!(eval(Fas, Cell(box Atom(a / 2), b)));
            eval(Fas, Cell(box Atom(if a % 2 == 0 { 2 } else { 3 }), box x))
        }
        (Fas, a) => Err(a),

        (Tar, Cell(a, box Cell(box Cell(b, c), d))) => {
            let x = try!(eval(Tar, Cell(a.clone(), box Cell(b, c))));
            let y = try!(eval(Tar, Cell(a, d)));
            Ok(Cell(box x, box y))
        }

        // TODO: Numbered tar ops.

        (Tar, a) => Err(a)
    }
}

pub fn nock(noun: Noun) -> Result<Noun, Noun> { eval(Wut, noun) }

#[test]
fn it_works() {
    println!("{}", Cell(box Atom(1), box Cell(box Atom(2), box Atom(3))));
    println!("{:?}", eval(Tis, Cell(box Atom(2), box Atom(2))));
    println!("{:?}", eval(Fas, Cell(box Atom(7), box Cell(box Cell(box Atom(4), box Atom(5)), box Cell(box Atom(6), box Cell(box Atom(14), box Atom(15)))))));
    assert!(false);
}
