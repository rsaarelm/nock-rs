//! Implementation of the Nock virtual machine.
//!
//! See http://urbit.org/docs/theory/whitepaper for more information on Nock.
//!
//! The Nock spec:
//!
//! ```notrust
//! A noun is an atom or a cell.
//! An atom is a natural number.
//! A cell is an ordered pair of nouns.
//!
//! nock(a)          *a
//! [a b c]          [a [b c]]
//!
//! ?[a b]           0
//! ?a               1
//! +[a b]           +[a b]
//! +a               1 + a
//! =[a a]           0
//! =[a b]           1
//! =a               =a
//!
//! /[1 a]           a
//! /[2 a b]         a
//! /[3 a b]         b
//! /[(a + a) b]     /[2 /[a b]]
//! /[(a + a + 1) b] /[3 /[a b]]
//! /a               /a
//!
//! *[a [b c] d]     [*[a b c] *[a d]]
//!
//! *[a 0 b]         /[b a]
//! *[a 1 b]         b
//! *[a 2 b c]       *[*[a b] *[a c]]
//! *[a 3 b]         ?*[a b]
//! *[a 4 b]         +*[a b]
//! *[a 5 b]         =*[a b]
//!
//! *[a 6 b c d]     *[a 2 [0 1] 2 [1 c d] [1 0] 2 [1 2 3] [1 0] 4 4 b]
//! *[a 7 b c]       *[a 2 b 1 c]
//! *[a 8 b c]       *[a 7 [[7 [0 1] b] 0 1] c]
//! *[a 9 b c]       *[a 7 c 2 [0 1] 0 b]
//! *[a 10 [b c] d]  *[a 8 c 7 [0 3] d]
//! *[a 10 b c]      *[a c]
//!
//! *a               *a
//! ```

#![crate_name="nock"]

extern crate num;

use std::fmt;
use std::iter;
use std::str;
use std::rc::Rc;
use num::bigint::BigUint;
use num::traits::{ToPrimitive, Zero};

/// A Nock noun, the basic unit of representation
///
/// # Examples
///
/// ```
/// let noun: nock::Noun = "[19 4 0 1]".parse().unwrap();
/// assert_eq!(format!("{}", noun.nock().unwrap()), "20");
/// ```
#[derive(Clone, PartialEq, Eq)]
pub enum Noun {
    /// A single positive integer
    Atom(u32),
    BigAtom(BigUint),
    /// A a pair of two nouns
    Cell(Rc<Noun>, Rc<Noun>),
}

impl Noun {
    /// Evaluate the noun using the function `nock(a) = *a` as defined in
    /// the Nock spec.
    pub fn nock(&self) -> NockResult {
        tar(self)
    }

    /// If the noun has structure [a [b c]], return a tuple of a, b and c.
    fn as_triple(&self) -> Option<(Rc<Noun>, Rc<Noun>, Rc<Noun>)> {
        if let &Cell(ref a, ref bc) = self {
            if let Cell(ref b, ref c) = **bc {
                return Some((a.clone(), b.clone(), c.clone()));
            }
        }
        None
    }
}

impl Into<Noun> for u32 {
    fn into(self) -> Noun {
        Noun::Atom(self)
    }
}

impl iter::FromIterator<Noun> for Noun {
    fn from_iter<T>(iterator: T) -> Self
        where T: IntoIterator<Item = Noun>
    {
        let mut v: Vec<Noun> = iterator.into_iter().collect();
        v.reverse();

        v.into_iter()
         .fold(None, |acc, i| {
             acc.map_or_else(|| Some(i.clone()),
                             |a| Some(Noun::Cell(Rc::new(i.clone()), Rc::new(a))))
         })
         .expect("Can't make noun from empty list")
    }
}

impl fmt::Display for Noun {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Atom(ref n) => return dot_separators(f, &n),
            &BigAtom(ref n) => return dot_separators(f, &n),
            &Cell(ref a, ref b) => {
                try!(write!(f, "[{} ", a));
                // List pretty-printer.
                let mut cur = b;
                loop {
                    match **cur {
                        Cell(ref a, ref b) => {
                            try!(write!(f, "{} ", a));
                            cur = &b;
                        }
                        Atom(ref n) => {
                            try!(dot_separators(f, &n));
                            return write!(f, "]");
                        }
                        BigAtom(ref n) => {
                            try!(dot_separators(f, &n));
                            return write!(f, "]");
                        }
                    }
                }
            }
        }

        fn dot_separators<T: fmt::Display>(f: &mut fmt::Formatter, item: &T) -> fmt::Result {
            let s = format!("{}", item);
            let phase = s.len() % 3;
            for (i, c) in s.chars().enumerate() {
                if i > 0 && i % 3 == phase {
                    try!(write!(f, "."));
                }
                try!(write!(f, "{}", c));
            }
            Ok(())
        }
    }

}

impl fmt::Debug for Noun {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct ParseError;

impl str::FromStr for Noun {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, ParseError> {
        return parse(&mut s.chars().peekable());

        fn parse<I: Iterator<Item = char>>(input: &mut iter::Peekable<I>) -> Result<Noun, ParseError> {
            eat_space(input);
            match input.peek().map(|&x| x) {
                Some(c) if c.is_digit(10) => parse_atom(input),
                Some(c) if c == '[' => parse_cell(input),
                _ => Err(ParseError),
            }
        }

        /// Parse an atom, a positive integer.
        fn parse_atom<I: Iterator<Item = char>>(input: &mut iter::Peekable<I>)
                                                -> Result<Noun, ParseError> {
            use std::u32;
            use num::traits::ToPrimitive;

            let mut buf = Vec::new();

            loop {
                if let Some(&c) = input.peek() {
                    if c.is_digit(10) {
                        input.next();
                        buf.push(c);
                    } else if c == '.' {
                        // Dot is used as a sequence separator (*not* as
                        // decimal point). It can show up anywhere in the
                        // digit sequence and will be ignored.
                        input.next();
                    } else if c == '[' || c == ']' || c.is_whitespace() {
                        // Whitespace or cell brackets can terminate the
                        // digit sequence.
                        break;
                    } else {
                        // Anything else in the middle of the digit sequence
                        // is an error.
                        return Err(ParseError);
                    }
                } else {
                    break;
                }
            }

            if buf.len() == 0 {
                return Err(ParseError);
            }

            let num: BigUint = buf.into_iter().collect::<String>().parse().expect("Failed to parse atom");

            if num <= BigUint::from(u32::MAX) {
                Ok(Noun::Atom(num.to_u32().expect("Failed to make 32-bit atom")))
            } else {
                Ok(Noun::BigAtom(num))
            }
        }

        /// Parse a cell, a bracketed pair of nouns.
        ///
        /// For additional complication, cells can have the form [a b c] which
        /// parses to [a [b c]].
        fn parse_cell<I: Iterator<Item = char>>(input: &mut iter::Peekable<I>)
                                                -> Result<Noun, ParseError> {
            let mut elts = Vec::new();

            if input.next() != Some('[') {
                panic!("Bad cell start");
            }

            // A cell must have at least two nouns in it.
            elts.push(try!(parse(input)));
            elts.push(try!(parse(input)));

            // It can have further trailing nouns.
            loop {
                eat_space(input);
                match input.peek().map(|&x| x) {
                    Some(c) if c.is_digit(10) => elts.push(try!(parse_atom(input))),
                    Some(c) if c == '[' => elts.push(try!(parse_cell(input))),
                    Some(c) if c == ']' => {
                        input.next();
                        break;
                    }
                    _ => return Err(ParseError),
                }
            }

            Ok(elts.into_iter().collect())
        }

        fn eat_space<I: Iterator<Item = char>>(input: &mut iter::Peekable<I>) {
            loop {
                match input.peek().map(|&x| x) {
                    Some(c) if c.is_whitespace() => {
                        input.next();
                    }
                    _ => return,
                }
            }
        }
    }
}

/// Macro for noun literals.
///
/// Rust n![1, 2, 3] corresponds to Nock [1 2 3]
#[macro_export]
macro_rules! n {
    [$x:expr, $y:expr] => { ::nock::Noun::Cell(::std::rc::Rc::new($x.into()), ::std::rc::Rc::new($y.into())) };
    [$x:expr, $y:expr, $($ys:expr),+] => { ::nock::Noun::Cell(::std::rc::Rc::new($x.into()), ::std::rc::Rc::new(n![$y, $($ys),+])) };
}

use Noun::*;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct NockError;

pub type NockResult = Result<Rc<Noun>, NockError>;

/*
fn wut(noun: Noun) -> NockResult {
    match noun {
        Cell(_, _) => Ok(Atom(0)),
        Atom(_) => Ok(Atom(1)),
    }
}

fn lus(noun: Noun) -> NockResult {
    match noun {
        Atom(n) => Ok(Atom(n + 1)),
        _ => Err(NockError),
    }
}

fn tis(noun: Noun) -> NockResult {
    match noun {
        Cell(a, b) => Ok(Atom(if a == b {
            0
        } else {
            1
        })),
        _ => Err(NockError),
    }
}

fn fas(noun: Noun) -> NockResult {
    match noun {
        Cell(box Atom(1), box a) => Ok(a),
        Cell(box Atom(2), box Cell(box a, _)) => Ok(a),
        Cell(box Atom(3), box Cell(_, box b)) => Ok(b),

        Cell(box Atom(a), b) => {
            if a <= 3 {
                Err(NockError)
            } else {
                let x = try!(fas(Cell(box Atom(a / 2), b)));
                fas(Cell(box Atom(2 + a % 2), box x))
            }
        }

        _ => Err(NockError),
    }
}

fn tar(mut noun: Noun) -> NockResult {
    // FIXME: Rust won't pattern-match the complex Cell expressions if they're
    // the top-level expression but will handle fine if they're wrapped in a
    // trivial tuple.
    loop {
        match ((), noun) {
            ((), Cell(box a, box Cell(box Cell(box b, box c), box d))) => {
                // Multiple opcodes, split up.
                let x = try!(tar(n![a.clone(), b, c]));
                let y = try!(tar(n![a, d]));
                return Ok(n![x, y]);
            }
            ((), Cell(box a, box Cell(box Atom(opcode), box args))) => {
                match opcode {
                    0 => return fas(n![args, a]),
                    1 => return Ok(args),
                    2 => {
                        if let Cell(box b, box c) = args {
                            let x = try!(tar(n![a.clone(), b]));
                            let y = try!(tar(n![a, c]));
                            noun = n![x, y];
                        } else {
                            break;
                        }
                    }
                    3 => return wut(try!(tar(n![a, args]))),
                    4 => return lus(try!(tar(n![a, args]))),
                    5 => return tis(try!(tar(n![a, args]))),
                    6 => {
                        if let ((), Cell(box b, box Cell(box c, box d))) = ((), args) {
                            noun = n![a,
                                      2,
                                      n![0, 1],
                                      2,
                                      n![1, c, d],
                                      n![1, 0],
                                      2,
                                      n![1, 2, 3],
                                      n![1, 0],
                                      4,
                                      4,
                                      b];
                        } else {
                            break;
                        }
                    }
                    7 => {
                        if let Cell(box b, box c) = args {
                            noun = n![a, 2, b, 1, c];
                        } else {
                            break;
                        }
                    }
                    8 => {
                        if let Cell(box b, box c) = args {
                            noun = n![a, 7, n![n![7, n![0, 1], b], 0, 1], c];
                        } else {
                            break;
                        }
                    }
                    9 => {
                        if let Cell(box b, box c) = args {
                            noun = n![a, 7, c, 2, n![0, 1], 0, b];
                        } else {
                            break;
                        }
                    }
                    10 => {
                        if let Cell(box Cell(_, box c), box d) = args {
                            noun = n![a, 8, c, 7, n![0, 3], d];
                        } else if let Cell(_, box c) = args {
                            noun = n![a, c]
                        } else {
                            break;
                        }
                    }
                    _ => break,
                }
            }
            _ => break,
        }
    }

    Err(NockError)
}
*/

fn tar(noun: &Noun) -> NockResult {
    if let Some((subject, ops, tail)) = noun.as_triple() {
        loop {
            match *ops {
                BigAtom(_) => {
                    // Huge opcodes are not handled.
                    return Err(NockError);
                }
                Atom(x) => {
                    return run_op(subject, x, tail)
                }
                Cell(_, _) => {
                    let a = try!(tar(&Cell(subject.clone(), ops.clone())));
                    let b = try!(tar(&Cell(subject, tail)));
                    return Ok(Rc::new(Cell(a, b)));
                }
            }
        }
    }
    return Err(NockError);

    fn run_op(subject: Rc<Noun>, op: u32, tail: Rc<Noun>) -> NockResult {
        match op {
            // Axis
            0 => {
                unimplemented!();
            }
            // Just
            1 => {
                Ok(tail)
            }

            _ => { Err(NockError) }
        }
    }
}

/// Try to represent a Nock atom as a string.
///
/// The atom is interpreted as a string of ASCII bytes in least significant
/// byte first order. The atom must evaluate entirely into printable ASCII-7.
pub fn cord(mut atom: BigUint) -> Option<String> {
    let mut ret = String::new();
    while atom > Zero::zero() {
        let ch = (atom.clone() % BigUint::from(0x100u32)).to_u8().unwrap();
        if ch >= 0x20 && ch < 0x80 {
            ret.push(ch as char);
        } else {
            return None;
        }
        atom = atom >> 8;
    }

    Some(ret)
}


// Re-export hack for testing macros.
mod nock {
    pub use Noun;
}

#[cfg(test)]
mod test {
    use std::rc::Rc;
    use num::bigint::BigUint;
    use super::Noun::{self, Atom, BigAtom, Cell};

    fn parses(input: &str, output: super::Noun) {
        assert_eq!(input.parse::<super::Noun>().ok().expect("Parsing failed"),
                   output);
    }

    fn produces(input: &str, output: &str) {
        assert_eq!(format!("{}",
                           input.parse::<Noun>()
                                .ok()
                                .expect("Parsing failed")
                                .nock()
                                .ok()
                                .expect("Eval failed")),
                   output);
    }

    #[test]
    fn test_macro() {
        assert_eq!(n![1, 2], Cell(Rc::new(Atom(1)), Rc::new(Atom(2))));
        assert_eq!(n![1, n![2, 3]],
                   Cell(Rc::new(Atom(1)), Rc::new(Cell(Rc::new(Atom(2)), Rc::new(Atom(3))))));
        assert_eq!(n![1, 2, 3],
                   Cell(Rc::new(Atom(1)), Rc::new(Cell(Rc::new(Atom(2)), Rc::new(Atom(3))))));
        assert_eq!(n![n![1, 2], 3],
                   Cell(Rc::new(Cell(Rc::new(Atom(1)), Rc::new(Atom(2)))), Rc::new(Atom(3))));
        assert_eq!(n![n![1, 2], n![3, 4]],
                   Cell(Rc::new(Cell(Rc::new(Atom(1)), Rc::new(Atom(2)))),
                        Rc::new(Cell(Rc::new(Atom(3)), Rc::new(Atom(4))))));
        assert_eq!(n![n![1, 2], n![3, 4], n![5, 6]],
                   Cell(Rc::new(Cell(Rc::new(Atom(1)), Rc::new(Atom(2)))),
                        Rc::new(Cell(Rc::new(Cell(Rc::new(Atom(3)), Rc::new(Atom(4)))),
                                 Rc::new(Cell(Rc::new(Atom(5)), Rc::new(Atom(6))))))));
    }

    #[test]
    fn test_from_iter() {
        assert_eq!(Atom(1), vec![Atom(1)].into_iter().collect());
        assert_eq!(n![1, 2], vec![Atom(1), Atom(2)].into_iter().collect());
        assert_eq!(n![1, 2, 3],
                   vec![Atom(1), Atom(2), Atom(3)].into_iter().collect());
        assert_eq!(n![1, n![2, 3]],
                   vec![Atom(1), n![2, 3]].into_iter().collect());
    }

    #[test]
    fn test_parser() {
        use num::traits::Num;

        assert!("".parse::<Noun>().is_err());
        assert!("12ab".parse::<Noun>().is_err());
        assert!("[]".parse::<Noun>().is_err());
        assert!("[1]".parse::<Noun>().is_err());

        parses("0", Atom(0));
        parses("1", Atom(1));
        parses("1.000.000", Atom(1_000_000));

        parses("4294967295", Atom(4294967295));
        parses("4294967296", BigAtom(BigUint::from(4294967296u64)));

        parses("999.999.999.999.999.999.999.999.999.999.999.999.999.999.999.999.999.999.999.999",
               BigAtom(BigUint::from_str_radix(
                       "999999999999999999999999999999999999999999999999999999999999",
                                10).unwrap()));

        parses("[1 2]", n![1, 2]);
        parses("[1 2 3]", n![1, 2, 3]);
        parses("[1 [2 3]]", n![1, 2, 3]);
        parses("[[1 2] 3]", n![n![1, 2], 3]);
    }

    #[test]
    fn test_nock() {
        produces("[42 [4 0 1] [3 0 1]]", "[43 1]");

        // Operator 0: Axis
        produces("[[19 42] [0 3] 0 2]", "[42 19]");
        produces("[[19 42] 0 3]", "42");
        produces("[[[97 2] [1 42 0]] 0 7]", "[42 0]");

        // Operator 1: Just
        produces("[42 1 57]", "57");

        // Operator 2: Fire
        produces("[[[40 43] [4 0 1]] [2 [0 4] [0 3]]]", "41");
        produces("[[[40 43] [4 0 1]] [2 [0 5] [0 3]]]", "44");
        produces("[77 [2 [1 42] [1 1 153 218]]]", "[153 218]");

        // Operator 3: Depth
        produces("[1 3 0 1]", "1");
        produces("[[2 3] 3 0 1]", "0");

        // Operator 4: Bump
        produces("[57 4 0 1]", "58");

        // Operator 5: Same
        produces("[[1 1] 5 0 1]", "0");
        produces("[[1 2] 5 0 1]", "1");

        // Operator 6: If
        produces("[[40 43] 6 [3 0 1] [4 0 2] [4 0 1]]", "41");
        produces("[42 6 [1 0] [4 0 1] 1 233]", "43");
        produces("[42 6 [1 1] [4 0 1] 1 233]", "233");

        // Operator 7: Compose
        produces("[[42 44] [7 [4 0 3] [3 0 1]]]", "1");

        // Operator 8: Push

        // Operator 9: Call

        // Operator 10: Hint

        produces("[[132 19] [10 37 [4 0 3]]]", "20");

        // Fibonacci numbers,
        // https://groups.google.com/forum/#!topic/urbit-dev/K7QpBge30JI
        produces("[10 8 [1 1 1] 8 [1 0] 8 [1 6 [5 [0 15] 4 0 6] [0 28] 9 2 [0 2] [4 0 6] [[0 29] \
                  7 [0 14] 8 [1 0] 8 [1 6 [5 [0 14] 0 6] [0 15] 9 2 [0 2] [4 0 6] [0 14] 4 0 15] \
                  9 2 0 1] 0 15] 9 2 0 1]",
                 "55");
    }

    #[test]
    fn test_stack() {
        // Subtraction. Tests tail call elimination, will trash stack if it
        // doesn't work.
        produces("[10.000 8 [1 0] 8 [1 6 [5 [0 7] 4 0 6] [0 6] 9 2 [0 2] [4 0 6] 0 7] 9 2 0 1]",
                 "9.999");
    }

    #[test]
    fn test_cord() {
        use super::cord;
        assert_eq!(cord(BigUint::from(0u32)), Some("".to_string()));
        assert_eq!(cord(BigUint::from(190u32)), None);
        assert_eq!(cord(BigUint::from(0x6f_6f66u32)), Some("foo".to_string()));
    }
}
