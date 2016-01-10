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

#![feature(test)]
extern crate test;
extern crate num;

use std::fmt;
use std::iter;
use std::str;
use std::rc::Rc;
use num::bigint::BigUint;
use num::traits::{ToPrimitive, FromPrimitive, Zero, One};

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
    /// A single positive large integer
    BigAtom(BigUint), // Might want to make this Rc too to ensure Noun stays lightweight.
    /// A a pair of two nouns
    Cell(Rc<Noun>, Rc<Noun>),
}

impl Noun {
    /// Construct an atom equivalent to a list of bytes.
    ///
    /// If the bytes are a text string, the atom will be a cord with that
    /// text.
    pub fn from_bytes(bytes: &[u8]) -> Noun {
        let mut x = Zero::zero();
        for i in bytes.iter().rev() {
            x = x << 8;
            x = x + BigUint::from_u8(*i).unwrap();
        }

        Noun::from_biguint(x)
    }

    /// Evaluate the noun using the function `nock(a) = *a` as defined in
    /// the Nock spec.
    pub fn nock(self) -> NockResult {
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

    /// Try to represent a Nock atom as a string.
    ///
    /// The atom is interpreted as a string of ASCII bytes in least significant
    /// byte first order. The atom must evaluate entirely into printable ASCII-7.
    pub fn to_cord(&self) -> Option<String> {
        let atom;
        match self {
            &Atom(ref x) => atom = BigUint::from_u32(*x).unwrap(),
            &BigAtom(ref x) => atom = x.clone(),
            _ => return None,
        }

        parse_cord(atom)
    }

    /// Build either small or large atom, depending on the size of the
    /// BigUint.
    pub fn from_biguint(num: BigUint) -> Noun {
        num.to_u32().map_or(BigAtom(num), |x| Atom(x))
    }
}

impl Into<Noun> for u32 {
    fn into(self) -> Noun {
        Noun::Atom(self)
    }
}

impl Into<Noun> for BigUint {
    fn into(self) -> Noun {
        Noun::from_biguint(self)
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

            let num: BigUint = buf.into_iter()
                                  .collect::<String>()
                                  .parse()
                                  .expect("Failed to parse atom");

            Ok(Noun::from_biguint(num))
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

fn parse_cord(mut atom: BigUint) -> Option<String> {
    let mut ret = String::new();
    while atom > Zero::zero() {
        let ch = (atom.clone() % BigUint::from_u32(0x100).unwrap()).to_u8().unwrap();
        if ch >= 0x20 && ch < 0x80 {
            ret.push(ch as char);
        } else {
            return None;
        }
        atom = atom >> 8;
    }

    Some(ret)
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

fn tar(mut noun: Noun) -> NockResult {
    use std::u32;
    loop {
        if let Some((subject, ops, tail)) = noun.as_triple() {
            match *ops {
                BigAtom(_) => {
                    // Huge opcodes are not handled.
                    return Err(NockError);
                }
                Atom(op) => {
                    match op {
                        // Axis
                        0 => {
                            match *tail {
                                Atom(ref x) => return axis(*x, subject),
                                BigAtom(ref x) => return big_axis(x.clone(), subject),
                                _ => return Err(NockError),
                            }
                        }
                        // Just
                        1 => return Ok(tail),
                        // Fire
                        2 => {
                            match *tail {
                                Cell(ref b, ref c) => {
                                    let p = try!(tar(Cell(subject.clone(), b.clone())));
                                    let q = try!(tar(Cell(subject, c.clone())));
                                    noun = Cell(p, q);
                                    continue;
                                }
                                _ => return Err(NockError),
                            }
                        }
                        // Depth
                        3 => {
                            let p = try!(tar(Cell(subject, tail)));
                            return match *p {
                                Cell(_, _) => Ok(Rc::new(Atom(0))),
                                _ => Ok(Rc::new(Atom(1))),
                            };
                        }
                        // Bump
                        4 => {
                            let p = try!(tar(Cell(subject, tail)));
                            return match *p {
                                // Switch to BigAtoms at regular atom size limit.
                                Atom(u32::MAX) => {
                                    Ok(Rc::new(BigAtom(BigUint::from_u32(u32::MAX).unwrap() +
                                                       BigUint::one())))
                                }
                                Atom(ref x) => Ok(Rc::new(Atom(x + 1))),
                                BigAtom(ref x) => Ok(Rc::new(BigAtom(x + BigUint::one()))),
                                _ => Err(NockError),
                            };
                        }
                        // Same
                        5 => {
                            let p = try!(tar(Cell(subject, tail)));
                            return match *p {
                                Cell(ref a, ref b) => {
                                    if a == b {
                                        return Ok(Rc::new(Atom(0)));
                                    } else {
                                        return Ok(Rc::new(Atom(1)));
                                    }
                                }
                                _ => return Err(NockError),
                            };
                        }

                        // If
                        6 => {
                            if let Some((b, c, d)) = tail.as_triple() {
                                let p = try!(tar(Cell(subject.clone(), b)));
                                match *p {
                                    Atom(0) => noun = Cell(subject, c),
                                    Atom(1) => noun = Cell(subject, d),
                                    _ => return Err(NockError),
                                }
                                continue;
                            } else {
                                return Err(NockError);
                            }
                        }

                        // Compose
                        7 => {
                            match *tail {
                                Cell(ref b, ref c) => {
                                    let p = try!(tar(Cell(subject, b.clone())));
                                    noun = Cell(p, c.clone());
                                    continue;
                                }
                                _ => return Err(NockError),
                            }
                        }

                        // Push
                        8 => {
                            match *tail {
                                Cell(ref b, ref c) => {
                                    let p = try!(tar(Cell(subject.clone(), b.clone())));
                                    noun = Cell(Rc::new(Cell(p, subject)), c.clone());
                                    continue;
                                }
                                _ => return Err(NockError),
                            }
                        }

                        // Call
                        9 => {
                            match *tail {
                                Cell(ref b, ref c) => {
                                    let p = try!(tar(Cell(subject.clone(), c.clone())));
                                    let q = try!(tar(Cell(p.clone(),
                                                          Rc::new(Cell(Rc::new(Atom(0)),
                                                                       b.clone())))));
                                    noun = Cell(p, q);
                                    continue;
                                }
                                _ => return Err(NockError),
                            }
                        }

                        // Hint
                        10 => {
                            match *tail {
                                Cell(ref _b, ref c) => {
                                    // Throw away b.
                                    // XXX: Should check if b is a cell and fail if it
                                    // would crash.
                                    noun = Cell(subject, c.clone());
                                    continue;
                                }
                                _ => return Err(NockError),
                            }
                        }

                        _ => return Err(NockError),
                    }
                }
                Cell(_, _) => {
                    // Autocons
                    let a = try!(tar(Cell(subject.clone(), ops.clone())));
                    let b = try!(tar(Cell(subject, tail)));
                    return Ok(Rc::new(Cell(a, b)));
                }
            }
        }
        return Err(NockError);
    }
}

fn axis(x: u32, noun: Rc<Noun>) -> NockResult {
    match x {
        0 => Err(NockError),
        1 => Ok(noun),
        n => {
            match *noun {
                Cell(ref a, ref b) => {
                    if n == 2 {
                        Ok(a.clone())
                    } else if n == 3 {
                        Ok(b.clone())
                    } else {
                        let p = try!(axis(x / 2, noun.clone()));
                        if n % 2 == 0 {
                            axis(2, p)
                        } else {
                            axis(3, p)
                        }
                    }
                }
                _ => Err(NockError),
            }
        }
    }
}

fn big_axis(x: BigUint, noun: Rc<Noun>) -> NockResult {
    // Assuming x is actually big, will switch to regular axis when we go down
    // in size.
    if let Cell(_, _) = *noun {
        let half = x.clone() >> 1;
        let p = try!(if half.bits() < 30 {
            axis(half.to_u32().unwrap(), noun.clone())
        } else {
            big_axis(half, noun.clone())
        });

        if x % BigUint::from_u32(2).unwrap() == BigUint::zero() {
            axis(2, p)
        } else {
            axis(3, p)
        }
    } else {
        Err(NockError)
    }
}


// Re-export hack for testing macros.
mod nock {
    pub use Noun;
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use num::bigint::BigUint;
    use num::traits::{FromPrimitive, One};
    use test::Bencher;
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
    fn test_from_biguint() {
        assert_eq!(Noun::from_biguint(BigUint::one()), Atom(1));
        assert_eq!(Noun::from_biguint(BigUint::one()), Atom(1));

        let small = BigUint::from_u64(4294967295).unwrap();
        assert_eq!(Noun::from_biguint(small), Atom(4294967295));

        let big = BigUint::from_u64(4294967296).unwrap();
        assert_eq!(Noun::from_biguint(big.clone()), BigAtom(big));
    }

    #[test]
    fn test_macro() {
        assert_eq!(n![1, 2], Cell(Rc::new(Atom(1)), Rc::new(Atom(2))));
        assert_eq!(n![1, n![2, 3]],
                   Cell(Rc::new(Atom(1)),
                        Rc::new(Cell(Rc::new(Atom(2)), Rc::new(Atom(3))))));
        assert_eq!(n![1, 2, 3],
                   Cell(Rc::new(Atom(1)),
                        Rc::new(Cell(Rc::new(Atom(2)), Rc::new(Atom(3))))));
        assert_eq!(n![n![1, 2], 3],
                   Cell(Rc::new(Cell(Rc::new(Atom(1)), Rc::new(Atom(2)))),
                        Rc::new(Atom(3))));
        assert_eq!(n![n![1, 2], n![3, 4]],
                   Cell(Rc::new(Cell(Rc::new(Atom(1)), Rc::new(Atom(2)))),
                        Rc::new(Cell(Rc::new(Atom(3)), Rc::new(Atom(4))))));
        assert_eq!(n![n![1, 2], n![3, 4], n![5, 6]],
                   Cell(Rc::new(Cell(Rc::new(Atom(1)), Rc::new(Atom(2)))),
                        Rc::new(Cell(Rc::new(Cell(Rc::new(Atom(3)), Rc::new(Atom(4)))),
                                     Rc::new(Cell(Rc::new(Atom(5)), Rc::new(Atom(6))))))));

        let big = BigUint::from_u64(4294967296).unwrap();
        assert_eq!(n![0, big.clone()],
                   Cell(Rc::new(Atom(0)), Rc::new(BigAtom(big))));
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
        parses("4294967296",
               BigAtom(BigUint::from_u64(4294967296).unwrap()));

        parses("999.999.999.999.999.999.999.999.999.999.999.999.999.999.999.999.999.999.999.999",
               BigAtom(BigUint::from_str_radix("99999999999999999999999999999999999999999999999\
                                                9999999999999",
                                               10)
                           .unwrap()));

        parses("[1 2]", n![1, 2]);
        parses("[1 2 3]", n![1, 2, 3]);
        parses("[1 [2 3]]", n![1, 2, 3]);
        parses("[[1 2] 3]", n![n![1, 2], 3]);
    }

    #[test]
    fn test_autocons() {
        produces("[42 [4 0 1] [3 0 1]]", "[43 1]");
    }

    #[test]
    fn test_axis() {
        // Operator 0: Axis
        produces("[[19 42] [0 3] 0 2]", "[42 19]");
        produces("[[19 42] 0 3]", "42");
        produces("[[[97 2] [1 42 0]] 0 7]", "[42 0]");

        // Bignum axis.
        produces("[[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 \
                  29 30 31 32 33] 0 8589934591]",
                 "33");
    }

    #[test]
    fn test_just() {
        // Operator 1: Just
        produces("[42 1 57]", "57");
    }

    #[test]
    fn test_fire() {
        // Operator 2: Fire
        produces("[[[40 43] [4 0 1]] [2 [0 4] [0 3]]]", "41");
        produces("[[[40 43] [4 0 1]] [2 [0 5] [0 3]]]", "44");
        produces("[77 [2 [1 42] [1 1 153 218]]]", "[153 218]");
    }

    #[test]
    fn test_depth() {
        // Operator 3: Depth
        produces("[1 3 0 1]", "1");
        produces("[[2 3] 3 0 1]", "0");
    }

    #[test]
    fn test_bump() {
        // Operator 4: Bump
        produces("[57 4 0 1]", "58");
    }

    #[test]
    fn test_bigint() {
        // 32-bit limit, bump up needs bignums if atom is u32
        produces("[4294967295 4 0 1]", "4.294.967.296");
        // 64-bit limit, bump up needs bignums if atom is u64
        produces("[18446744073709551615 4 0 1]", "18.446.744.073.709.551.616");
        // Bignum-to-bignum bump works, even if base atoms are 64-bit.
        produces("[18446744073709551616 4 0 1]", "18.446.744.073.709.551.617");
    }

    #[test]
    fn test_same() {
        // Operator 5: Same
        produces("[[1 1] 5 0 1]", "0");
        produces("[[1 2] 5 0 1]", "1");
        // Various bignum combinations.
        produces("[[18446744073709551615 18446744073709551615] 5 0 1]", "0");
        produces("[[18446744073709551615 18446744073709551616] 5 0 1]", "1");
        produces("[[18446744073709551615 2] 5 0 1]", "1");
        produces("[[2 18446744073709551615] 5 0 1]", "1");
    }

    #[test]
    fn test_if() {
        // Operator 6: If
        produces("[[40 43] 6 [3 0 1] [4 0 2] [4 0 1]]", "41");
        produces("[42 6 [1 0] [4 0 1] 1 233]", "43");
        produces("[42 6 [1 1] [4 0 1] 1 233]", "233");
    }

    #[test]
    fn test_misc_nock() {
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

    #[bench]
    fn test_stack(b: &mut Bencher) {
        // Subtraction. Tests tail call elimination, will trash stack if it
        // doesn't work.
        b.iter(|| {
            produces("[10.000 8 [1 0] 8 [1 6 [5 [0 7] 4 0 6] [0 6] 9 2 [0 2] [4 0 6] 0 7] 9 2 0 1]",
                     "9.999")
        })
    }

    #[test]
    fn test_cord() {
        assert_eq!("0".parse::<Noun>().unwrap().to_cord(), Some("".to_string()));
        assert_eq!("190".parse::<Noun>().unwrap().to_cord(), None);
        assert_eq!("7303014".parse::<Noun>().unwrap().to_cord(),
                   Some("foo".to_string()));
    }

    #[test]
    fn test_from_bytes() {
        assert_eq!(Noun::from_bytes("".as_bytes()), Noun::Atom(0));
        assert_eq!(Noun::from_bytes("a".as_bytes()).to_cord(),
                   Some("a".to_string()));
        assert_eq!(Noun::from_bytes("nock".as_bytes()).to_cord(),
                   Some("nock".to_string()));
        assert_eq!(Noun::from_bytes("nock".as_bytes()), Noun::Atom(1801678702));
        assert_eq!(Noun::from_bytes("antidisestablishmentarianism".as_bytes()).to_cord(),
                   Some("antidisestablishmentarianism".to_string()));
    }
}
