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
use std::default;
use std::rc::Rc;
use std::hash;
use num::bigint::BigUint;
use num::traits::{ToPrimitive, FromPrimitive, Zero, One};
use num::integer::Integer;

mod draft;
mod digit_slice;

/// A Nock noun, the basic unit of representation.
///
/// A noun is an atom or a cell. An atom is any natural number. A cell is any
/// ordered pair of nouns.
///
/// # Examples
///
/// ```
/// let subject: nock::Noun = "19".parse().unwrap();
/// let formula: nock::Noun = "[4 0 1]".parse().unwrap();
/// assert_eq!(format!("{}", nock::nock_on(&subject, &formula).unwrap()), "20");
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
    /// Bytes before the first non-zero byte in the byte slice will be
    /// ignored. This is because the most significant bit of the binary
    /// representation of an atom will always be 1.
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

    /// If the noun has structure [a [b c]], return a tuple of a, b and c.
    fn as_triple(&self) -> Option<(Rc<Noun>, Rc<Noun>, Rc<Noun>)> {
        if let &Cell(ref a, ref bc) = self {
            if let Cell(ref b, ref c) = **bc {
                return Some((a.clone(), b.clone(), c.clone()));
            }
        }
        None
    }

    /// If the noun is an atom, build a byte slice representation of it.
    pub fn to_bytes(&self) -> Option<Vec<u8>> {
        let mut ret = Vec::new();
        let mut atom;
        match self {
            &Atom(ref x) => atom = BigUint::from_u32(*x).unwrap(),
            &BigAtom(ref x) => atom = x.clone(),
            _ => return None,
        }

        while atom > Zero::zero() {
            let (a, byte) = atom.div_mod_floor(&BigUint::from_u32(0x100).unwrap());
            ret.push(byte.to_u8().unwrap());
            atom = a;
        }

        Some(ret)
    }

    /// Build either small or large atom, depending on the size of the
    /// BigUint.
    pub fn from_biguint(num: BigUint) -> Noun {
        num.to_u32().map_or(BigAtom(num), |x| Atom(x))
    }

    /// Run a memoizing fold over the noun
    pub fn fold<'a, F, T: Clone>(&'a self, mut f: F) -> T
        where F: FnMut(FoldState<'a, T>) -> T
    {
        use std::collections::HashMap;
        fn g<'a, F, T: Clone>(noun: &'a Rc<Noun>, memo: &mut HashMap<usize, T>, f: &mut F) -> T
            where F: FnMut(FoldState<'a, T>) -> T
        {
            let key = &*noun as *const _ as usize;
            if memo.contains_key(&key) {
                memo.get(&key).unwrap().clone()
            } else {
                let ret = h(noun, memo, f);
                memo.insert(key, ret.clone());
                ret
            }
        }

        fn h<'a, F, T: Clone>(noun: &'a Noun, memo: &mut HashMap<usize, T>, f: &mut F) -> T
            where F: FnMut(FoldState<'a, T>) -> T
        {
            match noun {
                &Atom(ref a) => f(FoldState::Atom(*a)),
                &BigAtom(ref a) => f(FoldState::BigAtom(a)),
                &Cell(ref p, ref q) => {
                    let p = g(p, memo, f);
                    let q = g(q, memo, f);
                    f(FoldState::Cell(p, q))
                }
            }
        }

        h(self, &mut HashMap::new(), &mut f)
    }
}

pub enum FoldState<'a, T> {
    Atom(u32),
    BigAtom(&'a BigUint),
    Cell(T, T),
}

impl default::Default for Noun {
    fn default() -> Noun {
        Noun::Atom(0)
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

impl hash::Hash for Noun {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        fn f<H: hash::Hasher>(state: &mut H, x: FoldState<u64>) -> u64 {
            match x {
                FoldState::Atom(a) => {
                    a.hash(state);
                    state.finish()
                }
                FoldState::BigAtom(a) => {
                    a.hash(state);
                    state.finish()
                }
                FoldState::Cell(p, q) => {
                    p.hash(state);
                    q.hash(state);
                    state.finish()
                }
            }
        }
        self.fold(|x| f(state, x)).hash(state);
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


/// A trait for types that can be instantiated from a Nock noun.
pub trait FromNoun: Sized {
    /// The associated error.
    type Err;

    /// Try to convert a noun to an instance of the type.
    fn from_noun(n: &Noun) -> Result<Self, Self::Err>;
}

impl FromNoun for u32 {
    type Err = ();

    fn from_noun(n: &Noun) -> Result<u32, ()> {
        match n {
            &Atom(ref n) => Ok(*n),
            _ => Err(())
        }
    }
}

impl Into<Noun> for u32 {
    fn into(self) -> Noun {
        Noun::Atom(self)
    }
}

impl FromNoun for BigUint {
    type Err = ();

    fn from_noun(n: &Noun) -> Result<BigUint, ()> {
        match n {
            &Atom(ref n) => Ok(BigUint::from_u32(*n).unwrap()),
            &BigAtom(ref n) => Ok(n.clone()),
            _ => Err(())
        }
    }
}

impl Into<Noun> for BigUint {
    fn into(self) -> Noun {
        Noun::from_biguint(self)
    }
}

// TODO: FromNoun impls for
// - primitive unsigned integer types
// - BigUint
// - Signed integer types (use the Hoon encoding)
// - (FromNoun, FromNoun) as [a b]
// - FromIterator<FromNoun> (~-terminated cell sequence)
// - The above two should get us HashMap and Vec conversion for free...
// - Strings are tricky, as we can get them from either a cord (atom as &[u8])
//   or a rope (~-terminated list of char atoms). Hm... Maybe prefer cord as
//   the "natural" conversion and have a separate function for ropes.
// - &[u8] is the base case for Strings, and also tricky since there are &[u8]
//   values which can not be represented in cord style (ones with heading zero
//   bytes). Can we do into-from for them with good conscience if the
//   roundtrip won't preserve some values?
// TODO: Implement From<X> for Noun for all X in list above.

/// Macro for noun literals.
///
/// Rust n![1, 2, 3] corresponds to Nock [1 2 3]
#[macro_export]
macro_rules! n {
    [$x:expr, $y:expr] => { $crate::Noun::Cell(::std::rc::Rc::new($x.into()), ::std::rc::Rc::new($y.into())) };
    [$x:expr, $y:expr, $($ys:expr),+] => { $crate::Noun::Cell(::std::rc::Rc::new($x.into()), ::std::rc::Rc::new(n![$y, $($ys),+])) };
}

use Noun::*;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct NockError;

pub type NockResult = Result<Rc<Noun>, NockError>;

/// Evaluate the nock `*[subject formula]`
pub fn nock_on(subject: &Noun, formula: &Noun) -> NockResult {
    tar(Cell(Rc::new(subject.clone()), Rc::new(formula.clone())))
}

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


#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::hash;
    use num::bigint::BigUint;
    use num::traits::{FromPrimitive, One};
    use test::Bencher;
    use super::Noun::{self, Atom, BigAtom, Cell};

    fn parses(input: &str, output: super::Noun) {
        assert_eq!(input.parse::<super::Noun>().ok().expect("Parsing failed"),
                   output);
    }

    fn produces(input: &str, output: &str) {
        use super::nock_on;

        let (s, f) = match input.parse::<Noun>() {
            Err(_) => panic!("Parsing failed"),
            Ok(Cell(s, f)) => (s, f),
            _ => panic!("Unnockable input"),
        };
        assert_eq!(format!("{}", nock_on(&s, &f).ok().expect("Eval failed")),
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

    fn to_cord(n: Noun) -> Option<String> {
        n.to_bytes().and_then(|b| String::from_utf8(b).ok())
    }

    #[test]
    fn test_cord() {
        assert_eq!(to_cord("0".parse::<Noun>().unwrap()), Some("".to_string()));
        assert_eq!(to_cord("190".parse::<Noun>().unwrap()), None);
        assert_eq!(to_cord("7303014".parse::<Noun>().unwrap()),
                   Some("foo".to_string()));
    }

    #[test]
    fn test_from_bytes() {
        assert_eq!(Noun::from_bytes("".as_bytes()), Noun::Atom(0));
        assert_eq!(to_cord(Noun::from_bytes("a".as_bytes())),
                   Some("a".to_string()));
        assert_eq!(to_cord(Noun::from_bytes("nock".as_bytes())),
                   Some("nock".to_string()));
        assert_eq!(Noun::from_bytes("nock".as_bytes()), Noun::Atom(1801678702));
        assert_eq!(to_cord(Noun::from_bytes("antidisestablishmentarianism".as_bytes())),
                   Some("antidisestablishmentarianism".to_string()));
    }

    fn hash<T: hash::Hash>(t: &T) -> u64 {
        use std::hash::Hasher;
        let mut s = hash::SipHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    #[test]
    fn test_fold() {
        assert_eq!(hash(&n![1, 2, 3]), hash(&n![1, 2, 3]));
        assert!(hash(&n![1, 2, 3]) != hash(&n![1, 2]));
    }
}
