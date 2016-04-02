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
extern crate fnv;

use std::collections::HashMap;
use std::rc::Rc;
use std::error::Error;
use std::str;
use std::fmt;
use std::iter;
use std::hash;
use std::default;
use num::BigUint;
pub use digit_slice::{DigitSlice, FromDigits, msb};

pub use nock::{Nock, get_axis};

mod digit_slice;
mod nock;

/// A wrapper for referencing Noun-like patterns.
#[derive(Copy, Clone)]
pub enum Shape<A, N> {
    Atom(A),
    Cell(N, N),
}

/// A Nock noun, the basic unit of representation.
///
/// A noun is an atom or a cell. An atom is any natural number. A cell is any
/// ordered pair of nouns.
///
/// Atoms are represented by a little-endian byte array of 8-bit digits.
#[derive(Clone, PartialEq, Eq)]
pub struct Noun {
    hash: u32,
    value: Inner,
}

#[derive(Clone, PartialEq, Eq)]
enum Inner {
    Atom(Rc<Vec<u8>>),
    Cell(Rc<Noun>, Rc<Noun>),
}

pub type NounShape<'a> = Shape<&'a [u8], &'a Noun>;

impl Noun {
    /// Get a shape wrapper for the noun to examine its structure.
    pub fn get<'a>(&'a self) -> NounShape<'a> {
        match self.value {
            Inner::Atom(ref v) => Shape::Atom(&v),
            Inner::Cell(ref a, ref b) => Shape::Cell(&*a, &*b),
        }
    }

    /// Pattern-match a noun with shape [p q r].
    ///
    /// The digit sequence shows the branch length of each leaf node in the
    /// expression being matched. 122 has the leftmost leaf 1 step away from
    /// the root and the two leaves on the right both 2 steps away from the
    /// root.
    pub fn get_122<'a>(&'a self) -> Option<(&'a Noun, &'a Noun, &'a Noun)> {
        if let Shape::Cell(ref a, ref b) = self.get() {
            if let Shape::Cell(ref b, ref c) = b.get() {
                return Some((a, b, c));
            }
        }
        None
    }

    /// Address of the noun's data in memory, usable as an unique identifier.
    ///
    /// Nouns with the same address are always the same, but nouns with
    /// different addresses are not guaranteed to have different values.
    #[inline]
    pub fn addr(&self) -> usize {
        &*self as *const _ as usize
    }

    pub fn mug(&self) -> u32 {
        self.hash
    }

    /// Build a new atom noun from a little-endian 8-bit digit sequence.
    pub fn atom(digits: &[u8]) -> Noun {
        Noun {
            hash: mug_atom(digits, 2_166_136_261),
            value: Inner::Atom(Rc::new(digits.to_vec())),
        }
    }

    /// Build a new cell noun from two existing nouns.
    pub fn cell(a: Noun, b: Noun) -> Noun {
        Noun {
            hash: mug_pair(a.mug(), b.mug()),
            value: Inner::Cell(Rc::new(a), Rc::new(b)),
        }
    }

    /// Build a noun from a convertible value.
    pub fn from<T: ToNoun>(item: T) -> Noun {
        item.to_noun()
    }

    /// Match noun if it's an atom that's a small integer.
    ///
    /// Will not match atoms that are larger than 2^32, but is not guaranteed
    /// to match atoms that are smaller than 2^32 but not by much.
    pub fn as_u32(&self) -> Option<u32> {
        if let Shape::Atom(ref digits) = self.get() {
            u32::from_digits(digits).ok()
        } else {
            None
        }
    }

    /// Run a memoizing fold over the noun.
    ///
    /// Each noun with an unique memory address will only be processed once, so
    /// the fold method allows efficient computation over nouns that may be
    /// extremely large logically, but involve a great deal of reuse of the same
    /// subnoun objects in practice.
    pub fn fold<'a, F, T>(&'a self, mut f: F) -> T
        where F: FnMut(Shape<&'a [u8], T>) -> T,
              T: Clone
    {
        fn h<'a, F, T, S: hash::BuildHasher>(noun: &'a Noun,
                                             memo: &mut HashMap<usize, T, S>,
                                             f: &mut F)
                                             -> T
            where F: FnMut(Shape<&'a [u8], T>) -> T,
                  T: Clone
        {
            let key = noun.addr();

            if memo.contains_key(&key) {
                memo.get(&key).unwrap().clone()
            } else {
                let ret = match noun.get() {
                    Shape::Atom(x) => f(Shape::Atom(x)),
                    Shape::Cell(ref a, ref b) => {
                        let a = h(*a, memo, f);
                        let b = h(*b, memo, f);
                        let ret = f(Shape::Cell(a, b));
                        ret
                    }
                };
                memo.insert(key, ret.clone());
                ret
            }
        }

        let fnv = hash::BuildHasherDefault::<fnv::FnvHasher>::default();
        h(self, &mut HashMap::with_hasher(fnv), &mut f)
    }

    /// Return whether a noun is a list with more than n elements.
    fn is_wider_than(&self, n: usize) -> bool {
        if n == 0 {
            true
        } else {
            match self.get() {
                Shape::Cell(_, ref a) => a.is_wider_than(n - 1),
                _ => false,
            }
        }
    }

    /// 3-letter abbrevation identifier for noun.
    fn glyph(&self) -> String {
        let alpha = "abcdefghijklmnopqrstuvwxyz".as_bytes();
        let mut ret = String::new();
        let mut mug = self.mug() as usize;
        for _ in 0..3 {
            ret.push(alpha[mug % alpha.len()] as char);
            mug /= alpha.len();
        }
        ret
    }

    /// Print the complete noun, even if it's very large.
    pub fn print_full(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f, false)
    }

    /// Print a truncated noun.
    pub fn print_abbrev(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print(f, true)
    }

    fn print(&self, f: &mut fmt::Formatter, abbrev: bool) -> fmt::Result {
        const MAX_ATOM_BITS: usize = 128;
        const MAX_CELL_WIDTH: usize = 12;

        match self.value {
            Inner::Atom(ref n) => {
                if abbrev && msb(n) > MAX_ATOM_BITS {
                    // Print huge atoms as abbreviated glyphs
                    return write!(f, "@{}", self.glyph());
                }
                // Print dot-separated integer.
                let s = format!("{}", BigUint::from_digits(n).unwrap());
                let phase = s.len() % 3;
                for (i, c) in s.chars().enumerate() {
                    if i > 0 && i % 3 == phase {
                        try!(write!(f, "."));
                    }
                    try!(write!(f, "{}", c));
                }
                Ok(())
            }

            Inner::Cell(ref a, ref b) => {
                if abbrev && self.is_wider_than(MAX_CELL_WIDTH) {
                    return write!(f, "[{}]", self.glyph());
                }

                try!(write!(f, "["));
                try!(a.print(f, abbrev));
                try!(write!(f, " "));
                // List pretty-printer.
                let mut cur = b;
                loop {
                    match cur.value {
                        Inner::Cell(ref a, ref b) => {
                            try!(a.print(f, abbrev));
                            try!(write!(f, " "));
                            cur = &b;
                        }
                        Inner::Atom(_) => {
                            try!(cur.print(f, abbrev));
                            return write!(f, "]");
                        }
                    }
                }
            }
        }
    }
}


impl default::Default for Noun {
    fn default() -> Self {
        Noun::from(0u32)
    }
}

impl hash::Hash for Noun {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl iter::FromIterator<Noun> for Noun {
    fn from_iter<T>(iterator: T) -> Self
        where T: IntoIterator<Item = Noun>
    {
        let mut v: Vec<Noun> = iterator.into_iter().collect();
        v.reverse();

        v.into_iter()
         .fold(None, move |acc, i| {
             acc.map_or_else(|| Some(i.clone()),
                             |a| Some(Noun::cell(i.clone(), a)))
         })
         .expect("Can't make noun from empty list")
    }
}

fn mug_atom(a: &[u8], init: u32) -> u32 {
    let mut c = init;
    for i in a.iter() {
        c = _fnv(*i as u32 ^ c);
    }

    let ret = (c >> 31) ^ (c & 0x7fffffff);
    if ret != 0 {
        ret
    } else {
        mug_atom(a, init.wrapping_add(1))
    }
}

fn mug_pair(p: u32, q: u32) -> u32 {
    let c = _fnv(p ^ _fnv(q));
    let ret = (c >> 31) ^ (c & 0x7fffffff);
    if ret != 0 {
        ret
    } else {
        mug_pair(p, q.wrapping_add(1))
    }
}

fn _fnv(x: u32) -> u32 {
    x.wrapping_mul(16_777_619)
}


/// Trait for types that can convert themselves to a noun.
pub trait ToNoun {
    fn to_noun(&self) -> Noun;
}

impl<T> ToNoun for T
    where T: DigitSlice
{
    fn to_noun(&self) -> Noun {
        Noun::atom(self.as_digits())
    }
}

/// A trait for types that can be instantiated from a Nock noun.
pub trait FromNoun: Sized {
    /// Try to convert a noun to an instance of the type.
    fn from_noun(n: &Noun) -> Result<Self, NockError>;
}

impl FromNoun for Noun {
    fn from_noun(n: &Noun) -> Result<Self, NockError> {
        Ok((*n).clone())
    }
}

impl FromNoun for Rc<Vec<u8>> {
    fn from_noun(n: &Noun) -> Result<Self, NockError> {
        match n.value {
            Inner::Atom(ref v) => Ok(v.clone()),
            _ => Err(NockError(format!("FromNoun Rc<Vec<u8>> not an atom"))),
        }
    }
}

impl ToNoun for Vec<u8> {
    fn to_noun(&self) -> Noun {
        Noun::atom(&self[..])
    }
}

impl<T> FromNoun for T
    where T: FromDigits
{
    fn from_noun(n: &Noun) -> Result<Self, NockError> {
        match n.get() {
            Shape::Atom(x) => {
                T::from_digits(x)
                    .map_err(|_| NockError(format!("FromNoun FromDigits")))
            }
            _ => Err(NockError(format!("FromNoun FromDigits not an atom"))),
        }
    }
}

impl<T> FromNoun for (T,)
    where T: FromNoun
{
    fn from_noun(n: &Noun) -> Result<Self, NockError> {
        Ok((try!(T::from_noun(n)),))
    }
}

impl<T, U> FromNoun for (T, U)
    where T: FromNoun,
          U: FromNoun
{
    fn from_noun(n: &Noun) -> Result<Self, NockError> {
        match n.get() {
            Shape::Cell(a, b) => {
                let t = try!(T::from_noun(a));
                let u = try!(U::from_noun(b));
                Ok((t, u))
            }
            _ => Err(NockError(format!("FromNoun (T, U) not a cell"))),
        }
    }
}

impl<T1, T2, T3> FromNoun for (T1, T2, T3)
    where T1: FromNoun,
          T2: FromNoun,
          T3: FromNoun
{
    fn from_noun(n: &Noun) -> Result<Self, NockError> {
        match n.get_122() {
            Some((t1, t2, t3)) => {
                let t1 = try!(T1::from_noun(t1));
                let t2 = try!(T2::from_noun(t2));
                let t3 = try!(T3::from_noun(t3));
                Ok((t1, t2, t3))
            }
            _ => Err(NockError(format!("FromNoun (T, U, V) not a tuple"))),
        }
    }
}

impl FromNoun for String {
    fn from_noun(n: &Noun) -> Result<Self, NockError> {
        match n.get() {
            Shape::Atom(bytes) => {
                String::from_utf8(bytes.to_vec())
                    .map_err(|_| NockError(format!("FromNoun String")))
            }
            _ => Err(NockError(format!("FromNoun String not an atom"))),
        }
    }
}

impl ToNoun for str {
    fn to_noun(&self) -> Noun {
        Noun::atom(self.as_bytes())
    }
}

impl FromNoun for bool {
    fn from_noun(n: &Noun) -> Result<Self, NockError> {
        Ok(n.as_u32() == Some(0))
    }
}

impl ToNoun for bool {
    fn to_noun(&self) -> Noun {
        Noun::from(if *self {
            0u32
        } else {
            1u32
        })
    }
}

impl<T: FromNoun> FromNoun for Vec<T> {
    // Use the Urbit convention of 0-terminated list to match Rust vectors.
    fn from_noun(mut n: &Noun) -> Result<Self, NockError> {
        let mut ret = Vec::new();

        loop {
            // List terminator.
            if n == &Noun::from(0u32) {
                return Ok(ret);
            }

            if let Shape::Cell(ref head, ref tail) = n.get() {
                ret.push(try!(T::from_noun(head)));
                n = tail;
            } else {
                return Err(NockError(format!("FromNoun Vec<T>")));
            }
        }
    }
}

// TODO: HashMap conversion.

// TODO: FromNoun/ToNoun for signed numbers using the Urbit representation
// convention.


#[derive(Clone, PartialEq, Eq, Debug)]
pub struct NockError(pub String);

impl fmt::Display for NockError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for NockError {
    fn description(&self) -> &str {
        &self.0[..]
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

pub type NockResult = Result<Noun, NockError>;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct ParseError;

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

impl Error for ParseError {
    fn description(&self) -> &str {
        "Nock parsing failed"
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

impl str::FromStr for Noun {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, ParseError> {
        return parse(&mut s.chars().peekable());

        fn parse<I: Iterator<Item = char>>(input: &mut iter::Peekable<I>)
                                           -> Result<Noun, ParseError> {
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

            Ok(Noun::from(num))
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
                    Some(c) if c.is_digit(10) => {
                        elts.push(try!(parse_atom(input)))
                    }
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

impl fmt::Display for Noun {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print_abbrev(f)
    }
}

impl fmt::Debug for Noun {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.print_abbrev(f)
    }
}


#[cfg(test)]
mod tests {
    use std::hash;
    use num::BigUint;
    use super::{Nock, Noun, Shape, FromNoun, ToNoun};

    struct VM;
    impl Nock for VM {}

    /// Macro for noun literals.
    ///
    /// Rust n![1, 2, 3] corresponds to Nock [1 2 3]
    macro_rules! n {
    [$x:expr, $y:expr] => { super::Noun::cell($x.into(), $y.into()) };
    [$x:expr, $y:expr, $($ys:expr),+] => { super::Noun::cell($x.into(), n![$y, $($ys),+]) };
    }

    // Into-conversion is only used so that we can put untyped numeric literals in
    // the noun-constructing macro and have them typed as unsigned. If the noun
    // constructor uses ToNoun, literals are assumed to be i32, which does not map
    // to atoms in quite the way we want.
    impl Into<Noun> for u64 {
        fn into(self) -> Noun {
            Noun::from(self)
        }
    }


    fn parses(input: &str, output: Noun) {
        assert_eq!(input.parse::<Noun>().ok().expect("Parsing failed"), output);
    }

    fn produces(input: &str, output: &str) {
        let (s, f) = match input.parse::<Noun>() {
            Err(_) => panic!("Parsing failed"),
            Ok(x) => {
                if let Shape::Cell(ref s, ref f) = x.get() {
                    ((*s).clone(), (*f).clone())
                } else {
                    panic!("Unnockable input")
                }
            }
        };
        assert_eq!(format!("{}", VM.nock_on(s, f).ok().expect("Eval failed")),
                   output);
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
        assert!(hash(&n![n![1, 2], 3]) != hash(&n![1, 2, 3]));
        assert!(hash(&n![1, 2, 3]) != hash(&n![1, 2]));
    }

    #[test]
    fn test_parser() {
        use num::traits::Num;

        assert!("".parse::<Noun>().is_err());
        assert!("12ab".parse::<Noun>().is_err());
        assert!("[]".parse::<Noun>().is_err());
        assert!("[1]".parse::<Noun>().is_err());

        parses("0", Noun::from(0u32));
        parses("1", Noun::from(1u32));
        parses("1.000.000", Noun::from(1_000_000u32));

        parses("4294967295", Noun::from(4294967295u32));
        parses("4294967296", Noun::from(4294967296u64));

        parses("999.999.999.999.999.999.999.999.999.999.999.999.999.999.999.\
                999.999.999.999.999",
               Noun::from(BigUint::from_str_radix("999999999999999999999999\
                                                   999999999999999999999999\
                                                   999999999999",
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
        produces("[[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \
                  23 24 25 26 27 28 29 30 31 32 33] 0 8589934591]",
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
        produces("[10 8 [1 1 1] 8 [1 0] 8 [1 6 [5 [0 15] 4 0 6] [0 28] 9 2 \
                  [0 2] [4 0 6] [[0 29] 7 [0 14] 8 [1 0] 8 [1 6 [5 [0 14] 0 \
                  6] [0 15] 9 2 [0 2] [4 0 6] [0 14] 4 0 15] 9 2 0 1] 0 15] \
                  9 2 0 1]",
                 "55");
    }

    #[test]
    fn test_stack() {
        // Subtraction. Tests tail call elimination, will trash stack if it
        // doesn't work.
        produces("[10.000 8 [1 0] 8 [1 6 [5 [0 7] 4 0 6] [0 6] 9 2 [0 2] \
                  [4 0 6] 0 7] 9 2 0 1]",
                 "9.999");
    }

    #[test]
    fn test_cord() {
        assert_eq!(String::from_noun(&Noun::from(0u32)), Ok("".to_string()));
        assert!(String::from_noun(&Noun::from(190u32)).is_err());
        assert_eq!(String::from_noun(&Noun::from(7303014u32)),
                   Ok("foo".to_string()));
        assert_eq!(String::from_noun(&"quux".to_noun()),
                   Ok("quux".to_string()));
    }

    #[test]
    fn test_mug() {
        assert_eq!(Noun::from(0u32).mug(), 18_652_612);
        assert_eq!(Noun::from(1u32).mug(), 67_918_732);
        assert_eq!(Noun::from(126u32).mug(), 2_064_403_808);
        assert_eq!(Noun::from(10_000u32).mug(), 178_152_889);
        assert_eq!(Noun::from(10_001u32).mug(), 714_838_017);
        assert_eq!("123.456.789.123.456.789".parse::<Noun>().unwrap().mug(),
                   322_093_503);
        assert_eq!("123.456.789.123.456.789.123.456.789"
                       .parse::<Noun>()
                       .unwrap()
                       .mug(),
                   61_582_623);
        assert_eq!(n![1, 2, 3, 4, 5, 0].mug(), 1_067_931_605);
        // This would get mugged to zero without the nonzero check.
        assert_eq!(Noun::from(2_048_341_237u32).mug(), 1_229_723_070);
    }
}
