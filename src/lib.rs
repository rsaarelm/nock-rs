#![crate_name="nock"]

#![feature(box_syntax, box_patterns)]

#[macro_use]
extern crate log;

use std::fmt;
use std::iter;

/// A Nock noun, the basic unit of representation.
///
/// The noun is either an atom a that's a positive integer or a pair of two
/// nouns, [a b].
#[derive(Clone, PartialEq, Eq)]
pub enum Noun {
    Atom(u64), // TODO: Need bigints?
    Cell(Box<Noun>, Box<Noun>),
}

impl Noun {
    pub fn new<T: Into<Noun>>(val: T) -> Noun {
        val.into()
    }
}

impl Into<Noun> for u64 {
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
                             |a| Some(Noun::Cell(box i.clone(), box a)))
         })
         .expect("Can't make noun from empty list")
    }
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

/// Macro for noun literals.
///
/// Rust n![1, 2, 3] corresponds to Nock [1 2 3]
macro_rules! n {
    [$x:expr, $y:expr] => { ::nock::Noun::Cell(box ::nock::Noun::new($x), box ::nock::Noun::new($y)) };
    [$x:expr, $y:expr, $($ys:expr),+] => { ::nock::Noun::Cell(box ::nock::Noun::new($x), box n![$y, $($ys),+]) };
}

use Noun::*;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Bottom;

pub type NockResult = Result<Noun, Bottom>;

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

fn wut(noun: Noun) -> NockResult {
    info!("?{}", noun);
    match noun {
        Cell(_, _) => Ok(a(0)),
        Atom(_) => Ok(a(1)),
    }
}

fn lus(noun: Noun) -> NockResult {
    info!("+{}", noun);
    match noun {
        Atom(n) => Ok(a(n + 1)),
        _ => Err(Bottom),
    }
}

fn tis(noun: Noun) -> NockResult {
    info!("={}", noun);
    match noun {
        Cell(a, b) => Ok(Atom(if a == b {
            0
        } else {
            1
        })),
        _ => Err(Bottom),
    }
}

fn fas(noun: Noun) -> NockResult {
    info!("/{}", noun);
    match noun {
        Cell(box Atom(1), box a) => Ok(a),
        Cell(box Atom(2), box Cell(box a, _)) => Ok(a),
        Cell(box Atom(3), box Cell(_, box b)) => Ok(b),

        Cell(box Atom(a), b) => {
            if a <= 3 {
                Err(Bottom)
            } else {
                let x = try!(fas(Cell(box Atom(a / 2), b)));
                fas(Cell(box Atom(2 + a % 2), box x))
            }
        }

        _ => Err(Bottom),
    }
}

pub fn tar(noun: Noun) -> NockResult {
    // FIXME: Rust won't pattern-match the complex Cell expressions if they're
    // the top-level expression but will handle fine if they're wrapped in a
    // trivial tuple.
    info!("*{}", noun);
    match ((), noun) {
        ((), Cell(box a, box Cell(box Cell(box b, box c), box d))) => {
            let x = try!(tar(n![a.clone(), b, c]));
            let y = try!(tar(n![a, d]));
            Ok(n![x, y])
        }

        ((), Cell(a, box Cell(box Atom(0), b))) => fas(Cell(b, a)),

        ((), Cell(_a, box Cell(box Atom(1), box b))) => Ok(b),

        ((), Cell(a, box Cell(box Atom(2), box Cell(b, c)))) => {
            let x = try!(tar(Cell(a.clone(), b)));
            let y = try!(tar(Cell(a, c)));
            tar(Cell(box x, box y))
        }

        ((), Cell(a, box Cell(box Atom(3), b))) => {
            let x = try!(tar(Cell(a, b)));
            wut(x)
        }

        ((), Cell(a, box Cell(box Atom(4), b))) => {
            let x = try!(tar(Cell(a, b)));
            lus(x)
        }

        ((), Cell(a, box Cell(box Atom(5), b))) => {
            let x = try!(tar(Cell(a, b)));
            tis(x)
        }

        ((), Cell(box a, box Cell(box Atom(6), box Cell(box b, box Cell(box c, box d))))) =>
            tar(n![a, 2, n![0, 1], 2, n![1, c, d], n![1, 0], 2, n![1, 2, 3], n![1, 0], 4, 4, b]),

        ((), Cell(box a, box Cell(box Atom(7), box Cell(box b, box c)))) => tar(n![a, 2, b, 1, c]),

        ((), Cell(box a, box Cell(box Atom(8), box Cell(box b, box c)))) =>
            tar(n![a, 7, n![n![7, n![0, 1], b], 0, 1], c]),

        ((), Cell(box a, box Cell(box Atom(9), box Cell(box b, box c)))) =>
            tar(n![a, 7, c, 2, n![0, 1], 0, b]),

        ((), Cell(box a, box Cell(box Atom(10), box Cell(box Cell(_b, box c), box d)))) =>
            tar(n![a, 8, c, 7, n![0, 3], d]),

        ((), Cell(box a, box Cell(box Atom(10), box Cell(_b, box c)))) => tar(n![a, c]),

        _ => Err(Bottom),
    }
}


pub fn a(val: u64) -> Noun {
    Noun::Atom(val)
}

/// Parse tokens
enum Tok {
    Sel, // [
    Ser, // ]
    Wut, // ?
    Lus, // +
    Tis, // =
    Fas, // /
    Tar, // *
    Atom(u64),

    Error(String),
}

struct Tokenizer<I: Iterator> {
    input: iter::Peekable<I>,
}

impl<I: Iterator<Item=char>> Tokenizer<I> {
    pub fn new(input: I) -> Tokenizer<I> {
        Tokenizer { input: input.peekable() }
    }
}

impl<I: Iterator<Item=char>> Iterator for Tokenizer<I> {
    type Item = Tok;

    fn next(&mut self) -> Option<Tok> {
        use Tok::*;

        let peek = self.input.peek().map(|&x| x);
        match peek {
            None => {
                None
            }
            Some(x) if x.is_whitespace() => {
                self.input.next();
                self.next()
            }

            // Read as many consecutive digits as there are.
            Some(x) if x.is_digit(10) => {
                let mut buf = vec![x];
                // XXX: Can this be made shorter?

                // TODO: Bignum issues.
                loop {
                    self.input.next();
                    if let Some(&c) = self.input.peek() {
                        // Take in digits.
                        if c.is_digit(10) {
                            buf.push(c);
                        } else if c == '.' {
                            // Dot is used as a sequence separator (*not* as
                            // decimal point). It can show up anywhere in the
                            // digit sequence and will be ignored.
                        } else if c == '[' || c == ']' || c.is_whitespace() {
                            // Whitespace or cell brackets can terminate the
                            // digit sequence.
                            break;
                        } else {
                            // Anything else in the middle of the digit sequence
                            // is an error.
                            buf.push(c);
                            return Some(Error(buf.into_iter().collect()));
                        }
                    } else {
                        break;
                    }
                }

                Some(Atom(buf.into_iter()
                             .collect::<String>()
                             .parse::<u64>()
                             .unwrap()))
            }

            Some('[') => {
                self.input.next();
                Some(Sel)
            }
            Some(']') => {
                self.input.next();
                Some(Ser)
            }
            Some('?') => {
                self.input.next();
                Some(Wut)
            }
            Some('+') => {
                self.input.next();
                Some(Lus)
            }
            Some('=') => {
                self.input.next();
                Some(Tis)
            }
            Some('/') => {
                self.input.next();
                Some(Fas)
            }
            Some('*') => {
                self.input.next();
                Some(Tar)
            }

            // XXX: Is there a better way to handle errors?
            Some(c) => {
                self.input.next();
                Some(Error(Some(c).into_iter().collect()))
            }
        }
    }
}

pub fn parse(input: &str) -> NockResult {
    parse_tokens(&mut Tokenizer::new(input.chars()))
}

/// Parses either a noun or a formula.
///
/// Formulas will be evaluated on the spot. Failure to evaluate the formula
/// will result an error.
fn parse_tokens<I: Iterator<Item = Tok>>(input: &mut I) -> NockResult {
    use Tok::*;
    match input.next() {
        Some(Sel) => parse_cell(input),
        Some(Wut) => wut(try!(parse_noun(input))),
        Some(Lus) => lus(try!(parse_noun(input))),
        Some(Tis) => tis(try!(parse_noun(input))),
        Some(Fas) => fas(try!(parse_noun(input))),
        Some(Tar) => tar(try!(parse_noun(input))),
        Some(Atom(n)) => Ok(Noun::Atom(n)),
        _ => Err(Bottom),
    }
}

/// Parses a cell, must have at least two nouns inside.
fn parse_cell<I: Iterator<Item = Tok>>(input: &mut I) -> NockResult {
    let mut elts = Vec::new();
    // Must have at least two formulas/nouns inside.
    elts.push(try!(parse_tokens(input)));
    elts.push(try!(parse_tokens(input)));
    // Then can have zero to many further tail nouns.
    loop {
        match parse_cell_tail(input) {
            Some(n) => elts.push(try!(n)),
            None => break,
        }
    }

    Ok(elts.into_iter().collect())
}

/// Parses either an end of cell or a further element.
fn parse_cell_tail<I: Iterator<Item = Tok>>(input: &mut I) -> Option<NockResult> {
    use Tok::*;
    match input.next() {
        Some(Ser) => None,
        Some(Sel) => Some(parse_cell(input)),

        Some(Wut) => parse_noun(input).ok().map_or(Some(Err(Bottom)), |n| Some(wut(n))),
        Some(Lus) => parse_noun(input).ok().map_or(Some(Err(Bottom)), |n| Some(lus(n))),
        Some(Tis) => parse_noun(input).ok().map_or(Some(Err(Bottom)), |n| Some(tis(n))),
        Some(Fas) => parse_noun(input).ok().map_or(Some(Err(Bottom)), |n| Some(fas(n))),
        Some(Tar) => parse_noun(input).ok().map_or(Some(Err(Bottom)), |n| Some(tar(n))),

        Some(Atom(n)) => Some(Ok(Noun::Atom(n))),
        _ => Some(Err(Bottom)),
    }
}

fn parse_noun<I: Iterator<Item = Tok>>(input: &mut I) -> NockResult {
    use Tok::*;
    match input.next() {
        Some(Sel) => parse_cell(input),
        Some(Atom(n)) => Ok(Noun::Atom(n)),
        _ => Err(Bottom),
    }
}

// Re-export hack for testing macros.
mod nock {
    pub use Noun;
}

#[cfg(test)]
mod test {
    fn produces(input: &str, output: &str) {
        assert_eq!(
            format!("{}", super::parse(input).expect("Syntax error")), output);
    }

    fn fails(input: &str) {
        assert!(super::parse(input).is_err());
    }

    #[test]
    fn test_macro() {
        use super::Noun::*;

        assert_eq!(n![1, 2], Cell(box Atom(1), box Atom(2)));
        assert_eq!(n![1, n![2, 3]],
                   Cell(box Atom(1), box Cell(box Atom(2), box Atom(3))));
        assert_eq!(n![1, 2, 3],
                   Cell(box Atom(1), box Cell(box Atom(2), box Atom(3))));
        assert_eq!(n![n![1, 2], 3],
                   Cell(box Cell(box Atom(1), box Atom(2)), box Atom(3)));
        assert_eq!(n![n![1, 2], n![3, 4]],
                   Cell(box Cell(box Atom(1), box Atom(2)),
                        box Cell(box Atom(3), box Atom(4))));
        assert_eq!(n![n![1, 2], n![3, 4], n![5, 6]],
                   Cell(box Cell(box Atom(1), box Atom(2)),
                        box Cell(box Cell(box Atom(3), box Atom(4)),
                                 box Cell(box Atom(5), box Atom(6)))));
    }

    #[test]
    fn test_from_iter() {
        use super::Noun;
        use super::a;

        assert_eq!(a(1), vec![a(1)].into_iter().collect::<Noun>());
        assert_eq!(n![1, 2], vec![a(1), a(2)].into_iter().collect::<Noun>());
        assert_eq!(n![1, 2, 3],
                   vec![a(1), a(2), a(3)].into_iter().collect::<Noun>());
        assert_eq!(n![1, n![2, 3]],
                   vec![a(1), n![2, 3]].into_iter().collect::<Noun>());
    }

    #[test]
    fn test_eval() {
        produces("=[0 0]", "0");
        produces("=[2 2]", "0");
        produces("=[0 1]", "1");

        produces("/[1 [97 2] [1 42 0]]", "[[97 2] 1 42 0]");
        produces("/[2 [97 2] [1 42 0]]", "[97 2]");

        produces("/[3 [97 2] [1 42 0]]", "[1 42 0]");
        produces("/[6 [97 2] [1 42 0]]", "1");
        produces("/[7 [97 2] [1 42 0]]", "[42 0]");

        produces("+4", "5");

        produces("*[[19 42] [0 3] 0 2]", "[42 19]");

        fails("+[1 2]");
        fails("/[3 8]");
    }

    #[test]
    fn test_parse_noun() {
        use super::parse;

        assert!(parse("").is_err());
        assert!(parse("[]").is_err());
        assert!(parse("[1]").is_err());
        assert!(parse("[x y]").is_err());
        assert_eq!(parse("[1 2]"), Ok(n![1, 2]));
        assert_eq!(parse("[1 2 3]"), Ok(n![1, 2, 3]));
        assert_eq!(parse("[[1 2] 3]"), Ok(n![n![1, 2], 3]));
        assert_eq!(parse("[1 [2 3]]"), Ok(n![1, n![2, 3]]));
    }
}
