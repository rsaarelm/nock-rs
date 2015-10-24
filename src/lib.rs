#![crate_name="nock"]

#![feature(box_syntax, box_patterns)]

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


/// An operator for a Nock formula.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Op {
    Wut, // ?
    Lus, // +
    Tis, // =
    Fas, // /
    Tar, // *
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Op::*;
        write!(f,
               "{}",
               match self {
                   &Wut => '?',
                   &Lus => '+',
                   &Tis => '=',
                   &Fas => '/',
                   &Tar => '*',
               })
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Formula(pub Op, pub Noun);

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.0, self.1)
    }
}

impl fmt::Debug for Formula {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

use Op::*;
use Noun::*;

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

impl Formula {
    pub fn eval(self) -> Result<Noun, Formula> {
        match (self.0, self.1) {
            (Wut, Cell(_, _)) => Ok(Atom(0)),
            (Wut, Atom(_)) => Ok(Atom(1)),

            // Math
            (Lus, Atom(a)) => Ok(Atom(1 + a)),

            (Lus, a) => Err(Formula(Lus, a)),
            (Tis, Cell(a, b)) => Ok(Atom(if a == b {
                0
            } else {
                1
            })),
            (Tis, a) => Err(Formula(Tis, a)),

            (Fas, Cell(box Atom(1), box a)) => Ok(a),
            (Fas, Cell(box Atom(2), box Cell(box a, _))) => Ok(a),
            (Fas, Cell(box Atom(3), box Cell(_, box b))) => Ok(b),

            // Math
            (Fas, Cell(box Atom(a), b)) => {
                let x = try!(Formula(Fas, Cell(box Atom(a / 2), b)).eval());
                eval(Fas,
                     Cell(box Atom(if a % 2 == 0 {
                              2
                          } else {
                              3
                          }),
                          box x))
            }

            (Fas, a) => Err(Formula(Fas, a)),

            (Tar, Cell(box a, box Cell(box Cell(box b, box c), box d))) => {
                let x = try!(Formula(Tar, n![a.clone(), b, c]).eval());
                let y = try!(Formula(Tar, n![a, d]).eval());
                Ok(n![x, y])
            }

            (Tar, Cell(a, box Cell(box Atom(0), b))) => Formula(Fas, Cell(b, a)).eval(),

            (Tar, Cell(_a, box Cell(box Atom(1), box b))) => Ok(b),

            (Tar, Cell(a, box Cell(box Atom(2), box Cell(b, c)))) => {
                let x = try!(Formula(Tar, Cell(a.clone(), b)).eval());
                let y = try!(Formula(Tar, Cell(a, c)).eval());
                Formula(Tar, Cell(box x, box y)).eval()
            }

            (Tar, Cell(a, box Cell(box Atom(3), b))) => {
                let x = try!(Formula(Tar, Cell(a, b)).eval());
                Formula(Wut, x).eval()
            }

            (Tar, Cell(a, box Cell(box Atom(4), b))) => {
                let x = try!(Formula(Tar, Cell(a, b)).eval());
                Formula(Lus, x).eval()
            }

            (Tar, Cell(a, box Cell(box Atom(5), b))) => {
                let x = try!(Formula(Tar, Cell(a, b)).eval());
                Formula(Tis, x).eval()
            }

            (Tar,
             Cell(box a, box Cell(box Atom(6), box Cell(box b, box Cell(box c, box d))))) =>
                Formula(Tar,
                        n![a,
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
                           b])
                    .eval(),

            (Tar,
             Cell(box a, box Cell(box Atom(7), box Cell(box b, box c)))) =>
                Formula(Tar, n![a, 2, b, 1, c]).eval(),

            (Tar,
             Cell(box a, box Cell(box Atom(8), box Cell(box b, box c)))) =>
                Formula(Tar, n![a, 7, n![n![7, n![0, 1], b], 0, 1], c]).eval(),

            (Tar,
             Cell(box a, box Cell(box Atom(9), box Cell(box b, box c)))) =>
                Formula(Tar, n![a, 7, c, 2, n![0, 1], 0, b]).eval(),

            (Tar,
             Cell(box a, box Cell(box Atom(10), box Cell(box Cell(_b, box c), box d)))) =>
                Formula(Tar, n![a, 8, c, 7, n![0, 3], d]).eval(),

            (Tar,
             Cell(box a, box Cell(box Atom(10), box Cell(_b, box c)))) =>
                Formula(Tar, n![a, c]).eval(),

            (Tar, a) => Err(Formula(Tar, a)),
        }
    }
}

pub fn eval(op: Op, noun: Noun) -> Result<Noun, Formula> {
    Formula(op, noun).eval()
}

pub fn nock(noun: Noun) -> Result<Noun, Formula> {
    eval(Wut, noun)
}

pub fn a(val: u64) -> Noun {
    Noun::Atom(val)
}

/// Parse tokens
enum Tok {
    Sel, // [
    Ser, // ]
    Op(Op),
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
        use Op::*;

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
                            // Whitespace or cell brackets can terminate the
                            // digit sequence.
                        } else if c == '[' || c == ']' || c.is_whitespace() {
                            break;
                            // Anything else in the middle of the digit sequence
                            // is an error.
                        } else {
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
                Some(Op(Wut))
            }
            Some('+') => {
                self.input.next();
                Some(Op(Lus))
            }
            Some('=') => {
                self.input.next();
                Some(Op(Tis))
            }
            Some('/') => {
                self.input.next();
                Some(Op(Fas))
            }
            Some('*') => {
                self.input.next();
                Some(Op(Tar))
            }

            // XXX: Is there a better way to handle errors?
            Some(c) => {
                self.input.next();
                Some(Error(Some(c).into_iter().collect()))
            }
        }
    }
}

pub fn parse(input: &str) -> Result<Noun, ()> {
    parse_tokens(&mut Tokenizer::new(input.chars()))
}

/// Parses either a noun or a formula.
///
/// Formulas will be evaluated on the spot. Failure to evaluate the formula
/// will result an error.
fn parse_tokens<I: Iterator<Item = Tok>>(input: &mut I) -> Result<Noun, ()> {
    use Tok::*;
    match input.next() {
        Some(Sel) => parse_cell(input),
        Some(Op(op)) => parse_formula(op, input),
        Some(Atom(n)) => Ok(Noun::Atom(n)),
        _ => Err(()),
    }
}

/// Parses a cell, must have at least two nouns inside.
fn parse_cell<I: Iterator<Item = Tok>>(input: &mut I) -> Result<Noun, ()> {
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
fn parse_cell_tail<I: Iterator<Item = Tok>>(input: &mut I) -> Option<Result<Noun, ()>> {
    use Tok::*;
    match input.next() {
        Some(Ser) => None,
        Some(Sel) => Some(parse_cell(input)),
        Some(Op(op)) => Some(parse_formula(op, input)),
        Some(Atom(n)) => Some(Ok(Noun::Atom(n))),
        _ => Some(Err(())),
    }
}

fn parse_noun<I: Iterator<Item = Tok>>(input: &mut I) -> Result<Noun, ()> {
    use Tok::*;
    match input.next() {
        Some(Sel) => parse_cell(input),
        Some(Atom(n)) => Ok(Noun::Atom(n)),
        _ => Err(()),
    }
}

fn parse_formula<I: Iterator<Item = Tok>>(op: Op, input: &mut I) -> Result<Noun, ()> {
    let noun = try!(parse_noun(input));
    Formula(op, noun).eval().map_err(|_| ())
}

// Re-export hack for testing macros.
mod nock {
    pub use Noun;
}

#[cfg(test)]
mod test {
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
        use super::Op::*;
        use super::{a, eval, Formula};

        // Examples from
        // https://github.com/cgyarvin/urbit/blob/master/doc/book/1-nock.markdown
        assert_eq!(eval(Tis, n![0, 0]), Ok(a(0)));
        assert_eq!(eval(Tis, n![2, 2]), Ok(a(0)));
        assert_eq!(eval(Tis, n![0, 1]), Ok(a(1)));

        assert_eq!(eval(Fas, n![1, n![n![4, 5], n![6, 14, 15]]]),
                   Ok(n![n![4, 5], n![6, 14, 15]]));
        assert_eq!(eval(Fas, n![2, n![n![4, 5], n![6, 14, 15]]]), Ok(n![4, 5]));
        assert_eq!(eval(Fas, n![3, n![n![4, 5], n![6, 14, 15]]]),
                   Ok(n![6, 14, 15]));
        assert_eq!(eval(Fas, n![7, n![n![4, 5], n![6, 14, 15]]]),
                   Ok(n![14, 15]));

        assert_eq!(eval(Lus, n![1, 2]), Err(Formula(Lus, n![1, 2])));
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

    #[test]
    fn test_parse_formula() {
        use super::{parse, a};

        assert!(parse("+[1 2]").is_err());
        assert_eq!(parse("=[0 0]"), Ok(a(0)));
        assert_eq!(parse("=[0 1]"), Ok(a(1)));
        assert_eq!(parse("/[3 [[4 5] [6 14 15]]]"), Ok(n![6, 14, 15]));
    }
}
