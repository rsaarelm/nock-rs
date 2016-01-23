use std::fmt;
use std::default::Default;
use std::rc::Rc;
use num::bigint::BigUint;
use num::traits::{FromPrimitive, ToPrimitive, Zero};

/*
pub trait Atom {
}

pub enum Shape<'a, N: Noun> {
    Atom(&'a N::Atom),
    Cell(&'a N, &'a N),
}
*/

/// A Nock noun, the basic unit of representation.
///
/// A noun is an atom or a cell. An atom is any natural number. A cell is any
/// ordered pair of nouns.
pub trait Noun: Sized {
    type Builder: Builder;

    /// If the noun is an atom, return an arbitrary-size integer representing
    /// it.
    fn to_atom(&self) -> Option<BigUint>;

    /// If the noun is a cell, return references to the cell components.
    fn as_cell<'a>(&'a self) -> Option<(&'a Self, &'a Self)>;
}

/*
impl<N, B> fmt::Display for N where N: Noun<Builder=B>, B: Builder<Noun=N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(atom) = self.to_atom() {
            return dot_separators(f, &atom);
        }

        let (ref a, ref b) = self.as_cell().unwrap();
        try!(write!(f, "[{} ", a));
        // List pretty-printer.
        let mut cur = b;
        loop {
            if let Some((ref a, ref b)) = cur.as_cell() {
                try!(write!(f, "{} ", a));
                cur = &b;
                continue;
            }
            try!(dot_separators(f, cur.to_atom().unwrap()));
            return write!(f, "]");
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
impl<B: Builder> fmt::Debug for Noun<Builder=B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}
*/

pub trait Builder: Sized {
    type Noun: Noun<Builder=Self>;

    /// Build a new atom noun.
    fn atom<T: Into<BigUint>>(&mut self, T) -> Self::Noun;

    /// Build a new cell noun from two existing nouns.
    fn cell(&mut self, Self::Noun, Self::Noun) -> Self::Noun;

    /// Build an atom equivalent to a list of bytes.
    ///
    /// Bytes before the first non-zero byte in the byte slice will be
    /// ignored. This is because the most significant bit of the binary
    /// representation of an atom will always be 1.
    ///
    /// If the bytes are a text string, the atom will be a cord with that
    /// text.
    fn from_bytes(&mut self, bytes: &[u8]) -> Self::Noun {
        let mut x: BigUint = Zero::zero();
        for i in bytes.iter().rev() {
            x = x << 8;
            x = x + BigUint::from_u8(*i).unwrap();
        }

        self.atom(x)
    }
}

pub trait Buildable {
    fn build<N, B>(self, &mut B) -> N
        where N: Noun<Builder = B>,
              B: Builder<Noun = N>;
}

/// Trait for constructing nouns without a Builder instance for Builder types
/// that don't need a common instance for constructing nouns.
///
/// By convention if a Builder implements Default, it's assumed that Builder
/// state doesn't matter and new Builders can be constructed at will.
pub trait Build<T> : Builder {
    fn noun(T) -> Self::Noun;
}

impl<B, T> Build<T> for B
    where T: Buildable,
          B: Builder + Default
{
    fn noun(val: T) -> B::Noun {
        let mut builder = Default::default();
        val.build(&mut builder)
    }
}

impl<T: Into<BigUint>> Buildable for T {
    fn build<N, B>(self, builder: &mut B) -> N
        where N: Noun<Builder = B>,
              B: Builder<Noun = N> {
        builder.atom(self)
     }
}

#[derive(Clone, PartialEq, Eq)]
enum HeapAtom {
    Small(u32),
    Big(Rc<BigUint>),
}

#[derive(Clone, PartialEq, Eq)]
enum HeapNounInner {
    Atom(HeapAtom),
    Cell(Rc<HeapNoun>, Rc<HeapNoun>),
}

#[derive(Clone, PartialEq, Eq)]
pub struct HeapNoun(HeapNounInner);

#[derive(Default)]
pub struct HeapBuilder;

impl Noun for HeapNoun {
    type Builder = HeapBuilder;

    fn to_atom(&self) -> Option<BigUint> {
        match self.0 {
            HeapNounInner::Atom(HeapAtom::Small(ref n)) => {
                Some(BigUint::from_u32(*n).unwrap())
            }
            HeapNounInner::Atom(HeapAtom::Big(ref n)) => Some((**n).clone()),
            _ => None,
        }
    }

    fn as_cell<'a>(&'a self) -> Option<(&'a HeapNoun, &'a HeapNoun)> {
        match self.0 {
            HeapNounInner::Cell(ref a, ref b) => Some((&*a, &*b)),
            _ => None,
        }
    }
}

impl Builder for HeapBuilder {
    type Noun = HeapNoun;

    fn atom<T: Into<BigUint>>(&mut self, value: T) -> Self::Noun {
        // TODO: Small atoms.
        let atom: BigUint = value.into();
        let atom = if let Some(small) = atom.to_u32() {
            HeapAtom::Small(small)
        } else {
            HeapAtom::Big(Rc::new(atom))
        };
        HeapNoun(HeapNounInner::Atom(atom))
    }

    /// Build a new cell noun from two existing nouns.
    fn cell(&mut self, a: Self::Noun, b: Self::Noun) -> Self::Noun {
        HeapNoun(HeapNounInner::Cell(Rc::new(a), Rc::new(b)))
    }
}

#[cfg(test)]
mod tests {
    use super::{HeapBuilder, Build};

    #[test]
    fn scratch() {
        let x = HeapBuilder::noun(123u32);
        assert!(x == HeapBuilder::noun(123u8));
    }
}
