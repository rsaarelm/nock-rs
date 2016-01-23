use std::default::Default;
use std::rc::Rc;
use digit_slice::{DigitSlice};

pub enum Shape<'a, N: 'a + Noun> {
    // A natural number represented as a little-endian sequence of 8-bit
    // digits.
    Atom(&'a [u8]),
    // An ordered pair of nouns.
    Cell(&'a N, &'a N),
}

/// A Nock noun, the basic unit of representation.
///
/// A noun is an atom or a cell. An atom is any natural number. A cell is any
/// ordered pair of nouns.
pub trait Noun: Sized {
    fn get<'a>(&'a self) -> Shape<'a, Self>;
}

pub trait Context: Sized {
    type Noun: Noun;

    /// Build a new atom noun from a little-endian 8-bit digit sequence.
    fn atom(&mut self, digits: &[u8]) -> Self::Noun;

    /// Build a new cell noun from two existing nouns.
    fn cell(&mut self, a: Self::Noun, b: Self::Noun) -> Self::Noun;
}

/// Trait for types that can convert themselves to a noun.
pub trait ToNoun {
    fn to_noun<C: Context>(self, &mut C) -> C::Noun;
}

impl<T> ToNoun for T where T: DigitSlice {
    fn to_noun<C: Context>(self, ctx: &mut C) -> C::Noun {
        ctx.atom(self.as_digits())
    }
}

/// Trait for constructing nouns without a Builder instance for Builder types
/// that don't need a common instance for constructing nouns.
///
/// By convention if a Builder implements Default, it's assumed that Builder
/// state doesn't matter and new Builders can be constructed at will.
pub trait Build<T> : Context {
    fn noun(T) -> Self::Noun;
}

impl<C, T> Build<T> for C
    where T: ToNoun,
          C: Context + Default
{
    fn noun(val: T) -> C::Noun {
        let mut ctx: C = Default::default();
        val.to_noun(&mut ctx)
    }
}

#[derive(Clone, PartialEq, Eq)]
enum HeapNounInner {
    Atom(Vec<u8>),
    Cell(Rc<HeapNoun>, Rc<HeapNoun>),
}

#[derive(Clone, PartialEq, Eq)]
pub struct HeapNoun(HeapNounInner);

impl Noun for HeapNoun {
    fn get<'a>(&'a self) -> Shape<'a, Self> {
        match self.0 {
            HeapNounInner::Atom(ref v) => Shape::Atom(&v),
            HeapNounInner::Cell(ref a, ref b) => Shape::Cell(&*a, &*b),
        }
    }
}

#[derive(Default)]
pub struct Heap;

impl Context for Heap {
    type Noun = HeapNoun;

    /// Build a new atom noun.
    fn atom(&mut self, digits: &[u8]) -> Self::Noun {
        HeapNoun(HeapNounInner::Atom(digits.to_vec()))
    }

    /// Build a new cell noun from two existing nouns.
    fn cell(&mut self, a: Self::Noun, b: Self::Noun) -> Self::Noun {
        HeapNoun(HeapNounInner::Cell(Rc::new(a), Rc::new(b)))
    }
}

#[cfg(test)]
mod tests {
    use super::{Heap, Build};

    #[test]
    fn scratch() {
        let x = Heap::noun(123u32);
        assert!(x == Heap::noun(123u8));
    }
}
