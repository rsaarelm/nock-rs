use std::default::Default;
use num::bigint::BigUint;
use num::traits::{FromPrimitive, Zero};

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

pub trait ToNoun {
    fn to_noun<N, B>(self, &mut B) -> N
        where N: Noun<Builder = B>,
              B: Builder<Noun = N>;
}

// By convention, default-constructible Builders are instantiated at will so
// that we can use standalone construction methods.
impl<N, B, T> From<T> for N
    where T: ToNoun,
          B: Builder<Noun = N> + Default,
          N: Noun<Builder = B>
{
    fn from(val: T) -> N {
        let mut builder = Default::default();
        val.to_noun(&mut builder)
    }
}
