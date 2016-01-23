use std::collections::HashMap;
use std::rc::Rc;
use std::hash;
use digit_slice::DigitSlice;

pub enum Shape<'a> {
    /// A natural number represented as a little-endian sequence of 8-bit
    /// digits.
    Atom(&'a [u8]),
    /// An ordered pair of nouns.
    Cell(&'a Noun, &'a Noun),
}

/// A Nock noun, the basic unit of representation.
///
/// A noun is an atom or a cell. An atom is any natural number. A cell is any
/// ordered pair of nouns.
#[derive(Clone, PartialEq, Eq)]
pub struct Noun(Inner);

#[derive(Clone, PartialEq, Eq)]
enum Inner {
    Atom(Rc<Vec<u8>>),
    Cell(Rc<Noun>, Rc<Noun>),
}

impl Noun {
    fn get<'a>(&'a self) -> Shape<'a> {
        match self.0 {
            Inner::Atom(ref v) => Shape::Atom(&v),
            Inner::Cell(ref a, ref b) => Shape::Cell(&*a, &*b),
        }
    }

    /// Memory address or other unique identifier for the noun.
    fn addr(&self) -> usize {
        &*self as *const _ as usize
    }

    /// Run a memoizing fold over the noun
    fn fold<'a, F, G, T: Clone>(&'a self, mut leaf: F, mut branch: G) -> T
        where F: FnMut(&[u8]) -> T,
              G: FnMut(&T, &T) -> T
    {
        fn h<'a, F, G, T>(noun: &'a Noun,
                          memo: &mut HashMap<usize, T>,
                          leaf: &mut F,
                          branch: &mut G)
                          -> T
            where F: FnMut(&[u8]) -> T,
                  G: FnMut(&T, &T) -> T,
                  T: Clone
        {
            let key = noun.addr();

            if memo.contains_key(&key) {
                memo.get(&key).unwrap().clone()
            } else {
                let ret = match noun.get() {
                    Shape::Atom(x) => leaf(x),
                    Shape::Cell(ref a, ref b) => {
                        let a = h(*a, memo, leaf, branch);
                        let b = h(*b, memo, leaf, branch);
                        branch(&a, &b)
                    }
                };
                memo.insert(key, ret.clone());
                ret
            }
        }

        h(self, &mut HashMap::new(), &mut leaf, &mut branch)
    }

    /// Build a new atom noun from a little-endian 8-bit digit sequence.
    fn atom(digits: &[u8]) -> Noun {
        Noun(Inner::Atom(Rc::new(digits.to_vec())))
    }

    /// Build a new cell noun from two existing nouns.
    fn cell(&mut self, a: &Noun, b: &Noun) -> Noun {
        Noun(Inner::Cell(Rc::new(a.clone()), Rc::new(b.clone())))
    }

    fn noun<T: ToNoun>(item: T) -> Noun {
        item.to_noun()
    }
}

/// Trait for types that can convert themselves to a noun.
pub trait ToNoun {
    fn to_noun(&self) -> Noun;
}

impl<T> ToNoun for T where T: DigitSlice
{
    fn to_noun(&self) -> Noun {
        Noun::atom(self.as_digits())
    }
}

#[cfg(test)]
mod tests {
    use super::Noun;

    #[test]
    fn scratch() {
        let x = Noun::noun(123u32);
        assert!(x == Noun::noun(123u8));
    }
}
