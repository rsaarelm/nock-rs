use std::collections::HashMap;
use std::rc::Rc;
use std::hash;
use digit_slice::DigitSlice;

/// A wrapper for referencing Noun-like patterns.
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
pub struct Noun(Inner);

#[derive(Clone, PartialEq, Eq)]
enum Inner {
    Atom(Rc<Vec<u8>>),
    Cell(Rc<Noun>, Rc<Noun>),
}

impl Noun {
    fn get<'a>(&'a self) -> Shape<&'a [u8], &'a Noun> {
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
    fn fold<'a, F, T>(&'a self, mut f: F) -> T
        where F: FnMut(Shape<&'a [u8], T>) -> T,
              T: Clone
    {
        fn h<'a, F, T>(noun: &'a Noun,
                       memo: &mut HashMap<usize, T>,
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

        h(self, &mut HashMap::new(), &mut f)
    }

    /// Build a new atom noun from a little-endian 8-bit digit sequence.
    pub fn atom(digits: &[u8]) -> Noun {
        Noun(Inner::Atom(Rc::new(digits.to_vec())))
    }

    /// Build a new cell noun from two existing nouns.
    pub fn cell(&mut self, a: &Noun, b: &Noun) -> Noun {
        Noun(Inner::Cell(Rc::new(a.clone()), Rc::new(b.clone())))
    }

    pub fn from<T: ToNoun>(item: T) -> Noun {
        item.to_noun()
    }
}

impl hash::Hash for Noun {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        fn f<H: hash::Hasher>(state: &mut H, shape: Shape<&[u8], u64>) -> u64 {
            match shape {
                Shape::Atom(x) => x.hash(state),
                Shape::Cell(a, b) => {
                    a.hash(state);
                    b.hash(state);
                }
            }
            state.finish()
        }
        self.fold(|x| f(state, x))
            .hash(state);
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
        let x = Noun::from(123u32);
        assert!(x == Noun::from(123u8));
    }
}
