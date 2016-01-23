use std::rc::Rc;
use digit_slice::DigitSlice;

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
pub trait Noun: Sized+Clone {
    fn get<'a>(&'a self) -> Shape<'a, Self>;

    /// Memory address or other unique identifier for the noun.
    fn addr(&self) -> usize;

    /// Run a memoizing fold over the noun
    fn fold<'a, F, G, T: Clone>(&'a self, leaf: &mut F, branch: &mut G) -> T
        where F: FnMut(&[u8]) -> T,
              G: FnMut(&T, &T) -> T
    {
        /*
        use std::collections::HashMap;
        fn g<'a, F, T: Clone>(noun: &'a Self, memo: &mut HashMap<usize, T>, leaf: &mut F, branch: &mut G) -> T
            where F: FnMut(&[u8]) -> T,
                  G: FnMut(&T, &T) -> T
        {
            let key = noun.addr();
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
        */
        match self.get() {
            Shape::Atom(x) => leaf(x),
            Shape::Cell(ref a, ref b) => {
                let a = a.fold(leaf, branch);
                let b = b.fold(leaf, branch);
                branch(&a, &b)
            }
        }
    }
}

pub trait Context: Sized {
    type Noun: Noun;

    /// Build a new atom noun from a little-endian 8-bit digit sequence.
    fn atom(&mut self, digits: &[u8]) -> Self::Noun;

    /// Build a new cell noun from two existing nouns.
    fn cell(&mut self, a: Self::Noun, b: Self::Noun) -> Self::Noun;

    fn noun<T: ToNoun>(&mut self, item: T) -> Self::Noun {
        item.to_noun(self)
    }
}

/// Trait for types that can convert themselves to a noun.
pub trait ToNoun {
    fn to_noun<C: Context>(self, &mut C) -> C::Noun;
}

impl<T> ToNoun for T where T: DigitSlice
{
    fn to_noun<C: Context>(self, ctx: &mut C) -> C::Noun {
        ctx.atom(self.as_digits())
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

    fn addr(&self) -> usize {
        &*self as *const _ as usize
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
    use super::{Context, Heap};

    #[test]
    fn scratch() {
        let x = Heap.noun(123u32);
        assert!(x == Heap.noun(123u8));
    }
}
