use num::BigUint;
use num::traits::One;
use digit_slice::{DigitSlice, FromDigits};
use {Shape, Noun, NockError, NockResult};

/// Interface for a virtual machine for Nock code.
///
/// A virtual machine can process Hint operations and accelerate Call
/// operations.
pub trait Nock {
    /// Hook for accelerating the computation of a known formula.
    ///
    /// Nock `*[a 9 b c]`, will trigger `call(*[a c], *[a 0 b])`.
    /// If call returns a noun, that noun will be used as the result of the
    /// nock evaluation. Otherwise the formula will be evaluated as standard
    /// nock.
    #[allow(unused_variables)]
    fn call(&mut self, subject: &Noun, formula: &Noun) -> Option<Noun> {
        None
    }

    /// Allows the VM to handle a hint from a Nock 10 call.
    ///
    /// Nock `*[a 10 b c]` will trigger `hint(a, b)`.
    #[allow(unused_variables)]
    fn hint(&mut self, subject: &Noun, hint: &Noun) {}

    /// Evaluate the nock `*[subject formula]`
    fn nock_on(&mut self, mut subject: Noun, mut formula: Noun) -> NockResult {
        loop {
            if let Shape::Cell(ops, tail) = formula.clone().get() {
                match ops.as_u32() {
                    // Axis
                    Some(0) => return get_axis(&tail, &subject),

                    // Just
                    Some(1) => return Ok(tail.clone()),

                    // Fire
                    Some(2) => {
                        match tail.get() {
                            Shape::Cell(ref b, ref c) => {
                                let p = try!(self.nock_on(subject.clone(),
                                                          (*b).clone()));
                                let q = try!(self.nock_on(subject,
                                                          (*c).clone()));
                                subject = p;
                                formula = q;
                                continue;
                            }
                            _ => return Err(NockError),
                        }
                    }

                    // Depth
                    Some(3) => {
                        let p = try!(self.nock_on(subject.clone(),
                                                  (*tail).clone()));
                        return match p.get() {
                            Shape::Cell(_, _) => Ok(Noun::from(0u32)),
                            _ => Ok(Noun::from(1u32)),
                        };
                    }

                    // Bump
                    Some(4) => {
                        let p = try!(self.nock_on(subject.clone(),
                                                  (*tail).clone()));
                        return match p.get() {
                            Shape::Atom(ref x) => {
                                // TODO: Non-bignum optimization
                                Ok(Noun::from(BigUint::from_digits(x).unwrap() +
                                              BigUint::one()))
                            }
                            _ => Err(NockError),
                        };
                    }

                    // Same
                    Some(5) => {
                        let p = try!(self.nock_on(subject.clone(),
                                                  (*tail).clone()));
                        return match p.get() {
                            Shape::Cell(ref a, ref b) => {
                                if a == b {
                                    // Yes.
                                    return Ok(Noun::from(0u32));
                                } else {
                                    // No.
                                    return Ok(Noun::from(1u32));
                                }
                            }
                            _ => return Err(NockError),
                        };
                    }

                    // If
                    Some(6) => {
                        if let Some((b, c, d)) = tail.get_122() {
                            let p = try!(self.nock_on(subject.clone(),
                                                      (*b).clone()));
                            match p.get() {
                                Shape::Atom(ref x) => {
                                    if x == &0u32.as_digits() {
                                        formula = (*c).clone();
                                    } else if x == &1u32.as_digits() {
                                        formula = (*d).clone();
                                    } else {
                                        return Err(NockError);
                                    }
                                }
                                _ => return Err(NockError),
                            }
                            continue;
                        } else {
                            return Err(NockError);
                        }
                    }

                    // Compose
                    Some(7) => {
                        match tail.get() {
                            Shape::Cell(ref b, ref c) => {
                                let p = try!(self.nock_on(subject.clone(),
                                                          (*b).clone()));
                                subject = p;
                                formula = (*c).clone();
                                continue;
                            }
                            _ => return Err(NockError),
                        }
                    }

                    // Push
                    Some(8) => {
                        match tail.get() {
                            Shape::Cell(ref b, ref c) => {
                                let p = try!(self.nock_on(subject.clone(),
                                                          (*b).clone()));
                                subject = Noun::cell(p, subject);
                                formula = (*c).clone();
                                continue;
                            }
                            _ => return Err(NockError),
                        }
                    }

                    // Call
                    Some(9) => {
                        match tail.get() {
                            Shape::Cell(ref axis, ref c) => {
                                // Construct core.
                                subject = try!(self.nock_on(subject.clone(),
                                                            (*c).clone()));
                                // Fetch from core using axis.
                                formula = try!(get_axis(axis, &subject));

                                if let Some(result) = self.call(&subject,
                                                                &formula) {
                                    return Ok(result);
                                }

                                continue;
                            }
                            _ => return Err(NockError),
                        }
                    }

                    // Hint
                    Some(10) => {
                        match tail.get() {
                            Shape::Cell(ref b, ref c) => {
                                self.hint(&subject, b);
                                formula = (*c).clone();
                                continue;
                            }
                            _ => return Err(NockError),
                        }
                    }

                    // Unhandled opcode
                    Some(_) => {
                        return Err(NockError);
                    }

                    None => {
                        if let Shape::Cell(_, _) = ops.get() {
                            // Autocons
                            let a = try!(self.nock_on(subject.clone(),
                                                      ops.clone()));
                            let b = try!(self.nock_on(subject, tail.clone()));
                            return Ok(Noun::cell(a, b));
                        } else {
                            return Err(NockError);
                        }
                    }
                }
            } else {
                return Err(NockError);
            }
        }
    }
}

/// Evaluate nock `/[axis subject]`
pub fn get_axis(axis: &Noun, subject: &Noun) -> NockResult {
    fn fas(x: &[u8], n: usize, mut subject: &Noun) -> NockResult {
        for i in (0..(n - 1)).rev() {
            if let Shape::Cell(ref a, ref b) = subject.get() {
                if bit(x, i) {
                    subject = b;
                } else {
                    subject = a;
                }
            } else {
                return Err(NockError);
            }
        }
        Ok((*subject).clone())
    }

    match axis.get() {
        Shape::Atom(ref x) => {
            let start = msb(x);
            fas(x, start, subject)
        }
        _ => Err(NockError),
    }
}

#[inline]
fn bit(data: &[u8], pos: usize) -> bool {
    data[pos / 8] & (1 << (pos % 8)) != 0
}

/// Return the bit position of the most significant bit.
///
/// Interprets the data as a little-endian integer.
/// Assumes that the data has no trailing zeroes.
#[inline]
pub fn msb(data: &[u8]) -> usize {
    if data.len() == 0 {
        return 0;
    }

    let mut last = data[data.len() - 1];
    assert!(last != 0);
    let mut ret = (data.len() - 1) * 8;
    while last != 0 {
        ret += 1;
        last >>= 1;
    }

    ret
}

#[cfg(test)]
mod tests {
    use super::msb;

    #[test]
    fn test_msb() {
        assert_eq!(0, msb(&vec![]));
        assert_eq!(1, msb(&vec![1]));
        assert_eq!(4, msb(&vec![15]));
        assert_eq!(5, msb(&vec![16]));
        assert_eq!(13, msb(&vec![123, 16]));
    }
}
