use num::BigUint;
use num::traits::One;
use digit_slice::{DigitSlice, FromDigits, msb};
use {Shape, Noun, NockError, NockResult};

/// Interface for a virtual machine for Nock code.
///
/// A virtual machine can process Hint operations and accelerate Call
/// operations.
pub trait Nock {
    /// Accelerate computation of a formula, if possible.
    ///
    /// Nock `*[a 9 b c]`, will trigger `call(*[a c], *[*[a c] 0 b])`.
    /// If call returns a noun, that noun will be used as the result of the
    /// Nock evaluation. Otherwise the formula will be evaluated as standard
    /// Nock.
    ///
    /// The return value must be exactly the same as you would get for
    /// evaluating the formula on the subject with a standard Nock
    /// interpreter.
    #[allow(unused_variables)]
    fn call(&mut self, subject: &Noun, formula: &Noun) -> Option<Noun> {
        None
    }

    /// Handle a Nock hint.
    ///
    /// Nock `*[a 10 b c]` will trigger `hint(a, b, c)`.
    #[allow(unused_variables)]
    fn hint(&mut self,
            subject: &Noun,
            hint: &Noun,
            c: &Noun)
            -> Result<(), NockError> {
        Ok(())
    }

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
                            _ => return Err(NockError(format!("fire"))),
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
                            _ => Err(NockError(format!("bump"))),
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
                            _ => return Err(NockError(format!("same"))),
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
                                        return Err(NockError(format!("if")));
                                    }
                                }
                                _ => return Err(NockError(format!("if"))),
                            }
                            continue;
                        } else {
                            return Err(NockError(format!("if")));
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
                            _ => return Err(NockError(format!("compose"))),
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
                            _ => return Err(NockError(format!("push"))),
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
                            _ => return Err(NockError(format!("call"))),
                        }
                    }

                    // Hint
                    Some(10) => {
                        match tail.get() {
                            Shape::Cell(ref b, ref c) => {
                                try!(self.hint(&subject, b, c));
                                formula = (*c).clone();
                                continue;
                            }
                            _ => return Err(NockError(format!("hint"))),
                        }
                    }

                    // Unhandled opcode
                    Some(code) => {
                        return Err(NockError(format!("unknown opcode {}",
                                                     code)));
                    }

                    None => {
                        if let Shape::Cell(_, _) = ops.get() {
                            // Autocons
                            let a = try!(self.nock_on(subject.clone(),
                                                      ops.clone()));
                            let b = try!(self.nock_on(subject, tail.clone()));
                            return Ok(Noun::cell(a, b));
                        } else {
                            return Err(NockError(format!("autocons")));
                        }
                    }
                }
            } else {
                return Err(NockError(format!("nock")));
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
                return Err(NockError(format!("axis")));
            }
        }
        Ok((*subject).clone())
    }

    #[inline]
    fn bit(data: &[u8], pos: usize) -> bool {
        data[pos / 8] & (1 << (pos % 8)) != 0
    }

    match axis.get() {
        Shape::Atom(ref x) => {
            let start = msb(x);
            fas(x, start, subject)
        }
        _ => Err(NockError(format!("axis"))),
    }
}
