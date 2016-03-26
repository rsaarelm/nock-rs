use num::BigUint;
use num::traits::One;
use digit_slice::{DigitSlice, FromDigits};
use {Shape, Noun, NockError, NockResult};

/// Evaluate the nock `*[subject formula]`
pub fn nock_on(mut subject: Noun, mut formula: Noun) -> NockResult {
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
                            let p = try!(nock_on(subject.clone(),
                                                 (*b).clone()));
                            let q = try!(nock_on(subject, (*c).clone()));
                            subject = p;
                            formula = q;
                            continue;
                        }
                        _ => return Err(NockError),
                    }
                }

                // Depth
                Some(3) => {
                    let p = try!(nock_on(subject.clone(), (*tail).clone()));
                    return match p.get() {
                        Shape::Cell(_, _) => Ok(Noun::from(0u32)),
                        _ => Ok(Noun::from(1u32)),
                    };
                }

                // Bump
                Some(4) => {
                    let p = try!(nock_on(subject.clone(), (*tail).clone()));
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
                    let p = try!(nock_on(subject.clone(), (*tail).clone()));
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
                        let p = try!(nock_on(subject.clone(), (*b).clone()));
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
                            let p = try!(nock_on(subject.clone(),
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
                            let p = try!(nock_on(subject.clone(),
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
                            subject = try!(nock_on(subject.clone(),
                                                   (*c).clone()));
                            // Fetch from core using axis.
                            formula = try!(get_axis(axis, &subject));

                            continue;
                        }
                        _ => return Err(NockError),
                    }
                }

                // Hint
                Some(10) => {
                    match tail.get() {
                        Shape::Cell(ref b, ref c) => {
                            if let Shape::Cell(_, ref q) = b.get() {
                                // As per spec, try to nock a cell-shaped
                                // hint just to see whether it'll crash.
                                let _ = try!(nock_on(subject.clone(), (*q).clone()));
                            }

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
                        let a = try!(nock_on(subject.clone(), ops.clone()));
                        let b = try!(nock_on(subject, tail.clone()));
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
        _ => Err(NockError)
    }
}

#[inline]
fn bit(data: &[u8], pos: usize) -> bool {
    data[pos / 8] & (1 << (pos % 8)) != 0
}

/// Return the bit position of the most significant bit, interpreting the data
/// as a little-endian integer.
fn msb(data: &[u8]) -> usize {
    let mut offset = 0;
    let mut ret = 0;
    for byte in data.iter() {
        let mut b = *byte;
        if b == 0 { continue; }
        let mut x = 0;
        while b != 0 {
            x += 1;
            b >>= 1;
        }
        ret = offset * 8 + x;
        offset += 1;
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::msb;

    #[test]
    fn test_msb() {
        assert_eq!(0, msb(&vec![]));
        assert_eq!(0, msb(&vec![0]));
        assert_eq!(1, msb(&vec![1]));
        assert_eq!(4, msb(&vec![15]));
        assert_eq!(5, msb(&vec![16]));
        assert_eq!(13, msb(&vec![123, 16]));
        assert_eq!(13, msb(&vec![123, 16, 0]));
    }
}
