use num::BigUint;
use num::traits::{Zero, One, FromPrimitive};
use digit_slice::{DigitSlice, FromDigits};
use {Shape, Noun, NockError, NockResult, FromNoun};

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
                            /*
                            let core = try!(get_axis(&Noun::from(2u32), &subject));
                            println!("{} call {}", ::symhash(&core), ::symhash(&formula));
                            */

                            continue;
                        }
                        _ => return Err(NockError),
                    }
                }

                // Hint
                Some(10) => {
                    match tail.get() {
                        Shape::Cell(ref hint, ref c) => {
                            let (id, clue) = match hint.get() {
                                Shape::Cell(ref p, ref q) => {
                                    (p.clone(), try!(nock_on(subject.clone(), (*q).clone())))
                                }
                                Shape::Atom(_) => {
                                    (hint.clone(), Noun::from(0u32))
                                }
                            };

                            // TODO: Handle other hint types than %fast.
                            if String::from_noun(id).unwrap() == "fast" {
                                let core = try!(nock_on(subject.clone(), (*c).clone()));
                                if let Ok((_name, _axis, _hooks)) = parse_fast_clue(&clue) {
                                    /*
                                    if let Shape::Cell(ref bat, _) = core.get() {
                                        println!("{} register {} {} {:?}", ::symhash(bat), name, axis, hooks);
                                    }
                                    */
                                } else {
                                    //println!("Error parsing clue {:?}", &clue);
                                }

                                // TODO: Register core with info

                                return Ok(core);
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

fn get_axis(axis: &Noun, subject: &Noun) -> NockResult {
    // TODO: Optimize for small atoms.
    fn fas(x: BigUint, n: &Noun) -> NockResult {
        let two = BigUint::from_u32(2).unwrap();
        let three = BigUint::from_u32(3).unwrap();
        if x == BigUint::zero() {
            return Err(NockError);
        }
        if x == BigUint::one() {
            return Ok(n.clone());
        }
        if let Shape::Cell(ref a, ref b) = n.get() {
            if x == two {
                return Ok((*a).clone());
            } else if x == three {
                return Ok((*b).clone());
            } else {
                let half = x.clone() >> 1;
                let p = try!(fas(half, n));

                if x % two.clone() == BigUint::zero() {
                    fas(two, &p)
                } else {
                    fas(three, &p)
                }
            }
        } else {
            return Err(NockError);
        }
    }

    match axis.get() {
        Shape::Atom(ref x) => fas(BigUint::from_digits(x).unwrap(), subject),
        _ => Err(NockError)
    }
}

fn parse_fast_clue(clue: &Noun) -> Result<(String, u32, Vec<(String, Noun)>), NockError> {
    if let Some((ref name, ref axis_formula, ref hooks)) = clue.get_122() {
        let chum = try!(String::from_noun(name).map_err(|_| NockError));

        let axis = if let Shape::Cell(ref a, ref b) = axis_formula.get() {
            if let (Some(1), Some(0)) = (a.as_u32(), b.as_u32()) {
                0
            } else if let (Some(0), Some(axis)) = (a.as_u32(), b.as_u32()) {
                axis
            } else {
                return Err(NockError);
            }
        } else {
            return Err(NockError);
        };

        let hooks: Vec<(String, Noun)> = try!(FromNoun::from_noun(hooks).map_err(|_| NockError));

        Ok((chum, axis, hooks))
    } else {
        Err(NockError)
    }
}
