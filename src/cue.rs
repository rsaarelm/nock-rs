use std::collections::HashMap;
use bit_vec::BitVec;
use num::bigint::BigUint;
use num::traits::{One, ToPrimitive};
use Noun;

/// Unpack the data of an Urbit pillfile into a Nock noun.
pub fn unpack_pill(mut buf: Vec<u8>) -> Result<Noun, &'static str> {
    // Guarantee that the buffer is made of 32-bit chunks.
    while buf.len() % 4 != 0 {
        buf.push(0);
    }

    // Reverse bits of each byte to be little-endian all the way.
    for i in 0..buf.len() {
        let b = &mut buf[i];
        *b = (*b & 0xF0) >> 4 | (*b & 0x0F) << 4;
        *b = (*b & 0xCC) >> 2 | (*b & 0x33) << 2;
        *b = (*b & 0xAA) >> 1 | (*b & 0x55) << 1;
    }

    let bits = BitVec::from_bytes(&buf);

    cue(&bits)
}

/// Decode a lenght-encoded atom from a bit stream.
fn rub(bits: &BitVec, pos: usize) -> (usize, BigUint) {
    // Length of the prefix in bits is the count of initial zeroes before
    // the separator 1.

    let mut p = 0;

    // Assume the first bit is zero even if it isn't.
    let mut k = 1;
    p += 1;

    while !bits[pos + p] {
        k += 1;
        p += 1;
    }
    p += 1;

    // Read the prefix.
    let mut b = 0;
    if k > 1 {
        for i in 0..(k - 2) {
            if bits[pos + p] {
                b += 1 << i;
            }
            p += 1;
        }
        // Add the implicit top 1 to the prefix.
        b += 1 << (k - 2);
    }

    let mut q: BigUint = Default::default();
    for i in 0..b {
        if bits[pos + p] {
            q = q + (BigUint::one() << i);
        }
        p += 1;
    }
    (p, q)
}

/// Decode an encoded cell from a bit stream.
///
/// Return the Nock noun.
fn cue(bits: &BitVec) -> Result<Noun, &'static str> {
    let (_, noun) = try!(parse(0, bits, &mut Default::default()));
    return Ok(noun);

    fn parse(mut pos: usize,
             bits: &BitVec,
             dict: &mut HashMap<usize, Noun>)
             -> Result<(usize, Noun), &'static str> {
        let key = pos;
        if bits[pos] {
            pos += 1;
            if !bits[pos] {
                // 10: encode a pair.
                pos += 1;
                let (p, left) = try!(parse(pos, bits, dict));
                pos = p;
                let (p, right) = try!(parse(pos, bits, dict));
                pos = p;

                let ret = Noun::cell(left, right);
                dict.insert(key, ret.clone());
                Ok((pos, ret))
            } else {
                // 11: Repeat element
                // Read the index in bitstream where the value was first
                // encountered.
                let (p, q) = rub(&bits, pos);
                pos += p;
                let key = q.to_usize().unwrap();
                if let Some(x) = dict.get(&key) {
                    Ok((pos, x.clone()))
                } else {
                    Err("Bad cell index")
                }
            }
        } else {
            // Atom.
            let (p, q) = rub(&bits, pos);
            pos += p;
            let ret = Noun::from(q);
            dict.insert(key, ret.clone());
            Ok((pos, ret))
        }
    }
}

#[cfg(test)]
mod test {
    use Noun;
    use super::unpack_pill;

    #[test]
    fn test_read_pill() {
        let noun: Noun = "[18.446.744.073.709.551.616 8 [1 0] 8 [1 6 [5 [0 7] \
                          4 0 6] [0 6] 9 2 [0 2] [4 0 6] 0 7] 9 2 0 1]"
                             .parse()
                             .expect("Nock parse error");
        let pill: &[u8] = &[0x01, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x06, 0xc1, 0x62, 0x83, 0x60, 0x71,
                            0xd8, 0x85, 0x5b, 0xe2, 0x87, 0x99, 0xd8, 0x0d,
                            0xc1, 0x1a, 0x24, 0x43, 0x96, 0xc8, 0x86, 0x10,
                            0x1d, 0xc2, 0x32, 0x48, 0x86, 0x4c, 0x06];
        let unpacked = unpack_pill(pill.to_vec()).expect("Pill unpack failed");
        assert_eq!(noun, unpacked);
    }
}
