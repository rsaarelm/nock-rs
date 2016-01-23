use std::slice;
use std::mem;
use num::bigint::{BigUint, BigDigit};

pub trait DigitSlice {
    /// Return a little-endian byte slice corresponding to an in-memory unsigned
    /// integer value.
    ///
    /// Will fail horribly if your hardware is not little-endian.
    fn as_digits<'a>(&'a self) -> &'a [u8];
}

pub trait FromDigits: Sized {
    /// Construct an unsigned integer type from little-endian byte digits.
    fn from_digits(&[u8]) -> Option<Self>;
}

impl DigitSlice for BigUint {
    #[inline]
    fn as_digits<'a>(&'a self) -> &'a [u8] {
        let n_bytes = (self.bits() + 7) / 8;

        unsafe {
            let ptr = mem::transmute::<&BigUint, &Vec<BigDigit>>(self)
                          .as_ptr() as *const u8;
            slice::from_raw_parts(ptr, n_bytes)
        }
    }
}

impl FromDigits for BigUint {
    fn from_digits(digits: &[u8]) -> Option<BigUint> {
        let mut v = digits.to_vec();
        while (v.len() % 4) != 0 {
            v.push(0);
        }

        unsafe {
            let ptr = v.as_mut_ptr() as *mut BigDigit;
            let len = v.len() / 4;
            // I hope possibly truncating capacity down by 1 or 3 bytes won't
            // be a problem.
            let cap = v.capacity() / 4;
            // We'll reuse the bits, don't collect the old one.
            mem::forget(v);

            let biguint_vec: Vec<BigDigit> = Vec::from_raw_parts(ptr, len, cap);
            Some(mem::transmute(biguint_vec))
        }
    }
}


macro_rules! primitive_impl {
    ($t:ty) => {
        impl DigitSlice for $t {
            #[inline]
            fn as_digits<'a>(&'a self) -> &'a[u8] {
                let n_bytes = (mem::size_of::<$t>() * 8 - self.leading_zeros() as usize + 7) / 8;

                unsafe {
                    let ptr: *const u8 = mem::transmute(self);
                    slice::from_raw_parts(ptr, n_bytes)
                }
            }
        }

        impl FromDigits for $t {
            #[inline]
            fn from_digits(digits: &[u8]) -> Option<$t> {
                let bytes = mem::size_of::<$t>();
                if digits.len() > bytes { return None; }
                let mut ret = 0 as $t;
                for i in 0..bytes {
                    ret |= digits[i] as $t << (i * 8);
                }
                Some(ret)
            }
        }
    }
}

primitive_impl!(u8);
primitive_impl!(u16);
primitive_impl!(u32);
primitive_impl!(u64);
primitive_impl!(usize);

#[cfg(test)]
mod tests {
    use num::{BigUint, FromPrimitive};
    use super::{DigitSlice, FromDigits};

    #[test]
    fn test_slice() {
        assert_eq!(0u8.as_digits(), &[]);
        assert_eq!(1u8.as_digits(), &[1]);

        assert_eq!(BigUint::from_u32(12345).unwrap().as_digits(),
                   12345u32.as_digits());

        assert_eq!(255u32.as_digits(), &[255]);
        assert_eq!(256u32.as_digits(), &[0, 1]);

        assert_eq!(BigUint::parse_bytes(b"112233445566778899", 16)
                       .unwrap()
                       .as_digits(),
                   &[0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]);

        assert_eq!(u8::from_digits(&[0x44]), Some(0x44));
        assert_eq!(u8::from_digits(&[0x44, 0x33]), None);
        assert_eq!(u32::from_digits(&[0x44, 0x33]), Some(0x3344));
    }
}
