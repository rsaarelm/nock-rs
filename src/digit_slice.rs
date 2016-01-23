use std::slice;
use std::mem;
use num::bigint::{BigUint, BigDigit};

/// Return a little-endian byte slice corresponding to an in-memory unsigned
/// integer value.
///
/// Will fail horribly if your hardware is not little-endian.
pub trait DigitSlice {
    fn as_digits<'a>(&'a self) -> &'a [u8];
}

impl DigitSlice for BigUint {
    fn as_digits<'a>(&'a self) -> &'a [u8] {
        let n_bytes = (self.bits() + 7) / 8;

        unsafe {
            let ptr = mem::transmute::<&BigUint, &Vec<BigDigit>>(self)
                          .as_ptr() as *const u8;
            slice::from_raw_parts(ptr, n_bytes)
        }
    }
}


macro_rules! primitive_impl {
    ($t:ty) => {
        impl DigitSlice for $t {
            fn as_digits<'a>(&'a self) -> &'a[u8] {
                let n_bytes = (mem::size_of::<$t>() * 8 - self.leading_zeros() as usize + 7) / 8;

                unsafe {
                    let ptr: *const u8 = mem::transmute(self);
                    slice::from_raw_parts(ptr, n_bytes)
                }
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
    use super::DigitSlice;

    #[test]
    fn test_slice() {
        assert_eq!(0u8.as_digits(), &[]);
        assert_eq!(1u8.as_digits(), &[1]);

        assert_eq!(BigUint::from_u32(12345).unwrap().as_digits(),
                   12345u32.as_digits());

        assert_eq!(256u32.as_digits(), &[0, 1]);

        assert_eq!(BigUint::parse_bytes(b"112233445566778899", 16)
                       .unwrap()
                       .as_digits(),
                   &[0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]);
    }
}
