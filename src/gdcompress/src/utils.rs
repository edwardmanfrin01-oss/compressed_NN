pub(crate) const fn bits_needed_nonzero(value: usize) -> usize {
    if value <= 1 {
        1
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

pub(crate) const fn min_position_bits(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

pub(crate) const fn signed_half_wrapped(value: u8) -> u8 {
    ((value as i8) >> 1) as u8
}

pub(crate) const fn zigzag_encode_i16(value: i16) -> u16 {
    ((value << 1) ^ (value >> 15)) as u16
}

pub(crate) const fn zigzag_decode_i16(value: u16) -> i16 {
    ((value >> 1) as i16) ^ (-((value & 1) as i16))
}
