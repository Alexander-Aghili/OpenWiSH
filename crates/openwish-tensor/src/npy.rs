//! Minimal .npy reader and writer. Everything is converted to f32 in C order.
use anyhow::{bail, Context, Result};

use crate::network::Tensor;

const MAGIC: &[u8] = b"\x93NUMPY";

fn header_field<'a>(h: &'a str, key: &str) -> Option<&'a str> {
    let k = format!("'{key}'");
    let i = h.find(&k)? + k.len();
    h[i..].trim_start().strip_prefix(':').map(str::trim_start)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut e: i32 = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

/// What an .npy header says: the dtype code (byte order removed), the layout, the shape, and where the data starts.
struct Header {
    code: String,
    fortran: bool,
    shape: Vec<usize>,
    data_start: usize,
}

fn parse_header(bytes: &[u8]) -> Result<Header> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        bail!("not an .npy file");
    }
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                bail!("truncated .npy header");
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        v => bail!("unsupported .npy version {v}"),
    };
    let header = std::str::from_utf8(bytes.get(start..start + hlen).context("truncated .npy header")?)?;
    let descr: String = header_field(header, "descr")
        .context("no descr")?
        .trim_start_matches(['\'', '"'])
        .chars()
        .take_while(|c| *c != '\'' && *c != '"')
        .collect();
    let code = match descr.chars().next() {
        Some('>') => bail!("big-endian .npy not supported ({descr})"),
        Some('<' | '|' | '=') => descr[1..].to_string(),
        _ => descr,
    };
    let fortran = header_field(header, "fortran_order")
        .context("no fortran_order")?
        .starts_with("True");
    let shape_s = header_field(header, "shape").context("no shape")?;
    let shape_s = &shape_s[shape_s.find('(').context("bad shape")? + 1..shape_s.find(')').context("bad shape")?];
    let shape: Vec<usize> = shape_s
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect::<std::result::Result<_, _>>()?;
    Ok(Header {
        code,
        fortran,
        shape,
        data_start: start + hlen,
    })
}

/// Bytes per entry of a dtype code.
fn width(code: &str) -> Result<usize> {
    Ok(match code {
        "f4" | "i4" | "u4" => 4,
        "f8" | "i8" | "u8" => 8,
        "f2" | "i2" | "u2" | "V2" => 2,
        "i1" | "u1" | "b1" => 1,
        c => bail!("unsupported dtype {c}"),
    })
}

/// One little-endian entry of dtype `code` as f32.
fn decode(code: &str, b: &[u8]) -> Result<f32> {
    Ok(match code {
        "f4" => f32::from_le_bytes(b.try_into()?),
        "f8" => f64::from_le_bytes(b.try_into()?) as f32,
        "f2" => f16_to_f32(u16::from_le_bytes(b.try_into()?)),
        // bfloat16 as written by ml_dtypes ('<V2'): the upper half of an f32
        "V2" => f32::from_bits((u16::from_le_bytes(b.try_into()?) as u32) << 16),
        "i4" => i32::from_le_bytes(b.try_into()?) as f32,
        "u4" => u32::from_le_bytes(b.try_into()?) as f32,
        "i8" => i64::from_le_bytes(b.try_into()?) as f32,
        "u8" => u64::from_le_bytes(b.try_into()?) as f32,
        "i2" => i16::from_le_bytes(b.try_into()?) as f32,
        "u2" => u16::from_le_bytes(b.try_into()?) as f32,
        "i1" => b[0] as i8 as f32,
        _ => b[0] as f32,
    })
}

/// Column-major `data` of `shape` in row-major order.
fn fortran_to_c(data: &[f32], shape: &[usize]) -> Vec<f32> {
    let mut c = vec![0f32; data.len()];
    let mut idx = vec![0usize; shape.len()];
    for (fi, v) in data.iter().enumerate() {
        let mut r = fi;
        for (i, len) in idx.iter_mut().zip(shape) {
            *i = r % len;
            r /= len;
        }
        let ci = idx.iter().zip(shape).fold(0, |ci, (i, len)| ci * len + i);
        c[ci] = *v;
    }
    c
}

pub fn read_npy(bytes: &[u8]) -> Result<Tensor> {
    let h = parse_header(bytes)?;
    let n: usize = h.shape.iter().product();
    let width = width(&h.code)?;
    let body = &bytes[h.data_start..];
    if body.len() < n * width {
        bail!("truncated .npy data: need {} bytes, have {}", n * width, body.len());
    }
    let mut data = body[..n * width]
        .chunks_exact(width)
        .map(|b| decode(&h.code, b))
        .collect::<Result<Vec<f32>>>()?;
    if h.fortran && h.shape.len() > 1 {
        data = fortran_to_c(&data, &h.shape);
    }
    Ok(Tensor { shape: h.shape, data })
}

/// The magic string, version 1.0, header length and header (padded to 64 bytes) of an .npy file.
fn npy_header(descr: &str, fortran: bool, shape: &[usize]) -> Vec<u8> {
    let shape = match shape {
        [] => "()".to_string(),
        [n] => format!("({n},)"),
        _ => format!(
            "({})",
            shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        ),
    };
    let fortran = if fortran { "True" } else { "False" };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': {fortran}, 'shape': {shape}, }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out
}

/// `t` as a little-endian float32 .npy file in C order.
pub fn write_npy(t: &Tensor) -> Vec<u8> {
    let mut out = npy_header("<f4", false, &t.shape);
    for v in &t.data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy_bytes(descr: &str, fortran: bool, shape: &[usize], data: &[u8]) -> Vec<u8> {
        let mut out = npy_header(descr, fortran, shape);
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn f32_roundtrip() {
        let t = Tensor::new(vec![2, 3], vec![1.0, -2.0, 3.5, 0.0, 1e-3, 7.0]);
        let back = read_npy(&write_npy(&t)).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn f64_and_ints_convert() {
        let data: Vec<u8> = [1.5f64, -2.25].iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = read_npy(&npy_bytes("<f8", false, &[2], &data)).unwrap();
        assert_eq!(t.data, vec![1.5, -2.25]);
        let data: Vec<u8> = [3i64, -4].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            read_npy(&npy_bytes("<i8", false, &[2], &data)).unwrap().data,
            vec![3.0, -4.0]
        );
    }

    #[test]
    fn f16_converts() {
        // 1.0 = 0x3C00, -2.0 = 0xC000
        let data = [0x00u8, 0x3C, 0x00, 0xC0];
        assert_eq!(
            read_npy(&npy_bytes("<f2", false, &[2], &data)).unwrap().data,
            vec![1.0, -2.0]
        );
    }

    #[test]
    fn fortran_order_is_transposed_to_c() {
        // logical [[1,2,3],[4,5,6]] stored column-major: 1 4 2 5 3 6
        let data: Vec<u8> = [1f32, 4., 2., 5., 3., 6.]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = read_npy(&npy_bytes("<f4", true, &[2, 3], &data)).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data, vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn scalar_shape() {
        let data = 2.5f32.to_le_bytes();
        let t = read_npy(&npy_bytes("<f4", false, &[], &data)).unwrap();
        assert_eq!(t.shape, Vec::<usize>::new());
        assert_eq!(t.data, vec![2.5]);
    }

    #[test]
    fn rejects_garbage() {
        assert!(read_npy(b"not an npy").is_err());
    }
}
