//! Weights as text for prompts.
use crate::network::Tensor;

/// Decimal rendering: 1-D -> one line; 2-D -> one line per row; higher -> flattened leading axes.
/// `-0.000` is written as `0.000`.
fn fmt_dec(v: f32, decimals: usize) -> String {
    let s = format!("{:.*}", decimals, v);
    if s.starts_with('-') && s[1..].chars().all(|c| c == '0' || c == '.') {
        s[1..].to_string()
    } else {
        s
    }
}

fn row_len(t: &Tensor) -> usize {
    t.shape.last().copied().unwrap_or(1).max(1)
}

pub fn tensor_text(t: &Tensor, decimals: usize) -> String {
    let mut out = String::new();
    for row in t.data.chunks(row_len(t)) {
        out += &row.iter().map(|v| fmt_dec(*v, decimals)).collect::<Vec<_>>().join(" ");
        out.push('\n');
    }
    out
}

/// Integers times one scale. The scale is a power of ten chosen so that the largest |entry|
/// keeps `sig` significant digits. Returns (scale exponent, text); rows are prefixed "i: ".
pub fn int_scale(t: &Tensor, sig: u32) -> (i32, String) {
    let m = t.data.iter().fold(0f32, |a, v| a.max(v.abs()));
    let e = if m > 0.0 {
        (m as f64).log10().floor() as i32 - (sig as i32 - 1)
    } else {
        0
    };
    let scale = 10f64.powi(e);
    let ints = |row: &[f32]| {
        row.iter()
            .map(|v| ((*v as f64 / scale).round() as i64).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    let mut out = String::new();
    if t.shape.len() >= 2 {
        for (i, row) in t.data.chunks(row_len(t)).enumerate() {
            out += &format!("{i}: {}\n", ints(row));
        }
    } else {
        out += &ints(&t.data);
        out.push('\n');
    }
    (e, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimals_rows() {
        let t = Tensor::new(vec![2, 3], vec![0.0544, -0.0004, 1.0, 2.0, -3.0, 0.5]);
        assert_eq!(tensor_text(&t, 3), "0.054 0.000 1.000\n2.000 -3.000 0.500\n");
        let b = Tensor::new(vec![3], vec![1.0, -2.0, 0.25]);
        assert_eq!(tensor_text(&b, 2), "1.00 -2.00 0.25\n");
    }

    #[test]
    fn int_scale_three_sig_digits() {
        let t = Tensor::new(vec![2, 2], vec![0.0123, -0.4567, 0.001, 0.0]);
        let (e, s) = int_scale(&t, 3);
        assert_eq!(e, -3);
        assert_eq!(s, "0: 12 -457\n1: 1 0\n");
        let z = Tensor::new(vec![2], vec![0.0, 0.0]);
        assert_eq!(int_scale(&z, 3).1, "0 0\n");
    }
}
