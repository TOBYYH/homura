//! `Debug` and `Display` for tensors, following PyTorch's `repr` format.

use std::fmt;

use super::Tensor;
use crate::backend::cpu;
use crate::shape::contiguous_strides;
use crate::{DType, Device};

const PRECISION: usize = 4;
const THRESHOLD: usize = 1000;
const EDGE_ITEMS: usize = 3;
const LINE_WIDTH: usize = 80;

/// The elements of a tensor in row-major order.
enum Values {
    Float(Vec<f64>),
    Int(Vec<i64>),
    Bool(Vec<bool>),
}

/// Python's `{:.4e}`: Rust writes `1.2346e0` where Python writes `1.2346e+00`.
fn sci(v: f64) -> String {
    let s = format!("{v:.PRECISION$e}");
    match s.split_once('e') {
        Some((mantissa, exp)) => {
            let e: i32 = exp.parse().unwrap();
            format!("{mantissa}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
        }
        None => s,
    }
}

fn nonfinite(v: f64) -> Option<String> {
    if v.is_nan() {
        Some("nan".into())
    } else if v.is_infinite() {
        Some(if v > 0.0 { "inf" } else { "-inf" }.into())
    } else {
        None
    }
}

/// Chooses a common number format for the printed elements, like PyTorch's `_Formatter`.
struct Formatter {
    int_mode: bool,
    sci_mode: bool,
    width: usize,
}

impl Formatter {
    fn new(values: &Values, shown: &[usize]) -> Formatter {
        let mut f = Formatter {
            int_mode: true,
            sci_mode: false,
            width: 1,
        };
        match values {
            Values::Float(v) => {
                let finite: Vec<f64> = shown
                    .iter()
                    .map(|&i| v[i])
                    .filter(|x| x.is_finite() && *x != 0.0)
                    .collect();
                if finite.is_empty() {
                    return f;
                }
                let min = finite.iter().fold(f64::INFINITY, |m, x| m.min(x.abs()));
                let max = finite.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                f.int_mode = finite.iter().all(|x| x.ceil() == *x);
                let width = |s: String| s.len();
                if f.int_mode {
                    if max / min > 1000.0 || max > 1e8 {
                        f.sci_mode = true;
                        f.width = finite.iter().map(|&x| width(sci(x))).max().unwrap();
                    } else {
                        f.width = finite
                            .iter()
                            .map(|&x| width(format!("{x:.0}")) + 1)
                            .max()
                            .unwrap();
                    }
                } else if max / min > 1000.0 || max > 1e8 || min < 1e-4 {
                    f.sci_mode = true;
                    f.width = finite.iter().map(|&x| width(sci(x))).max().unwrap();
                } else {
                    f.width = finite
                        .iter()
                        .map(|&x| width(format!("{x:.PRECISION$}")))
                        .max()
                        .unwrap();
                }
            }
            Values::Int(v) => {
                f.width = shown
                    .iter()
                    .map(|&i| v[i].to_string().len())
                    .max()
                    .unwrap_or(1);
            }
            Values::Bool(v) => {
                f.width = shown
                    .iter()
                    .map(|&i| if v[i] { 4 } else { 5 })
                    .max()
                    .unwrap_or(1);
            }
        }
        f
    }

    fn format(&self, values: &Values, i: usize) -> String {
        let s = match values {
            Values::Float(v) => {
                let x = v[i];
                if self.sci_mode {
                    nonfinite(x).unwrap_or_else(|| sci(x))
                } else if self.int_mode {
                    nonfinite(x).unwrap_or_else(|| format!("{x:.0}."))
                } else {
                    nonfinite(x).unwrap_or_else(|| format!("{x:.PRECISION$}"))
                }
            }
            Values::Int(v) => v[i].to_string(),
            Values::Bool(v) => if v[i] { "True" } else { "False" }.to_string(),
        };
        format!("{s:>width$}", width = self.width)
    }
}

/// Positions along a dimension of size `n` that are printed.
fn shown_positions(n: usize, summarize: bool) -> Vec<Option<usize>> {
    if summarize && n > 2 * EDGE_ITEMS {
        (0..EDGE_ITEMS)
            .map(Some)
            .chain([None])
            .chain((n - EDGE_ITEMS..n).map(Some))
            .collect()
    } else {
        (0..n).map(Some).collect()
    }
}

/// Flat indices of every printed element.
fn shown_indices(
    shape: &[usize],
    strides: &[usize],
    base: usize,
    summarize: bool,
    out: &mut Vec<usize>,
) {
    if shape.is_empty() {
        out.push(base);
        return;
    }
    for p in shown_positions(shape[0], summarize).into_iter().flatten() {
        shown_indices(
            &shape[1..],
            &strides[1..],
            base + p * strides[0],
            summarize,
            out,
        );
    }
}

struct Printer<'a> {
    values: &'a Values,
    formatter: Formatter,
    summarize: bool,
}

impl Printer<'_> {
    fn tensor(&self, shape: &[usize], strides: &[usize], base: usize, indent: usize) -> String {
        match shape.len() {
            0 => self.formatter.format(self.values, base),
            1 => self.vector(shape[0], strides[0], base, indent),
            nd => {
                let slices: Vec<String> = shown_positions(shape[0], self.summarize)
                    .into_iter()
                    .map(|p| match p {
                        Some(p) => self.tensor(
                            &shape[1..],
                            &strides[1..],
                            base + p * strides[0],
                            indent + 1,
                        ),
                        None => "...".to_string(),
                    })
                    .collect();
                let sep = format!(",{}{}", "\n".repeat(nd - 1), " ".repeat(indent + 1));
                format!("[{}]", slices.join(&sep))
            }
        }
    }

    fn vector(&self, n: usize, stride: usize, base: usize, indent: usize) -> String {
        let per_line = ((LINE_WIDTH.saturating_sub(indent)) / (self.formatter.width + 2)).max(1);
        let items: Vec<String> = shown_positions(n, self.summarize)
            .into_iter()
            .map(|p| match p {
                Some(p) => self.formatter.format(self.values, base + p * stride),
                None => " ...".to_string(),
            })
            .collect();
        let lines: Vec<String> = items.chunks(per_line).map(|c| c.join(", ")).collect();
        format!(
            "[{}]",
            lines.join(&format!(",\n{}", " ".repeat(indent + 1)))
        )
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.data();
        let shape = data.layout.shape.clone();
        let dtype = data.dtype;
        let default = crate::default_dtype();
        let indent = "tensor(".len();
        let mut suffixes = vec![];
        if data.device() != Device::Cpu {
            suffixes.push(format!("device={}", data.device()));
        }
        let body = if data.device() == Device::Meta || shape.num_elements() == 0 {
            if data.device() == Device::Meta || shape.len() != 1 {
                suffixes.push(format!("size={shape:?}"));
            }
            if dtype != default {
                suffixes.push(format!("dtype={dtype}"));
            }
            if data.device() == Device::Meta {
                "...".to_string()
            } else {
                "[]".to_string()
            }
        } else {
            if !matches!(dtype, DType::I64 | DType::Bool) && dtype != default {
                suffixes.push(format!("dtype={dtype}"));
            }
            let values = if dtype.is_floating_point() {
                Values::Float(cpu::read_host(&data))
            } else if dtype == DType::Bool {
                Values::Bool(cpu::read_host(&data))
            } else {
                Values::Int(cpu::read_host(&data))
            };
            let strides = contiguous_strides(&shape);
            let summarize = shape.num_elements() > THRESHOLD;
            let mut shown = vec![];
            shown_indices(&shape, &strides, 0, summarize, &mut shown);
            let formatter = Formatter::new(&values, &shown);
            Printer {
                values: &values,
                formatter,
                summarize,
            }
            .tensor(&shape, &strides, 0, indent)
        };
        suffixes.extend(crate::autograd::repr_suffix(self));
        let mut out = format!("tensor({body}");
        // PyTorch's `_add_suffixes` counts the last line as `len(s) - s.rfind("\n") + 1`: two
        // more than its length.
        let mut last_line = out.len() - out.rfind('\n').map_or(0, |i| i + 1) + 2;
        for s in suffixes {
            if last_line + s.len() + 2 > LINE_WIDTH {
                out.push_str(&format!(",\n{}{s}", " ".repeat(indent)));
                last_line = indent + s.len();
            } else {
                out.push_str(&format!(", {s}"));
                last_line += s.len() + 2;
            }
        }
        out.push(')');
        f.write_str(&out)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Device, Tensor};

    #[test]
    fn repr_matches_pytorch() {
        let s = |t: Tensor| format!("{t:?}");
        assert_eq!(
            s(crate::tensor([[1., 2.], [3., 4.]])),
            "tensor([[1., 2.],\n        [3., 4.]])"
        );
        assert_eq!(s(crate::tensor([1.5, -2.25])), "tensor([ 1.5000, -2.2500])");
        assert_eq!(
            s(crate::tensor([1e-5, 1.0])),
            "tensor([1.0000e-05, 1.0000e+00])"
        );
        assert_eq!(s(crate::tensor([f64::NAN, 1.0])), "tensor([nan, 1.])");
        assert_eq!(s(crate::tensor([1, -20, 3])), "tensor([  1, -20,   3])");
        assert_eq!(s(crate::tensor([true, false])), "tensor([ True, False])");
        assert_eq!(s(crate::tensor(2.5)), "tensor(2.5000)");
        assert_eq!(
            s(crate::tensor_with([1.0, 2.0], DType::F16)),
            "tensor([1., 2.], dtype=f16)"
        );
        assert_eq!(s(crate::zeros([0, 3])), "tensor([], size=[0, 3])");
        assert_eq!(
            s(crate::zeros_with([2, 3], Device::Meta)),
            "tensor(..., device=meta, size=[2, 3])"
        );
        assert_eq!(s(crate::arange(3).view([1, 1, 3])), "tensor([[[0, 1, 2]]])");
        let x = crate::arange(2000);
        assert_eq!(s(x), "tensor([   0,    1,    2,  ..., 1997, 1998, 1999])");
        assert_eq!(
            s(crate::tensor([123456789.0, 1.0])),
            "tensor([1.2346e+08, 1.0000e+00])"
        );
        assert_eq!(
            s(crate::tensor([0.001, 1000.5])),
            "tensor([1.0000e-03, 1.0005e+03])"
        );
        assert_eq!(s(crate::zeros([0])), "tensor([])");
        assert_eq!(
            s(crate::zeros_with([0], DType::I64)),
            "tensor([], dtype=i64)"
        );
        assert_eq!(
            s(crate::tensor_with(
                [[0.0, 0.0, 0.0], [-0.0, 0.0, 0.0]],
                DType::F64
            )),
            "tensor([[0., 0., 0.],\n        [-0., 0., 0.]], dtype=f64)"
        );
        assert_eq!(
            s(crate::arange(30.0).view([2, 15]) * 1.1),
            "tensor([[ 0.0000,  1.1000,  2.2000,  3.3000,  4.4000,  5.5000,  6.6000,  7.7000,\n          \
             8.8000,  9.9000, 11.0000, 12.1000, 13.2000, 14.3000, 15.4000],\n        [16.5000, \
             17.6000, 18.7000, 19.8000, 20.9000, 22.0000, 23.1000, 24.2000,\n         25.3000, \
             26.4000, 27.5000, 28.6000, 29.7000, 30.8000, 31.9000]])"
        );
    }

    /// The strings are PyTorch's (tools/golden/check_m1_retain.py), except where noted.
    #[test]
    fn autograd_suffixes_match_pytorch() {
        let s = |t: &Tensor| format!("{t:?}");
        let x = crate::tensor([1.0, 2.0]);
        x.requires_grad_(true);
        assert_eq!(s(&x), "tensor([1., 2.], requires_grad=True)");
        assert_eq!(s(&(&x * 2.0)), "tensor([2., 4.], grad_fn=<MulBackward0>)");
        assert_eq!(s(&x.sum()), "tensor(3., grad_fn=<SumBackward0>)");
        let b = crate::zeros([2]);
        b.copy_(&x);
        assert_eq!(s(&b), "tensor([1., 2.], grad_fn=<CopyBackwards>)");
        let v = {
            let _no_grad = crate::no_grad();
            x.view([2])
        };
        assert_eq!(s(&v), "tensor([1., 2.], requires_grad=True)");
        assert_eq!(s(&x.detach()), "tensor([1., 2.])");
        let empty = crate::zeros([0]);
        empty.requires_grad_(true);
        assert_eq!(s(&empty), "tensor([], requires_grad=True)");
        // homura's own spelling of sizes, dtypes and devices; the suffix rules are PyTorch's.
        let empty = crate::zeros([0, 3]);
        empty.requires_grad_(true);
        assert_eq!(
            s(&(&empty * 1.0)),
            "tensor([], size=[0, 3], grad_fn=<MulBackward0>)"
        );
        let m = crate::zeros_with([2, 3], (DType::F16, Device::Meta));
        m.requires_grad_(true);
        assert_eq!(
            s(&(&m * 1.0)),
            "tensor(..., device=meta, size=[2, 3], dtype=f16, grad_fn=<MulBackward0>)"
        );
        let long = crate::arange(2000.0);
        long.requires_grad_(true);
        assert_eq!(
            s(&(&long * 1.0)),
            "tensor([0.0000e+00, 1.0000e+00, 2.0000e+00,  ..., 1.9970e+03, 1.9980e+03,\n        \
             1.9990e+03], grad_fn=<MulBackward0>)"
        );
        // Wrapping counts the last line two longer than it is, on one line and on several.
        let row = |n: usize| {
            let x = crate::arange(n as f64);
            x.requires_grad_(true);
            s(&(&x * 1.5))
        };
        assert_eq!(
            row(5),
            "tensor([0.0000, 1.5000, 3.0000, 4.5000, 6.0000], grad_fn=<MulBackward0>)"
        );
        assert_eq!(
            row(6),
            "tensor([0.0000, 1.5000, 3.0000, 4.5000, 6.0000, 7.5000],\n       \
             grad_fn=<MulBackward0>)"
        );
        let grid = |cols: usize| {
            let x = crate::arange((2 * cols) as f64);
            x.requires_grad_(true);
            s(&(&x.view([2, cols as i64]) * 1.25))
        };
        assert_eq!(
            grid(5),
            "tensor([[ 0.0000,  1.2500,  2.5000,  3.7500,  5.0000],\n        \
             [ 6.2500,  7.5000,  8.7500, 10.0000, 11.2500]], grad_fn=<MulBackward0>)"
        );
        assert_eq!(
            grid(6),
            "tensor([[ 0.0000,  1.2500,  2.5000,  3.7500,  5.0000,  6.2500],\n        \
             [ 7.5000,  8.7500, 10.0000, 11.2500, 12.5000, 13.7500]],\n       \
             grad_fn=<MulBackward0>)"
        );
        // A stale view: PyTorch replays it and shows `grad_fn=<AsStridedBackward0>`; homura shows
        // `Invalid`, PyTorch's text for a `grad_fn` that can't be computed, and doesn't panic.
        let b = &x * 1.0;
        let stale = b.view([2]);
        b.add_(1.0);
        assert_eq!(s(&stale), "tensor([2., 3.], grad_fn=<Invalid>)");
        assert_eq!(s(&b), "tensor([2., 3.], grad_fn=<AddBackward0>)");
    }
}
