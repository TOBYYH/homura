use crate::ops::{self, Pool2d};
use crate::{DType, Tensor};

/// The options of [`max_pool2d`] and [`max_pool2d_with_indices`], each for the height and then
/// the width, like PyTorch's arguments. [`new`](MaxPool2dOptions::new) sets the kernel size and
/// PyTorch's defaults for the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxPool2dOptions {
    /// The size of the window.
    pub kernel_size: [usize; 2],
    /// The step between windows; `None` (the default) is the kernel size.
    pub stride: Option<[usize; 2]>,
    /// The implicit `-inf` padding on both sides, `[0, 0]` by default. At most half of the
    /// kernel size, and of the kernel size widened by the dilation.
    pub padding: [usize; 2],
    /// The step between the elements of a window, `[1, 1]` by default.
    pub dilation: [usize; 2],
    /// Whether a last partial window counts (rounding the output size up), `false` by default.
    /// It must start inside the input or the left padding.
    pub ceil_mode: bool,
}

impl MaxPool2dOptions {
    /// The options for windows of `kernel_size` with PyTorch's defaults.
    pub fn new(kernel_size: [usize; 2]) -> MaxPool2dOptions {
        MaxPool2dOptions {
            kernel_size,
            stride: None,
            padding: [0, 0],
            dilation: [1, 1],
            ceil_mode: false,
        }
    }
}

/// The output size of a pooling dimension of `n` elements, PyTorch's `pooling_output_shape`,
/// with its checks.
fn pooling_output_size(n: usize, k: usize, p: usize, s: usize, d: usize, ceil_mode: bool) -> i64 {
    if s == 0 {
        panic!("stride should not be zero");
    }
    let (n, k, p, s, d) = (n as i64, k as i64, p as i64, s as i64, d as i64);
    // Rust's division truncates toward zero like C++'s, also for a kernel of 0.
    if p > ((k - 1) * d + 1) / 2 {
        panic!(
            "pad should be at most half of effective kernel size, but got pad={p}, \
             kernel_size={k} and dilation={d}"
        );
    }
    let numerator = n + 2 * p - d * (k - 1) - 1 + if ceil_mode { s - 1 } else { 0 };
    let mut size = numerator.div_euclid(s) + 1;
    // The last window must start inside the input or the left padding.
    if ceil_mode && (size - 1) * s >= n + p {
        size -= 1;
    }
    size
}

/// Checks the arguments of a max pooling of `input` like PyTorch, in its order, and those homura
/// adds (bool inputs, windows that cover only padding), and returns the geometry.
fn max_pool2d_geometry(input: &Tensor, options: &MaxPool2dOptions) -> Pool2d {
    let MaxPool2dOptions {
        kernel_size: kernel,
        stride,
        padding,
        dilation,
        ceil_mode,
    } = *options;
    let all = kernel
        .iter()
        .chain(stride.iter().flatten())
        .chain(&padding)
        .chain(&dilation);
    if all.into_iter().any(|&v| v > i32::MAX as usize) {
        panic!("value cannot be converted to type int without overflow");
    }
    let shape = input.shape();
    let nd = shape.len();
    if nd != 3 && nd != 4 {
        panic!("non-empty 3D or 4D (batch mode) tensor expected for input");
    }
    let stride = stride.unwrap_or(kernel);
    let (planes, height, width) = (shape[nd - 3], shape[nd - 2], shape[nd - 1]);
    let out_h = pooling_output_size(
        height,
        kernel[0],
        padding[0],
        stride[0],
        dilation[0],
        ceil_mode,
    );
    let out_w = pooling_output_size(
        width,
        kernel[1],
        padding[1],
        stride[1],
        dilation[1],
        ceil_mode,
    );
    if kernel.contains(&0) {
        panic!(
            "kernel size should be greater than zero, but got kH: {} kW: {}",
            kernel[0], kernel[1]
        );
    }
    if dilation.contains(&0) {
        panic!(
            "dilation should be greater than zero, but got dilationH: {} dilationW: {}",
            dilation[0], dilation[1]
        );
    }
    let nonempty = shape[nd - 3..].iter().all(|&n| n != 0) && (nd == 4 || shape[0] != 0);
    if !nonempty {
        panic!(
            "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, but \
             got:{:?}",
            &shape[..]
        );
    }
    if padding[1] > kernel[1] / 2 || padding[0] > kernel[0] / 2 {
        panic!(
            "pad should be smaller than or equal to half of kernel size, but got padW = {}, \
             padH = {}, kW = {}, kH = {}",
            padding[1], padding[0], kernel[1], kernel[0]
        );
    }
    if out_h < 1 || out_w < 1 {
        panic!(
            "Given input size: ({planes}x{height}x{width}). Calculated output size: \
             ({planes}x{out_h}x{out_w}). Output size is too small"
        );
    }
    if input.dtype() == DType::Bool {
        panic!("max_pool2d is not implemented for bool tensors");
    }
    let g = Pool2d {
        kernel,
        stride,
        padding,
        dilation,
        output: [out_h as usize, out_w as usize],
    };
    // With the pads checked above, a window that starts inside the input contains its start
    // (the last one starts before the end, also in ceil mode), and one that starts in the left
    // padding reaches the input. So only the first of them, and only for a dilation above the
    // input size, can step over the input: those are the windows checked, at most a few, so that
    // the check doesn't depend on the output size.
    for (d, (name, n)) in [("height", height), ("width", width)]
        .into_iter()
        .enumerate()
    {
        let in_padding = if dilation[d] > n {
            g.output[d].min(padding[d].div_ceil(stride[d]))
        } else {
            0
        };
        if let Some(o) = (0..in_padding).find(|&o| {
            let (first, end) = g.window(d, o, n);
            first >= end
        }) {
            panic!(
                "max_pool2d(): the window at output position {o} of the {name} covers only \
                 padding (input size {n}, kernel size {}, stride {}, padding {}, dilation {}); \
                 PyTorch returns -inf there with an index outside the input",
                kernel[d], stride[d], padding[d], dilation[d]
            );
        }
    }
    g
}

/// The maximum of each window of the last two dimensions of `input`, `[C, H, W]` or
/// `[N, C, H, W]`, PyTorch's `F.max_pool2d`; see [`max_pool2d_with_indices`].
///
/// ```
/// use homura::nn::functional::{MaxPool2dOptions, max_pool2d};
///
/// let x = homura::arange(16.0).reshape([1, 4, 4]);
/// let y = max_pool2d(&x, MaxPool2dOptions::new([2, 2]));
/// assert_eq!(y.to_vec::<f32>(), [5.0, 7.0, 13.0, 15.0]);
/// ```
///
/// # Panics
///
/// As [`max_pool2d_with_indices`].
pub fn max_pool2d(input: &Tensor, options: MaxPool2dOptions) -> Tensor {
    max_pool2d_with_indices(input, options).0
}

/// The maximum of each window of the last two dimensions of `input`, `[C, H, W]` or
/// `[N, C, H, W]` (`N` may be 0), and its position in the plane, `h * W + w` as `I64`:
/// PyTorch's `F.max_pool2d` with `return_indices=True`. Windows are `kernel_size` positions
/// `dilation` apart and start every `stride` positions of the input padded with `-inf`.
///
/// Like PyTorch's CPU kernel, a window selects its first maximum, or its last NaN, and never the
/// padding: a window whose input elements are all `-inf` selects the first of them. The values
/// are recorded as `MaxPool2DWithIndicesBackward0`, whose gradient adds the upstream gradient at
/// the selected positions, in output order and in the dtype like PyTorch, recorded in turn as
/// `MaxPool2DWithIndicesBackwardBackward0`. The indices have no history. Floating point and
/// integer inputs are accepted.
///
/// ```
/// use homura::nn::functional::{MaxPool2dOptions, max_pool2d_with_indices};
///
/// let x = homura::tensor([[[f32::NAN, 2.0], [f32::NAN, 1.0]]]);
/// let (values, indices) = max_pool2d_with_indices(&x, MaxPool2dOptions::new([2, 2]));
/// assert!(values.item::<f32>().is_nan());
/// assert_eq!(indices.item::<i64>(), 2);
/// ```
///
/// # Panics
///
/// With PyTorch's messages and in its order: if a value doesn't fit in `i32`, `input` isn't 3-
/// or 4-dimensional or has an empty dimension other than the batch, the stride, kernel size or
/// dilation is zero, the padding exceeds half of the effective or of the plain kernel size, or
/// the output would be empty. Unlike PyTorch, also for bool inputs (also on the meta device),
/// and if a window covers only padding, which can happen with a dilation above 1: PyTorch
/// returns `-inf` with an index outside the plane there.
pub fn max_pool2d_with_indices(input: &Tensor, options: MaxPool2dOptions) -> (Tensor, Tensor) {
    let g = max_pool2d_geometry(input, &options);
    ops::max_pool2d_with_indices(input, g)
}

/// The mean of each of `output_size` windows that cover the last two dimensions of `input`,
/// `[C, H, W]` or `[N, C, H, W]`, PyTorch's `F.adaptive_avg_pool2d`. Window `i` of a dimension
/// of `n` elements is `[floor(i * n / out), ceil((i + 1) * n / out))`, so windows may overlap
/// and the output may be larger than the input.
///
/// For `output_size` `[1, 1]` it is `input.mean_dim([-1, -2], true)` like in PyTorch, recorded as
/// `MeanBackward1`, for any rank of at least 3. Otherwise it is a single node,
/// `AdaptiveAvgPool2DBackward0`, computed like PyTorch's CPU kernel: each window summed in order
/// in `f32` (`f64` for `F64`), divided by its height, then by its width, and rounded once; its
/// gradient (`AdaptiveAvgPool2DBackwardBackward0`) divides and adds in the dtype like PyTorch's.
/// Inputs whose strides PyTorch takes as channels last use another kernel there, which divides
/// once, so its results can differ in the last bit. An output size of 0 gives an empty output,
/// whose gradient is zeros; PyTorch's backward pass rejects it.
///
/// ```
/// use homura::nn::functional::adaptive_avg_pool2d;
///
/// let x = homura::arange(5.0).reshape([1, 1, 5]);
/// assert_eq!(adaptive_avg_pool2d(&x, [1, 3]).to_vec::<f32>(), [0.5, 2.0, 3.5]);
/// ```
///
/// # Panics
///
/// With PyTorch's messages: if `input` has fewer than 3 dimensions; for output sizes other than
/// `[1, 1]`, if it has more than 4, an empty height or width, or isn't floating point (unlike
/// PyTorch, also where no element is computed).
pub fn adaptive_avg_pool2d(input: &Tensor, output_size: [usize; 2]) -> Tensor {
    let shape = input.shape();
    let nd = shape.len();
    if nd <= 2 {
        panic!("Input dimension should be at least 3");
    }
    if output_size == [1, 1] {
        return input.mean_dim([-1, -2], true);
    }
    if nd != 3 && nd != 4 {
        panic!(
            "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got {:?}",
            &shape[..]
        );
    }
    for d in [nd - 2, nd - 1] {
        if shape[d] == 0 {
            panic!(
                "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch \
                 dimensions, but input has sizes {:?} with dimension {d} being empty",
                &shape[..]
            );
        }
    }
    let dtype = input.dtype();
    if !dtype.is_floating_point() {
        panic!("adaptive_avg_pool2d is not implemented for {dtype} tensors");
    }
    ops::adaptive_avg_pool2d(input, output_size)
}
