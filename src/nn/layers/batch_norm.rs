use std::fmt;

use super::Forward;
use crate::nn::functional::BatchNormOptions;
use crate::nn::{Module, Parameter, Training, functional, init};
use crate::{DType, Device, Tensor, TensorOptions};

/// Batch normalization of `[N, C, H, W]` inputs over the batch and the plane, PyTorch's
/// `nn.BatchNorm2d`, with [`functional::batch_norm`].
///
/// By default it has a `weight` (ones) and a `bias` (zeros) of `num_features` elements, and
/// tracks running statistics in the buffers `running_mean` (zeros), `running_var` (ones) and
/// `num_batches_tracked`, a zero-dimensional `I64` count of the training passes; all with the
/// default dtype and device, `eps = 1e-5` and `momentum = 0.1`, like PyTorch. The state dict has
/// PyTorch's keys and order.
///
/// In training mode the layer normalizes with the statistics of the batch and updates the
/// running statistics (with a momentum of `None`, to the average of all batches so far); in
/// evaluation mode ([`Module::eval`]) it normalizes with the running statistics. A layer that
/// doesn't track them uses the statistics of the batch in both modes.
///
/// ```
/// use homura::nn::{BatchNorm2d, Module};
///
/// let norm = BatchNorm2d::new(3);
/// let y = norm.forward(&homura::randn([4, 3, 5, 5]));
/// assert_eq!(y.shape(), [4, 3, 5, 5]);
/// assert_eq!(norm.num_batches_tracked.as_ref().unwrap().item::<i64>(), 1);
/// assert_eq!(
///     norm.state_dict().keys().collect::<Vec<_>>(),
///     ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]
/// );
///
/// let plain = BatchNorm2d::new(3).affine(false).track_running_stats(false);
/// assert!(plain.state_dict().is_empty());
/// ```
#[derive(Module)]
#[non_exhaustive]
pub struct BatchNorm2d {
    /// The scale, `[num_features]`, with affine.
    pub weight: Option<Parameter>,
    /// The shift, `[num_features]`, with affine and a bias.
    pub bias: Option<Parameter>,
    /// The running mean, `[num_features]`, when tracking running statistics.
    pub running_mean: Option<Tensor>,
    /// The running variance (unbiased), `[num_features]`, when tracking running statistics.
    pub running_var: Option<Tensor>,
    /// The number of training passes, a zero-dimensional `I64` tensor, when tracking running
    /// statistics.
    pub num_batches_tracked: Option<Tensor>,
    pub num_features: usize,
    pub eps: f64,
    /// The weight of a batch's statistics in the running ones; `None` for the cumulative
    /// average.
    pub momentum: Option<f64>,
    /// Whether training passes update the running statistics, and evaluation passes use them.
    pub track_running_stats: bool,
    pub training: Training,
}

impl BatchNorm2d {
    /// A layer for inputs with `num_features` channels (`nn.BatchNorm2d(num_features)`), with a
    /// weight and a bias, tracking running statistics.
    #[track_caller]
    pub fn new(num_features: usize) -> BatchNorm2d {
        let c = [num_features];
        BatchNorm2d {
            weight: Some(Parameter::new(crate::ones(c))),
            bias: Some(Parameter::new(crate::zeros(c))),
            running_mean: Some(crate::zeros(c)),
            running_var: Some(crate::ones(c)),
            num_batches_tracked: Some(crate::zeros_with([], DType::I64)),
            num_features,
            eps: 1e-5,
            momentum: Some(0.1),
            track_running_stats: true,
            training: Training::default(),
        }
    }

    /// This layer with `eps`, which is added to the variance (`eps=...`).
    #[must_use]
    #[track_caller]
    pub fn eps(mut self, eps: f64) -> BatchNorm2d {
        self.eps = eps;
        self
    }

    /// This layer with `momentum` (`momentum=...`): a number, or `None` for the cumulative
    /// average.
    #[must_use]
    #[track_caller]
    pub fn momentum(mut self, momentum: impl Into<Option<f64>>) -> BatchNorm2d {
        self.momentum = momentum.into();
        self
    }

    /// This layer with a weight and a bias if `affine` is true (the default), and with neither
    /// otherwise (`affine=False`). New parameters are initialized like
    /// [`reset_parameters`](BatchNorm2d::reset_parameters) does, with the dtype and device of the
    /// floating point members the layer has (see [`track_running_stats`](Self::track_running_stats));
    /// a parameter it has stays.
    #[must_use]
    #[track_caller]
    pub fn affine(mut self, affine: bool) -> BatchNorm2d {
        if !affine {
            self.weight = None;
            self.bias = None;
        } else {
            let (like, c) = (self.like(), self.num_features);
            self.weight
                .get_or_insert_with(|| Parameter::new(new_tensor(like, c, 1.0)));
            self.bias
                .get_or_insert_with(|| Parameter::new(new_tensor(like, c, 0.0)));
        }
        self
    }

    /// This layer without a bias if `bias` is false (`bias=False`). With true (the default), a
    /// layer with affine gets a bias of zeros if it has none, with the dtype and device of its
    /// weight; a layer without affine stays without one, as in PyTorch.
    #[must_use]
    #[track_caller]
    pub fn bias(mut self, bias: bool) -> BatchNorm2d {
        if !bias {
            self.bias = None;
        } else if let Some(w) = &self.weight {
            let (like, c) = ((w.dtype(), w.device()).into(), self.num_features);
            self.bias
                .get_or_insert_with(|| Parameter::new(new_tensor(like, c, 0.0)));
        }
        self
    }

    /// This layer tracking running statistics if `track_running_stats` is true (the default),
    /// and without them otherwise (`track_running_stats=False`), which removes the buffers. New
    /// buffers start like [`reset_running_stats`](BatchNorm2d::reset_running_stats) sets them,
    /// with the dtype and device of the first floating point member the layer has (the weight,
    /// the bias, the running mean, the running variance), or the default ones; the count is
    /// `I64` on that device. A buffer the layer has stays.
    #[must_use]
    #[track_caller]
    pub fn track_running_stats(mut self, track_running_stats: bool) -> BatchNorm2d {
        self.track_running_stats = track_running_stats;
        if !track_running_stats {
            self.running_mean = None;
            self.running_var = None;
            self.num_batches_tracked = None;
        } else {
            let like = self.like();
            let c = self.num_features;
            self.running_mean
                .get_or_insert_with(|| new_tensor(like, c, 0.0));
            self.running_var
                .get_or_insert_with(|| new_tensor(like, c, 1.0));
            self.num_batches_tracked
                .get_or_insert_with(|| crate::zeros_with([], (DType::I64, like.device())));
        }
        self
    }

    /// The dtype and device of the first floating point member, or none (the defaults).
    fn like(&self) -> TensorOptions {
        let members = [
            self.weight.as_deref(),
            self.bias.as_deref(),
            self.running_mean.as_ref(),
            self.running_var.as_ref(),
        ];
        match members.into_iter().flatten().next() {
            Some(t) => (t.dtype(), t.device()).into(),
            None => TensorOptions::default(),
        }
    }

    /// Sets the running statistics of a layer that tracks them to their initial values, like
    /// PyTorch: the mean to zeros, the variance to ones and the count to 0, in place.
    #[track_caller]
    pub fn reset_running_stats(&self) {
        if self.track_running_stats {
            if let Some(mean) = &self.running_mean {
                init::zeros_(mean);
            }
            if let Some(var) = &self.running_var {
                init::ones_(var);
            }
            if let Some(count) = &self.num_batches_tracked {
                init::zeros_(count);
            }
        }
    }

    /// Initializes the running statistics again with
    /// [`reset_running_stats`](BatchNorm2d::reset_running_stats), and the parameters, like
    /// PyTorch: the weight with ones and the bias with zeros.
    #[track_caller]
    pub fn reset_parameters(&self) {
        self.reset_running_stats();
        if let Some(w) = &self.weight {
            init::ones_(w);
        }
        if let Some(b) = &self.bias {
            init::zeros_(b);
        }
    }

    /// The normalization of `input`, `[N, C, H, W]`, like PyTorch's layer: a training pass of a
    /// layer that tracks running statistics first adds 1 to `num_batches_tracked` (also for an
    /// empty batch, and also when [`functional::batch_norm`] then rejects the input), and
    /// updates the running statistics with `momentum`, or `1 / num_batches_tracked` for `None`.
    /// The statistics of the batch normalize the input in training mode, and in evaluation mode
    /// when the layer has neither running statistic; the running statistics are passed in
    /// evaluation mode and to a training pass that tracks them.
    ///
    /// # Panics
    ///
    /// Panics with PyTorch's messages if `input` isn't 4-dimensional (before counting), if the
    /// momentum is `None` and the count is on the meta device, which has no value to read, or is
    /// 0 after counting (a count of -1 was loaded), which leaves the running statistics as they
    /// are, and as [`functional::batch_norm`].
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let rank = input.num_dims();
        if rank != 4 {
            panic!("expected 4D input (got {rank}D input)");
        }
        let training = self.training.get();
        let mut factor = self.momentum.unwrap_or(0.0);
        if training
            && self.track_running_stats
            && let Some(count) = &self.num_batches_tracked
        {
            count.add_(1);
            if self.momentum.is_none() {
                if count.device() == Device::Meta {
                    panic!("Tensor.item() cannot be called on meta tensors");
                }
                let n = count.to(DType::F64).item::<f64>();
                // A count loaded as -1 reaches 0 here; Python's `1.0 / float(count)` raises.
                if n == 0.0 {
                    panic!("division by zero");
                }
                factor = 1.0 / n;
            }
        }
        let stats_used = training || self.running_mean.is_none() && self.running_var.is_none();
        let pass = !training || self.track_running_stats;
        functional::batch_norm(
            input,
            self.running_mean.as_ref().filter(|_| pass),
            self.running_var.as_ref().filter(|_| pass),
            self.weight.as_deref(),
            self.bias.as_deref(),
            BatchNormOptions {
                training: stats_used,
                momentum: factor,
                eps: self.eps,
            },
        )
    }
}

/// A tensor of `n` elements filled with `value`, with the dtype and device of `like` (the
/// defaults where it has none).
fn new_tensor(like: TensorOptions, n: usize, value: f64) -> Tensor {
    crate::full_with([n], value, (like.float_dtype(), like.device()))
}

impl Forward for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        BatchNorm2d::forward(self, input)
    }
}

/// The options, like PyTorch's `repr`:
/// `BatchNorm2d { num_features: 3, eps: 1e-5, momentum: Some(0.1), affine: true, bias: true, track_running_stats: true, .. }`.
impl fmt::Debug for BatchNorm2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchNorm2d")
            .field("num_features", &self.num_features)
            .field("eps", &self.eps)
            .field("momentum", &self.momentum)
            .field("affine", &self.weight.is_some())
            .field("bias", &self.bias.is_some())
            .field("track_running_stats", &self.track_running_stats)
            .finish_non_exhaustive()
    }
}
