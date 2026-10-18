//! [`Training`]: the train / eval mode of a module.

use std::sync::atomic::{AtomicBool, Ordering};

/// Whether a module is in training mode, PyTorch's `self.training`: a flag that
/// [`Module::train`](super::Module::train) and [`Module::eval`](super::Module::eval) set on every
/// `Training` in the module tree. It starts in training mode.
///
/// A module keeps one as a field and reports it with
/// [`Visitor::training`](super::Visitor::training); `train` and `eval` reach the flags of its
/// submodules too. Changing it needs only `&self`.
///
/// ```
/// use homura::nn::Training;
///
/// let training = Training::default();
/// assert!(training.get());
/// training.set(false);
/// assert!(!training.get());
/// ```
#[derive(Debug)]
pub struct Training(AtomicBool);

impl Training {
    /// Whether the module is in training mode.
    pub fn get(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Sets the mode: training if `training` is true, evaluation otherwise.
    pub fn set(&self, training: bool) {
        self.0.store(training, Ordering::Relaxed);
    }
}

/// Training mode, as modules start.
impl Default for Training {
    fn default() -> Self {
        Training(AtomicBool::new(true))
    }
}

/// A new flag with the same mode.
impl Clone for Training {
    fn clone(&self) -> Self {
        Training(AtomicBool::new(self.get()))
    }
}
