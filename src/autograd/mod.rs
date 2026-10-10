//! Automatic differentiation.
//!
//! Only the grad mode switch exists so far; recording the backward graph comes later. Tensors
//! already carry a slot for their autograd metadata so that adding it does not change their
//! layout.

use std::cell::Cell;
use std::marker::PhantomData;

/// Per-tensor autograd state. Always absent for now.
#[derive(Debug)]
pub(crate) struct AutogradMeta {}

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Whether operations on this thread record the backward graph.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

/// Enables or disables graph recording on this thread until the returned guard is dropped.
pub fn set_grad_enabled(enabled: bool) -> GradModeGuard {
    let previous = GRAD_ENABLED.with(|g| g.replace(enabled));
    GradModeGuard {
        previous,
        _not_send: PhantomData,
    }
}

/// Disables graph recording on this thread until the returned guard is dropped, like
/// `with torch.no_grad():`.
///
/// ```
/// let _guard = homura::no_grad();
/// assert!(!homura::is_grad_enabled());
/// ```
pub fn no_grad() -> GradModeGuard {
    set_grad_enabled(false)
}

/// Enables graph recording on this thread until the returned guard is dropped, like
/// `with torch.enable_grad():`.
pub fn enable_grad() -> GradModeGuard {
    set_grad_enabled(true)
}

/// Guard returned by [`no_grad`], [`enable_grad`] and [`set_grad_enabled`]; restores the
/// previous grad mode when dropped.
#[must_use = "grad mode is restored when the guard is dropped"]
#[derive(Debug)]
pub struct GradModeGuard {
    previous: bool,
    _not_send: PhantomData<*const ()>,
}

impl Drop for GradModeGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|g| g.set(self.previous));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guards_nest() {
        assert!(is_grad_enabled());
        {
            let _a = no_grad();
            assert!(!is_grad_enabled());
            {
                let _b = enable_grad();
                assert!(is_grad_enabled());
            }
            assert!(!is_grad_enabled());
        }
        assert!(is_grad_enabled());
    }
}
