//! Default dtype and device for newly created tensors.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::{DType, Device};

static DEFAULT_DTYPE: AtomicU8 = AtomicU8::new(DType::F32 as u8);
static DEFAULT_DEVICE: RwLock<Device> = RwLock::new(Device::Cpu);

thread_local! {
    static DEVICE_SCOPE: Cell<Option<Device>> = const { Cell::new(None) };
}

/// The dtype used for floating point tensors when none is given (`F32` unless changed).
#[track_caller]
pub fn default_dtype() -> DType {
    let v = DEFAULT_DTYPE.load(Ordering::Relaxed);
    DType::ALL.into_iter().find(|&d| d as u8 == v).unwrap()
}

/// Sets the default floating point dtype for the whole process.
///
/// # Panics
///
/// Panics if `dtype` is not a floating point type.
#[track_caller]
pub fn set_default_dtype(dtype: DType) {
    assert!(
        dtype.is_floating_point(),
        "only floating-point types are supported as the default type, got {dtype}"
    );
    DEFAULT_DTYPE.store(dtype as u8, Ordering::Relaxed);
}

/// The device used for new tensors when none is given: the innermost active [`device_scope`] on
/// this thread, otherwise the process-wide default (`Device::Cpu` unless changed).
#[track_caller]
pub fn default_device() -> Device {
    DEVICE_SCOPE
        .with(Cell::get)
        .unwrap_or_else(|| *DEFAULT_DEVICE.read().unwrap_or_else(|e| e.into_inner()))
}

/// Sets the process-wide default device.
#[track_caller]
pub fn set_default_device(device: Device) {
    *DEFAULT_DEVICE.write().unwrap_or_else(|e| e.into_inner()) = device;
}

/// Makes `device` the default device on this thread until the returned guard is dropped.
///
/// ```
/// let _guard = homura::device_scope(homura::Device::Meta);
/// assert_eq!(homura::zeros([2, 3]).device(), homura::Device::Meta);
/// ```
///
/// The guard must be bound to a named variable: `let _ = homura::device_scope(..);` drops it at
/// once. [`with_device`] avoids the mistake.
#[track_caller]
pub fn device_scope(device: Device) -> DeviceScope {
    let previous = DEVICE_SCOPE.with(|s| s.replace(Some(device)));
    DeviceScope {
        previous,
        _not_send: PhantomData,
    }
}

/// Runs `f` with `device` as the default device on this thread, like `with torch.device(device):`,
/// and returns its value. The previous default device is restored when `f` returns or panics.
///
/// ```
/// use homura::Device;
///
/// let x = homura::with_device(Device::Meta, || homura::zeros([2, 3]));
/// assert_eq!(x.device(), Device::Meta);
/// assert_eq!(homura::zeros([2]).device(), Device::Cpu);
/// ```
#[track_caller]
pub fn with_device<R>(device: Device, f: impl FnOnce() -> R) -> R {
    let _guard = device_scope(device);
    f()
}

/// Guard returned by [`device_scope`]; restores the previous default device when dropped.
#[must_use = "the scope ends when the guard is dropped"]
#[derive(Debug)]
pub struct DeviceScope {
    previous: Option<Device>,
    _not_send: PhantomData<*const ()>,
}

impl Drop for DeviceScope {
    fn drop(&mut self) {
        DEVICE_SCOPE.with(|s| s.set(self.previous));
    }
}
