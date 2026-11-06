//! Devices.

use std::fmt;
use std::str::FromStr;

use crate::{Error, Result};

/// Where a tensor's data lives.
///
/// `Device` is a lightweight identifier: it holds no runtime resources. Parse a device from a
/// string with `"rocm:0".parse::<Device>()`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Device {
    /// Host memory. The default device.
    #[default]
    Cpu,
    /// An AMD GPU, by ROCm device index.
    Rocm(u32),
    /// No data at all, only shapes and dtypes. Used to build large models without allocating
    /// memory before loading their weights.
    Meta,
}

impl Device {
    /// The number of ROCm devices: 0 when there are none, or when homura was built without the
    /// `rocm` feature.
    ///
    /// # Panics
    ///
    /// Panics if the HIP runtime fails to count them (other than finding no device or driver).
    #[track_caller]
    pub fn rocm_count() -> usize {
        #[cfg(feature = "rocm")]
        return crate::backend::rocm::device_count();
        #[cfg(not(feature = "rocm"))]
        0
    }

    /// Waits until all queued work on this device has finished. A no-op on the CPU and meta
    /// devices, where work runs synchronously.
    ///
    /// # Panics
    ///
    /// Panics for a ROCm device that doesn't exist or whose context can't be created, and if
    /// homura was built without the `rocm` feature.
    #[track_caller]
    pub fn synchronize(self) {
        if let Device::Rocm(i) = self {
            #[cfg(feature = "rocm")]
            crate::backend::rocm::synchronize(i);
            #[cfg(not(feature = "rocm"))]
            without_rocm(i);
        }
    }

    /// A human readable name, such as `"cpu"`.
    #[track_caller]
    pub fn name(self) -> String {
        self.to_string()
    }

    /// The properties of a GPU, or `None` for the CPU and meta devices.
    ///
    /// ```no_run
    /// # use homura::Device;
    /// let p = Device::Rocm(0).properties().unwrap();
    /// println!("{} ({}, {} compute units)", p.name, p.arch, p.compute_units);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for a ROCm device that doesn't exist, and if homura was built without the `rocm`
    /// feature.
    #[track_caller]
    pub fn properties(self) -> Option<DeviceProperties> {
        let Device::Rocm(i) = self else { return None };
        #[cfg(feature = "rocm")]
        return Some(crate::backend::rocm::properties(i));
        #[cfg(not(feature = "rocm"))]
        without_rocm(i)
    }

    /// Free and total memory of a GPU as the driver reports them (`hipMemGetInfo`), or `None`
    /// for the CPU and meta devices. Other processes' allocations count as used.
    ///
    /// # Panics
    ///
    /// Panics for a ROCm device that doesn't exist, and if homura was built without the `rocm`
    /// feature.
    #[track_caller]
    pub fn mem_get_info(self) -> Option<MemInfo> {
        let Device::Rocm(i) = self else { return None };
        #[cfg(feature = "rocm")]
        return Some(crate::backend::rocm::mem_get_info(i));
        #[cfg(not(feature = "rocm"))]
        without_rocm(i)
    }

    /// The memory statistics of this device: bytes allocated to tensors (and homura's temporary
    /// buffers), bytes homura holds from the device, and the peak of the allocated bytes since
    /// the start or the last [`reset_peak_memory_stats`](Device::reset_peak_memory_stats).
    ///
    /// On a ROCm device, allocations are rounded up (to 512 bytes, or 2 MiB above 1 MiB) and
    /// freed memory stays reserved in homura's cache for reuse until
    /// [`empty_cache`](Device::empty_cache). On the CPU, every buffer homura allocates counts,
    /// unrounded, and is returned when freed (reserved equals allocated); data mapped from a file
    /// doesn't count. The meta device has no memory: all zero.
    ///
    /// ```
    /// # use homura::Device;
    /// let x = homura::zeros([1024, 1024]);
    /// let stats = Device::Cpu.memory_stats();
    /// println!("{} MiB allocated, peak {} MiB", stats.allocated >> 20, stats.peak >> 20);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for a ROCm device that doesn't exist, and if homura was built without the `rocm`
    /// feature.
    #[track_caller]
    pub fn memory_stats(self) -> MemoryStats {
        match self {
            Device::Cpu => crate::storage::cpu_memory_counters().stats(),
            Device::Meta => MemoryStats::default(),
            #[cfg(feature = "rocm")]
            Device::Rocm(i) => crate::backend::rocm::memory_stats(i),
            #[cfg(not(feature = "rocm"))]
            Device::Rocm(i) => without_rocm(i),
        }
    }

    /// Resets the peak of [`memory_stats`](Device::memory_stats) to the bytes allocated now.
    ///
    /// # Panics
    ///
    /// Like [`memory_stats`](Device::memory_stats).
    #[track_caller]
    pub fn reset_peak_memory_stats(self) {
        match self {
            Device::Cpu => crate::storage::cpu_memory_counters().reset_peak(),
            Device::Meta => {}
            #[cfg(feature = "rocm")]
            Device::Rocm(i) => crate::backend::rocm::reset_peak_memory_stats(i),
            #[cfg(not(feature = "rocm"))]
            Device::Rocm(i) => without_rocm(i),
        }
    }

    /// Returns the memory that homura's cache holds on a ROCm device but no tensor uses to the
    /// device, after waiting for its queued work, so that other processes (or libraries) can use
    /// it. Memory still partly in use stays. A no-op on the CPU and meta devices, which have no
    /// cache.
    ///
    /// # Panics
    ///
    /// Like [`memory_stats`](Device::memory_stats), and if the device fails while waiting or
    /// freeing.
    #[track_caller]
    pub fn empty_cache(self) {
        if let Device::Rocm(i) = self {
            #[cfg(feature = "rocm")]
            crate::backend::rocm::empty_cache(i);
            #[cfg(not(feature = "rocm"))]
            without_rocm(i);
        }
    }
}

/// Panics for an operation on a ROCm device in a build without the `rocm` feature.
#[cfg(not(feature = "rocm"))]
#[track_caller]
pub(crate) fn without_rocm(_index: u32) -> ! {
    panic!("homura was built without ROCm support; enable the `rocm` feature")
}

/// The properties of a GPU, from [`Device::properties`].
#[derive(Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub struct DeviceProperties {
    /// The marketing name, such as `"AMD Radeon AI PRO R9700"`.
    pub name: String,
    /// The architecture, such as `"gfx1201"`.
    pub arch: String,
    /// The number of compute units, as `rocminfo` counts them. (HIP's `multiProcessorCount`
    /// counts workgroup processors on RDNA GPUs, which have two compute units each.)
    pub compute_units: u32,
    /// The number of threads in a wavefront: 32 on the GPUs homura supports.
    pub warp_size: u32,
    /// Device memory in bytes.
    pub total_memory: u64,
}

/// Memory statistics in bytes, from [`Device::memory_stats`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[non_exhaustive]
pub struct MemoryStats {
    /// Bytes allocated to tensors and homura's temporary buffers.
    pub allocated: u64,
    /// Bytes homura holds from the device: allocated, or cached for reuse.
    pub reserved: u64,
    /// The largest `allocated` since the start or the last reset.
    pub peak: u64,
}

/// Free and total device memory in bytes, from [`Device::mem_get_info`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub struct MemInfo {
    /// Bytes not allocated by any process.
    pub free: u64,
    /// All device memory.
    pub total: u64,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => f.write_str("cpu"),
            Device::Rocm(i) => write!(f, "rocm:{i}"),
            Device::Meta => f.write_str("meta"),
        }
    }
}

impl FromStr for Device {
    type Err = Error;

    /// Parses `"cpu"`, `"meta"`, `"rocm"` (device 0) or `"rocm:N"`.
    fn from_str(s: &str) -> Result<Self> {
        let err = || Error::Parse(format!("invalid device string: {s:?}"));
        match s {
            "cpu" => Ok(Device::Cpu),
            "meta" => Ok(Device::Meta),
            "rocm" => Ok(Device::Rocm(0)),
            _ => {
                let index = s.strip_prefix("rocm:").ok_or_else(err)?;
                index.parse().map(Device::Rocm).map_err(|_| err())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_display() {
        for d in [Device::Cpu, Device::Rocm(0), Device::Rocm(3), Device::Meta] {
            assert_eq!(d.to_string().parse::<Device>().unwrap(), d);
        }
        assert_eq!("rocm".parse::<Device>().unwrap(), Device::Rocm(0));
        assert!("cuda:0".parse::<Device>().is_err());
        assert!("rocm:x".parse::<Device>().is_err());
        assert!("rocm:-1".parse::<Device>().is_err());
    }
}
