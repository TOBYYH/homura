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
    /// Waits until all queued work on this device has finished. A no-op on the CPU and meta
    /// devices, where work runs synchronously.
    #[track_caller]
    pub fn synchronize(self) {}

    /// A human readable name, such as `"cpu"`.
    #[track_caller]
    pub fn name(self) -> String {
        self.to_string()
    }
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
