//! Error type for fallible operations.
//!
//! Programming errors such as shape, dtype or device mismatches panic, like PyTorch raising an
//! exception that is rarely caught. Operations that can fail for reasons outside the program's
//! control (IO, parsing, file formats) return [`Result`], and so do the `try_` variants of the
//! operations that allocate on a device that may run out of memory ([`Error::OutOfMemory`]).

use std::fmt;

use crate::Device;

/// Errors returned by fallible homura operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An IO error.
    Io(std::io::Error),
    /// A string could not be parsed (for example a [`Device`](crate::Device) name).
    Parse(String),
    /// A file is malformed or uses an unsupported feature.
    Format(String),
    /// A device refused memory that an operation needed, after releasing its cached memory.
    OutOfMemory(OutOfMemoryError),
}

/// A device's refusal of memory: which device, how many bytes, and a message that describes the
/// device's memory at the time (on ROCm, like PyTorch's: its capacity, what is free, what homura
/// holds).
#[derive(Clone, Debug)]
pub struct OutOfMemoryError {
    device: Device,
    requested: usize,
    message: String,
}

impl OutOfMemoryError {
    pub(crate) fn new(device: Device, requested: usize, message: String) -> Self {
        OutOfMemoryError {
            device,
            requested,
            message,
        }
    }

    /// The device that refused the memory.
    #[track_caller]
    pub fn device(&self) -> Device {
        self.device
    }

    /// The size of the refused request in bytes.
    #[track_caller]
    pub fn requested(&self) -> usize {
        self.requested
    }
}

impl fmt::Display for OutOfMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for OutOfMemoryError {}

/// `Result` with [`Error`] as the error type.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Parse(msg) => write!(f, "parse error: {msg}"),
            Error::Format(msg) => write!(f, "format error: {msg}"),
            Error::OutOfMemory(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::OutOfMemory(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
