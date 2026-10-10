//! Error type for fallible operations.
//!
//! Programming errors such as shape, dtype or device mismatches panic, like PyTorch raising an
//! exception that is rarely caught. Operations that can fail for reasons outside the program's
//! control (IO, parsing, file formats) return [`Result`].

use std::fmt;

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
}

/// `Result` with [`Error`] as the error type.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Parse(msg) => write!(f, "parse error: {msg}"),
            Error::Format(msg) => write!(f, "format error: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
