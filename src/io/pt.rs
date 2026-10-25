//! PyTorch's `.pt` format, as `torch.save` writes it since PyTorch 1.6: a zip archive with a
//! pickle (`<archive>/data.pkl`) whose persistent ids name storage records (`<archive>/data/<key>`).

use std::io::{Read, Seek};

use super::pickle::{Global, Id, Pickle, Tensors};
use super::value::Value;
use super::zip::{self, Archive};
use crate::nn::StateDict;
use crate::{Error, Result, Tensor};

/// The versions of the format that PyTorch 2.14 reads.
const VERSIONS: std::ops::RangeInclusive<u64> = 1..=10;

fn bad(msg: &str) -> Error {
    Error::Format(format!("invalid PyTorch file: {msg}"))
}

/// Parses the `version` record as PyTorch does: a decimal number, with optional whitespace
/// around it.
fn parse_version(bytes: &[u8]) -> Result<u64> {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.trim_ascii().parse().ok())
        .ok_or_else(|| bad("the version record is not a number"))
}

/// Checks the structure of the archive in `r` and returns its pickle.
///
/// The archive's directory is the one holding `data.pkl`, and every entry must be in it.
/// `version` must be one that PyTorch reads and `byteorder`, if there is one, `little` or `big`;
/// the other records are ignored. TorchScript archives (`torch.jit.save`) are rejected.
fn read_archive<R: Read + Seek>(r: &mut R) -> Result<Vec<u8>> {
    let archive = Archive::read(r)?;
    let mut pickles = archive.entries().iter().filter_map(|e| {
        let dir = e.name.strip_suffix("/data.pkl")?;
        (!dir.is_empty() && !dir.contains('/')).then_some(dir)
    });
    let dir = pickles
        .next()
        .ok_or_else(|| bad("there is no data.pkl, so this is not a file that torch.save wrote"))?;
    if pickles.next().is_some() {
        return Err(bad("there are several data.pkl records"));
    }
    let prefix = format!("{dir}/");
    if let Some(e) = archive
        .entries()
        .iter()
        .find(|e| !e.name.starts_with(&prefix))
    {
        return Err(bad(&format!(
            "{:?} is outside the archive's directory {dir:?}",
            e.name
        )));
    }
    let record = |name: &str| archive.get(&format!("{prefix}{name}"));
    if record("constants.pkl").is_some()
        || archive
            .entries()
            .iter()
            .any(|e| e.name.starts_with(&format!("{prefix}code/")))
    {
        return Err(Error::Format(
            "unsupported PyTorch file: this is a TorchScript archive (torch.jit.save); save a \
             state dict with torch.save instead"
                .into(),
        ));
    }
    let version = record("version").ok_or_else(|| bad("there is no version record"))?;
    let version = parse_version(&zip::read_entry(r, version)?)?;
    if !VERSIONS.contains(&version) {
        return Err(Error::Format(format!(
            "unsupported PyTorch file: version {version} (versions {} to {} are supported)",
            VERSIONS.start(),
            VERSIONS.end()
        )));
    }
    if let Some(byteorder) = record("byteorder") {
        let byteorder = zip::read_entry(r, byteorder)?;
        if byteorder != b"little" && byteorder != b"big" {
            return Err(bad("the byteorder record is neither little nor big"));
        }
    }
    zip::read_entry(r, record("data.pkl").unwrap())
}

/// Tensors can't be read from `.pt` files yet.
struct NoTensors;

impl Tensors for NoTensors {
    fn tensor(&mut self, _: &Pickle, _: Global, _: Id) -> Result<Tensor> {
        Err(Error::Format(
            "PyTorch .pt files with tensors can't be loaded yet".into(),
        ))
    }
}

/// Reads the object that the `.pt` file in `r` holds, with tensors from `tensors`.
fn read_value<R: Read + Seek>(r: &mut R, tensors: &mut dyn Tensors) -> Result<Value> {
    let pickle = Pickle::parse(&read_archive(r)?)?;
    pickle.to_value(tensors)
}

/// Reads the state dict that the `.pt` file in `r` holds: a dict or `OrderedDict` of tensors.
pub(super) fn read_state_dict<R: Read + Seek>(r: &mut R) -> Result<StateDict> {
    StateDict::try_from(&read_value(r, &mut NoTensors)?)
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{self, pt};
    use super::super::pickle::tests::{Placeholders, canonical};
    use super::super::zip::testing::{Options, Writer};
    use super::*;
    use std::io::Cursor;

    fn value(bytes: &[u8]) -> Result<Value> {
        read_value(&mut Cursor::new(bytes), &mut Placeholders::default())
    }

    fn err(result: Result<impl std::fmt::Debug>, message: &str) {
        match result {
            Err(Error::Format(m)) if m.contains(message) => {}
            other => panic!("expected a format error with {message:?}, got {other:?}"),
        }
    }

    #[test]
    fn values_match_the_reference() {
        if !fixtures::available() {
            return;
        }
        for name in [
            "values_p2",
            "values_p4",
            "values_p5",
            "repacked",
            "zip64",
            "crc0",
            "state_dict",
            "checkpoint",
            "build_once",
            "build_state_rebound",
            "build_state_extended",
            "build_empty_state",
            "build_replaced",
            "build_metadata_changed",
        ] {
            let expected = String::from_utf8(pt(&format!("{name}.expected"))).unwrap();
            let v = value(&pt(&format!("{name}.pt"))).unwrap();
            assert_eq!(canonical(&v), expected.trim_end(), "{name}");
        }
    }

    #[test]
    fn state_dicts_keep_their_versions() {
        if !fixtures::available() {
            return;
        }
        let expected: crate::nn::ModuleVersions = [("", 1), ("0", 1), ("1", 2), ("2", 1)]
            .into_iter()
            .collect();
        let sd = StateDict::try_from(&value(&pt("state_dict.pt")).unwrap()).unwrap();
        assert_eq!(sd.len(), 9);
        assert_eq!(sd.versions(), Some(&expected));
        // Inside a checkpoint too.
        let Value::Dict(checkpoint) = value(&pt("checkpoint.pt")).unwrap() else {
            panic!()
        };
        let model = StateDict::try_from(checkpoint.get("model").unwrap()).unwrap();
        assert_eq!(model.versions(), Some(&expected));
        assert!(matches!(checkpoint.get("epoch"), Some(Value::Int(3))));
        // Tensors can't be rebuilt yet; a file without them can be read.
        err(
            read_state_dict(&mut Cursor::new(pt("state_dict.pt"))),
            "can't be loaded yet",
        );
        let v = read_value(&mut Cursor::new(pt("values_p2.pt")), &mut NoTensors).unwrap();
        assert!(matches!(v, Value::Dict(_)));
    }

    #[test]
    fn expansion_and_cycles_are_errors() {
        if !fixtures::available() {
            return;
        }
        // 2**30 and 64**8 leaves from about a kilobyte of pickle: the budget stops them.
        err(
            value(&pt("deep.pt")),
            "expand to much more than the size of the file",
        );
        err(
            value(&pt("wide.pt")),
            "expand to much more than the size of the file",
        );
        err(value(&pt("cycle_list.pt")), "recursive pickle structures");
        err(value(&pt("cycle_dict.pt")), "recursive pickle structures");
    }

    /// A pickle of `{}`.
    const EMPTY: &[u8] = b"\x80\x02}q\x00.";

    /// A record of an archive: its name and data.
    type Record<'a> = (&'a str, &'a [u8]);

    fn archive(entries: &[Record]) -> Vec<u8> {
        let mut w = Writer::default();
        for (name, data) in entries {
            w.add(name, data, Options::default());
        }
        w.finish(false, b"")
    }

    #[test]
    fn archive_structure() {
        let ok = |entries: &[Record]| value(&archive(entries)).unwrap();
        ok(&[("a/data.pkl", EMPTY), ("a/version", b"3\n")]);
        // As PyTorch parses the version, from 1 to 10; byteorder is optional.
        for version in [&b"1"[..], b" 3\n", b"10\n"] {
            ok(&[("a/data.pkl", EMPTY), ("a/version", version)]);
        }
        ok(&[
            ("x/data.pkl", EMPTY),
            ("x/version", b"3"),
            ("x/byteorder", b"big"),
        ]);
        ok(&[
            ("x/data.pkl", EMPTY),
            ("x/version", b"3"),
            ("x/byteorder", b"little"),
        ]);
        let cases: [(&[Record], &str); 12] = [
            (&[("a/version", b"3")], "there is no data.pkl"),
            (
                &[("data.pkl", EMPTY), ("version", b"3")],
                "there is no data.pkl",
            ),
            (
                &[("a/b/data.pkl", EMPTY), ("a/b/version", b"3")],
                "there is no data.pkl",
            ),
            (
                &[
                    ("a/data.pkl", EMPTY),
                    ("b/data.pkl", EMPTY),
                    ("a/version", b"3"),
                ],
                "several data.pkl",
            ),
            (
                &[("a/data.pkl", EMPTY), ("a/version", b"3"), ("b/x", b"")],
                "\"b/x\" is outside the archive's directory \"a\"",
            ),
            (&[("a/data.pkl", EMPTY)], "there is no version record"),
            (&[("a/data.pkl", EMPTY), ("a/version", b"0\n")], "version 0"),
            (
                &[("a/data.pkl", EMPTY), ("a/version", b"11\n")],
                "version 11",
            ),
            (
                &[("a/data.pkl", EMPTY), ("a/version", b"x\n")],
                "not a number",
            ),
            (
                &[
                    ("a/data.pkl", EMPTY),
                    ("a/version", b"3"),
                    ("a/byteorder", b"middle"),
                ],
                "neither little nor big",
            ),
            (
                &[
                    ("a/data.pkl", EMPTY),
                    ("a/version", b"3"),
                    ("a/constants.pkl", EMPTY),
                ],
                "TorchScript",
            ),
            (
                &[
                    ("a/data.pkl", EMPTY),
                    ("a/version", b"3"),
                    ("a/code/m.py", b""),
                ],
                "TorchScript",
            ),
        ];
        for (entries, message) in cases {
            err(value(&archive(entries)), message);
        }
    }

    #[test]
    fn torchscript_archives_are_rejected() {
        if !fixtures::available() {
            return;
        }
        err(
            value(&pt("torchscript.pt")),
            "TorchScript archive (torch.jit.save)",
        );
    }

    /// A deterministic sequence of pseudo-random numbers.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as usize
        }
    }

    /// Changes a few bytes of `bytes` or truncates it.
    fn mutate(bytes: &[u8], rng: &mut Lcg) -> Vec<u8> {
        let mut out = bytes.to_vec();
        match rng.next() % 5 {
            0 => out.truncate(rng.next() % bytes.len()),
            1 => {
                let at = rng.next() % out.len();
                out[at] ^= 1 << (rng.next() % 8);
            }
            2 => {
                let at = rng.next() % out.len();
                out[at] = rng.next() as u8;
            }
            3 => {
                // A length or offset field: a small, a large or an extreme value.
                let at = rng.next() % out.len().saturating_sub(4).max(1);
                let value = [1u32, 0xff, 0xffff, 0xffff_ffff][rng.next() % 4];
                let end = (at + 4).min(out.len());
                out[at..end].copy_from_slice(&value.to_le_bytes()[..end - at]);
            }
            _ => {
                let at = rng.next() % out.len();
                out.insert(at, rng.next() as u8);
            }
        }
        out
    }

    /// Mutated archives and pickles give an error or a value, never a panic.
    #[test]
    fn mutated_input_never_panics() {
        if !fixtures::available() {
            return;
        }
        let rounds = if cfg!(debug_assertions) { 300 } else { 3000 };
        let mut rng = Lcg(3);
        for name in [
            "values_p2.pt",
            "values_p4.pt",
            "checkpoint.pt",
            "zip64.pt",
            "repacked.pt",
        ] {
            let bytes = pt(name);
            let pickle = read_archive(&mut Cursor::new(&bytes)).unwrap();
            for _ in 0..rounds {
                let _ = value(&mutate(&bytes, &mut rng));
                let mutated = mutate(&pickle, &mut rng);
                if let Ok(p) = Pickle::parse(&mutated) {
                    let _ = p.to_value(&mut Placeholders::default());
                }
            }
        }
    }
}
