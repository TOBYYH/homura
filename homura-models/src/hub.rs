//! Hugging Face repositories on disk: a local directory ([`Repo::local`]) or a repository in the
//! Hugging Face cache ([`Repo::cached`]), as `huggingface_hub` downloads it, and the weight files
//! that transformers would load from it ([`Repo::weight_files`]).
//!
//! homura reads the cache and never downloads: a repository missing from it is an error that says
//! to download it first (with the `hf` command line tool, for example `hf download
//! microsoft/resnet-50`, or with Python's `huggingface_hub.snapshot_download`).
//!
//! The rules are those of huggingface_hub 2.3.0 and transformers 5.20.2, offline:
//!
//! - the cache is the directory of [`cache_dir`];
//! - a repository `org/name` is `models--org--name` in it. A revision (a branch, a tag or a commit
//!   hash) names a file of its `refs` directory, which holds the commit hash (`refs/main`); a
//!   revision without one is used as the commit itself. The files of the commit are in
//!   `snapshots/<commit>`, as links to `blobs`;
//! - the weight files are, in this order, `model.safetensors`, `model.safetensors.index.json`,
//!   `pytorch_model.bin` and `pytorch_model.bin.index.json`: the first one present, without
//!   trying the others if it can't be read.
//!
//! Repository ids, revisions and file names are checked before they are used as paths, so that
//! none leaves the repository: unlike transformers, homura rejects `..` in a file name of a
//! local directory too (transformers allows the ones that stay inside) and in `transformers_weights`
//! for the cache (which transformers doesn't check offline).
//!
//! ```no_run
//! use homura_models::hub::{Repo, Revision};
//!
//! let repo = Repo::cached("microsoft/resnet-50", Revision::main())?;
//! let config = repo.path("config.json")?;
//! let weights = repo.weight_files(None, None)?;
//! println!("{} and {}", config.display(), weights.path.display());
//! # Ok::<(), homura_models::hub::HubError>(())
//! ```

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::{Path, PathBuf};

/// Why a repository or one of its files can't be used.
#[derive(Debug)]
#[non_exhaustive]
pub enum HubError {
    /// A repository id, revision or file name that isn't valid, or a ref that doesn't hold a
    /// commit.
    Invalid(String),
    /// The repository, the revision or a file isn't there.
    NotFound(String),
    /// A file of the cache can't be read.
    Io {
        /// The file.
        path: PathBuf,
        /// What reading it returned.
        error: std::io::Error,
    },
}

impl fmt::Display for HubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HubError::Invalid(message) | HubError::NotFound(message) => f.write_str(message),
            HubError::Io { path, error } => write!(f, "can't read {}: {error}", path.display()),
        }
    }
}

impl std::error::Error for HubError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HubError::Io { error, .. } => Some(error),
            HubError::Invalid(_) | HubError::NotFound(_) => None,
        }
    }
}

/// A revision of a repository in the cache: a branch, a tag or a commit hash.
///
/// A branch may have `/` (`refs/pr/1`, `feature/x`); its parts can't be empty, `.` or `..`, or
/// have `\` or `:`.
/// Revisions are checked when they are used.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Revision(String);

impl Revision {
    /// The default branch, `main`.
    #[track_caller]
    pub fn main() -> Revision {
        Revision("main".to_owned())
    }

    /// The branch, tag or commit hash `name`.
    #[track_caller]
    pub fn new(name: impl Into<String>) -> Revision {
        Revision(name.into())
    }

    /// The name of the revision.
    #[track_caller]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for Revision {
    fn default() -> Revision {
        Revision::main()
    }
}

impl fmt::Display for Revision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The format of a model's weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    /// safetensors: `model.safetensors` or the shards of `model.safetensors.index.json`.
    Safetensors,
    /// PyTorch's format: `pytorch_model.bin` or the shards of `pytorch_model.bin.index.json`.
    PyTorch,
}

impl WeightFormat {
    /// The single file and the index of the format, in the order they are looked for.
    fn names(self) -> [&'static str; 2] {
        match self {
            WeightFormat::Safetensors => ["model.safetensors", "model.safetensors.index.json"],
            WeightFormat::PyTorch => ["pytorch_model.bin", "pytorch_model.bin.index.json"],
        }
    }
}

/// The weight file of a repository that [`Repo::weight_files`] picked.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct WeightFiles {
    /// The file: the weights, or an index of shards. The shards are files of the repository's
    /// root, wherever the index is, as transformers finds them:
    /// [`homura::io::load_index(path, repo.dir())`](homura::io::load_index) reads them.
    pub path: PathBuf,
    /// The format of the weights.
    pub format: WeightFormat,
    /// Whether `path` is an index (`*.index.json`).
    pub sharded: bool,
}

/// A Hugging Face repository on disk: the directory of its files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repo {
    dir: PathBuf,
    cached: Option<Cached>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Cached {
    repo_id: String,
    revision: Revision,
    commit: String,
}

impl Repo {
    /// The repository in the directory `dir`, such as one that `save_pretrained` wrote or
    /// `hf download --local-dir` made.
    ///
    /// # Errors
    ///
    /// Returns [`HubError::NotFound`] if `dir` isn't a directory.
    #[track_caller]
    pub fn local(dir: impl AsRef<Path>) -> Result<Repo, HubError> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Err(HubError::NotFound(format!(
                "the repository {} isn't a directory",
                dir.display()
            )));
        }
        Ok(Repo {
            dir: dir.to_owned(),
            cached: None,
        })
    }

    /// The repository `repo_id` (`"org/name"`, or a name without an organization) at `revision`
    /// in the Hugging Face cache, [`cache_dir`].
    ///
    /// # Errors
    ///
    /// As for [`cached_in`](Repo::cached_in).
    #[track_caller]
    pub fn cached(repo_id: &str, revision: Revision) -> Result<Repo, HubError> {
        Repo::cached_in(cache_dir(), repo_id, revision)
    }

    /// The repository `repo_id` at `revision` in the cache directory `cache_dir`
    /// (`from_pretrained(repo_id, cache_dir=...)`).
    ///
    /// A repository id has 1 to 96 letters, digits, `_`, `-` and `.` (ASCII), optionally after
    /// an organization of the same form and a `/`; each part begins and ends with a letter, a digit
    /// or `_`, and the id has no `--` or `..` and doesn't end with `.git`, as huggingface_hub's
    /// `validate_repo_id` requires.
    ///
    /// # Errors
    ///
    /// Returns [`HubError::Invalid`] for an invalid repository id or revision, or a ref whose
    /// content can't be a commit, [`HubError::NotFound`] if the repository or the revision isn't in
    /// the cache, and [`HubError::Io`] if the ref can't be read.
    #[track_caller]
    pub fn cached_in(
        cache_dir: impl AsRef<Path>,
        repo_id: &str,
        revision: Revision,
    ) -> Result<Repo, HubError> {
        check_repo_id(repo_id)?;
        check_revision(&revision)?;
        let cache_dir = cache_dir.as_ref();
        let storage = cache_dir.join(format!("models--{}", repo_id.replace('/', "--")));
        if !storage.is_dir() {
            return Err(HubError::NotFound(format!(
                "the repository {repo_id} isn't in the Hugging Face cache {}: download it first, \
                 for example with `hf download {repo_id}` (homura only reads the cache)",
                cache_dir.display()
            )));
        }
        let mut ref_path = storage.join("refs");
        ref_path.extend(revision.as_str().split('/'));
        let commit = if ref_path.is_file() {
            let content = std::fs::read(&ref_path).map_err(|error| HubError::Io {
                path: ref_path.clone(),
                error,
            })?;
            let commit = String::from_utf8(content)
                .ok()
                .map(|s| s.trim().to_owned())
                .filter(|s| is_segment(s))
                .ok_or_else(|| {
                    HubError::Invalid(format!("{} doesn't hold a commit hash", ref_path.display()))
                })?;
            Some(commit)
        } else if is_segment(revision.as_str()) {
            Some(revision.as_str().to_owned())
        } else {
            None
        };
        let snapshot = commit.as_ref().map(|c| storage.join("snapshots").join(c));
        match (commit, snapshot) {
            (Some(commit), Some(snapshot)) if snapshot.is_dir() => Ok(Repo {
                dir: snapshot,
                cached: Some(Cached {
                    repo_id: repo_id.to_owned(),
                    revision,
                    commit,
                }),
            }),
            _ => Err(HubError::NotFound(format!(
                "the revision {revision} of {repo_id} isn't in the Hugging Face cache {}: \
                 download it first, for example with `hf download {repo_id} --revision {revision}` \
                 (homura only reads the cache)",
                cache_dir.display()
            ))),
        }
    }

    /// The directory of the repository's files: the local directory, or the snapshot of the
    /// commit in the cache.
    #[track_caller]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The repository id of a repository in the cache.
    #[track_caller]
    pub fn repo_id(&self) -> Option<&str> {
        self.cached.as_ref().map(|c| c.repo_id.as_str())
    }

    /// The revision that a repository in the cache was asked for.
    #[track_caller]
    pub fn revision(&self) -> Option<&Revision> {
        self.cached.as_ref().map(|c| &c.revision)
    }

    /// The commit of a repository in the cache: its snapshot's name, the content of the ref.
    #[track_caller]
    pub fn commit(&self) -> Option<&str> {
        self.cached.as_ref().map(|c| c.commit.as_str())
    }

    /// The path of the repository's file `filename`, a `/`-separated path relative to the
    /// repository (`"config.json"`, `"sub/model.safetensors"`), as huggingface_hub accepts it:
    /// not absolute, without `..` and without Windows drives or roots in any part.
    ///
    /// # Errors
    ///
    /// Returns [`HubError::Invalid`] for an invalid name and [`HubError::NotFound`] if the
    /// repository has no such file.
    #[track_caller]
    pub fn path(&self, filename: &str) -> Result<PathBuf, HubError> {
        check_filename(filename)?;
        let path = self.join(filename);
        if !path.is_file() {
            return Err(HubError::NotFound(format!(
                "{} has no file {filename}",
                self.describe()
            )));
        }
        Ok(path)
    }

    /// The weight file that transformers' `from_pretrained` loads: `transformers_weights` (the
    /// configuration's field) if it is given, whatever `format` is; otherwise the first of
    /// `model.safetensors`, `model.safetensors.index.json`, `pytorch_model.bin` and
    /// `pytorch_model.bin.index.json` that the repository has, or only those of `format`
    /// (`use_safetensors=True` or `False`).
    ///
    /// # Errors
    ///
    /// Returns [`HubError::Invalid`] if `transformers_weights` isn't a valid file name (see
    /// [`path`](Repo::path)) of a `.safetensors` or `.safetensors.index.json` file, and
    /// [`HubError::NotFound`] if the file isn't there.
    #[track_caller]
    pub fn weight_files(
        &self,
        format: Option<WeightFormat>,
        transformers_weights: Option<&str>,
    ) -> Result<WeightFiles, HubError> {
        if let Some(name) = transformers_weights {
            check_transformers_weights(name).map_err(HubError::Invalid)?;
            return Ok(WeightFiles {
                path: self.path(name)?,
                format: WeightFormat::Safetensors,
                sharded: name.ends_with(".index.json"),
            });
        }
        let formats = match format {
            None => &[WeightFormat::Safetensors, WeightFormat::PyTorch][..],
            Some(WeightFormat::Safetensors) => &[WeightFormat::Safetensors][..],
            Some(WeightFormat::PyTorch) => &[WeightFormat::PyTorch][..],
        };
        let mut names = Vec::new();
        for &format in formats {
            for (i, name) in format.names().into_iter().enumerate() {
                let path = self.join(name);
                if path.is_file() {
                    return Ok(WeightFiles {
                        path,
                        format,
                        sharded: i == 1,
                    });
                }
                names.push(name);
            }
        }
        Err(HubError::NotFound(format!(
            "{} has no weights: none of {}",
            self.describe(),
            names.join(", ")
        )))
    }

    fn join(&self, filename: &str) -> PathBuf {
        let mut path = self.dir.clone();
        path.extend(filename.split('/').filter(|s| !s.is_empty() && *s != "."));
        path
    }

    fn describe(&self) -> String {
        match &self.cached {
            Some(c) => format!("the repository {} (revision {})", c.repo_id, c.revision),
            None => format!("the repository {}", self.dir.display()),
        }
    }
}

/// The Hugging Face cache, as huggingface_hub finds it: `HF_HUB_CACHE`, else
/// `HUGGINGFACE_HUB_CACHE`, else `HF_HOME/hub`, where `HF_HOME` is by default
/// `$XDG_CACHE_HOME/huggingface`, and `XDG_CACHE_HOME` is by default `~/.cache`.
///
/// A variable that is set is used even if it is empty (`HF_HOME=""` gives the relative path
/// `hub`). As in huggingface_hub, `HF_HOME` and then the cache path are expanded: a leading `~`
/// or `~/` becomes `HOME` (or the user's home directory if `HOME` isn't set), and `$name` and
/// `${name}` become the variable's value, unless it isn't set. `~user` isn't expanded (Python looks
/// the user up). The rules are those of Unix paths on every platform.
#[track_caller]
pub fn cache_dir() -> PathBuf {
    let var = |name: &str| std::env::var_os(name).map(bytes);
    let home_cache = join(&expand_user(b"~".to_vec()), b".cache");
    let hf_home = var("HF_HOME")
        .unwrap_or_else(|| join(&var("XDG_CACHE_HOME").unwrap_or(home_cache), b"huggingface"));
    let hf_home = expand_vars(expand_user(hf_home));
    let legacy = var("HUGGINGFACE_HUB_CACHE").unwrap_or_else(|| join(&hf_home, b"hub"));
    let cache = expand_vars(expand_user(var("HF_HUB_CACHE").unwrap_or(legacy)));
    PathBuf::from(os_string(cache))
}

#[cfg(unix)]
fn bytes(s: OsString) -> Vec<u8> {
    use std::os::unix::ffi::OsStringExt;
    s.into_vec()
}

#[cfg(not(unix))]
fn bytes(s: OsString) -> Vec<u8> {
    s.to_string_lossy().into_owned().into_bytes()
}

#[cfg(unix)]
fn os_string(b: Vec<u8>) -> OsString {
    use std::os::unix::ffi::OsStringExt;
    OsString::from_vec(b)
}

#[cfg(not(unix))]
fn os_string(b: Vec<u8>) -> OsString {
    OsString::from(String::from_utf8_lossy(&b).into_owned())
}

/// Python's `posixpath.join(a, b)`.
fn join(a: &[u8], b: &[u8]) -> Vec<u8> {
    if b.starts_with(b"/") {
        return b.to_vec();
    }
    let mut path = a.to_vec();
    if !path.is_empty() && !path.ends_with(b"/") {
        path.push(b'/');
    }
    path.extend_from_slice(b);
    path
}

/// Python's `posixpath.expanduser`, without `~user`.
fn expand_user(path: Vec<u8>) -> Vec<u8> {
    if !path.starts_with(b"~") {
        return path;
    }
    let end = path[1..]
        .iter()
        .position(|&c| c == b'/')
        .map_or(path.len(), |i| i + 1);
    if end != 1 {
        return path;
    }
    let home = match std::env::var_os("HOME") {
        Some(home) => bytes(home),
        #[allow(deprecated)] // Correct on Unix, and only used without HOME.
        None => match std::env::home_dir() {
            Some(home) => bytes(home.into_os_string()),
            None => return path,
        },
    };
    let mut home = home.as_slice();
    while let [rest @ .., b'/'] = home {
        home = rest;
    }
    let mut expanded = home.to_vec();
    expanded.extend_from_slice(&path[end..]);
    if expanded.is_empty() {
        expanded.push(b'/');
    }
    expanded
}

/// Python's `posixpath.expandvars`: `$name` (ASCII letters, digits and `_`) and `${name}` become
/// the value of the variable, unless it isn't set. Values aren't expanded again.
fn expand_vars(path: Vec<u8>) -> Vec<u8> {
    if !path.contains(&b'$') {
        return path;
    }
    let mut out = Vec::with_capacity(path.len());
    let mut i = 0;
    while i < path.len() {
        if path[i] != b'$' {
            out.push(path[i]);
            i += 1;
            continue;
        }
        let rest = &path[i + 1..];
        // The match after `$` and the variable name in it (None: not a variable reference).
        let (len, name) = if rest.first() == Some(&b'{') {
            match rest.iter().position(|&c| c == b'}') {
                Some(close) => (close + 1, Some(&rest[1..close])),
                None => (rest.len(), None),
            }
        } else {
            let len = rest
                .iter()
                .take_while(|c| c.is_ascii_alphanumeric() || **c == b'_')
                .count();
            (len, Some(&rest[..len]))
        };
        if len == 0 {
            out.push(b'$');
            i += 1;
            continue;
        }
        match name.and_then(variable) {
            Some(value) => out.extend_from_slice(&value),
            None => out.extend_from_slice(&path[i..=i + len]),
        }
        i += len + 1;
    }
    out
}

/// The value of the environment variable `name`, if it is set (`std::env::var_os` may panic for
/// names that can't be variables, which are never set).
fn variable(name: &[u8]) -> Option<Vec<u8>> {
    if name.is_empty() || name.contains(&b'=') || name.contains(&0) {
        return None;
    }
    let name = os_string(name.to_vec());
    std::env::var_os(OsStr::new(&name)).map(bytes)
}

/// huggingface_hub's `validate_repo_id`.
fn check_repo_id(repo_id: &str) -> Result<(), HubError> {
    let part = |p: &str| {
        let b = p.as_bytes();
        let word = |c: &u8| c.is_ascii_alphanumeric() || *c == b'_';
        (1..=96).contains(&b.len())
            && b.iter().all(|c| word(c) || *c == b'-' || *c == b'.')
            && word(&b[0])
            && word(&b[b.len() - 1])
    };
    let valid = match repo_id.split_once('/') {
        Some((org, name)) => part(org) && part(name),
        None => part(repo_id),
    } && !repo_id.contains("--")
        && !repo_id.contains("..")
        && !repo_id.ends_with(".git");
    if valid {
        Ok(())
    } else {
        Err(HubError::Invalid(format!(
            "invalid repository id \"{}\": it must be \"name\" or \"org/name\", of letters, \
             digits, `_`, `-` and `.`, without `--` or `..`",
            repo_id.escape_debug()
        )))
    }
}

/// Whether `s` can be one directory name of a revision or commit: not empty, `.` or `..`,
/// without `/`, `\`, `:` (no Windows drive or prefix, which would replace the cache directory
/// when joined; git ref names can't have `:` either) or NUL.
fn is_segment(s: &str) -> bool {
    !s.is_empty() && s != "." && s != ".." && !s.contains(['/', '\\', ':', '\0'])
}

fn check_revision(revision: &Revision) -> Result<(), HubError> {
    if revision.as_str().split('/').all(is_segment) {
        Ok(())
    } else {
        Err(HubError::Invalid(format!(
            "invalid revision \"{}\": its `/`-separated parts can't be empty, `.` or `..`, or \
             have `\\` or `:`",
            revision.as_str().escape_debug()
        )))
    }
}

/// The reason `filename` isn't a valid file name of a repository (huggingface_hub's
/// `_validate_relative_filename`, and not empty or with NUL).
pub(crate) fn filename_problem(filename: &str) -> Option<&'static str> {
    if filename.is_empty() || filename.contains('\0') {
        return Some("it is empty or has a NUL character");
    }
    if filename.replace('\\', "/").split('/').any(|s| s == "..") {
        return Some("it has a `..` part");
    }
    let anchored = filename.starts_with('/')
        || filename
            .split('/')
            .any(|s| s.starts_with('\\') || s.chars().nth(1) == Some(':'));
    if anchored {
        return Some("it is an absolute path, or has a Windows drive or root");
    }
    None
}

fn check_filename(filename: &str) -> Result<(), HubError> {
    match filename_problem(filename) {
        None => Ok(()),
        Some(reason) => Err(HubError::Invalid(format!(
            "invalid file name \"{}\": {reason}",
            filename.escape_debug()
        ))),
    }
}

/// The reason `transformers_weights` can't name the weights.
pub(crate) fn check_transformers_weights(name: &str) -> Result<(), String> {
    if !name.ends_with(".safetensors") && !name.ends_with(".safetensors.index.json") {
        return Err(format!(
            "`transformers_weights` must name a safetensors file (*.safetensors) or index \
             (*.safetensors.index.json), found \"{}\"",
            name.escape_debug()
        ));
    }
    match filename_problem(name) {
        None => Ok(()),
        Some(reason) => Err(format!(
            "`transformers_weights` must name a file inside the repository, found \"{}\": {reason}",
            name.escape_debug()
        )),
    }
}
