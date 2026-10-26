//! `hub`: repositories in local directories and in caches laid out as huggingface_hub lays them
//! out, revisions, names and the weight files picked, against the rules that
//! tools/golden/check_m3_hub.py checks in huggingface_hub 2.3.0 and transformers 5.20.2.

use std::fs;
use std::path::{Path, PathBuf};

use homura_models::hub::{HubError, Repo, Revision, WeightFiles, WeightFormat};

const HASH: &str = "0123456789abcdef0123456789abcdef01234567";

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("hub")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write(path: &Path, data: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, data).unwrap();
}

/// A cache in huggingface_hub's layout with the repository `org/mini`: the files in `blobs`,
/// linked from `snapshots/<commit>`, and the refs.
fn write_cache(cache: &Path, files: &[(&str, &[u8])], refs: &[(&str, &str)], commit: &str) {
    let repo = cache.join("models--org--mini");
    fs::create_dir_all(repo.join("snapshots").join(commit)).unwrap();
    for (i, (name, data)) in files.iter().enumerate() {
        let blob = repo.join("blobs").join(format!("blob{i}"));
        write(&blob, data);
        let link = repo.join("snapshots").join(commit).join(name);
        fs::create_dir_all(link.parent().unwrap()).unwrap();
        let depth = name.split('/').count() + 1;
        let target = format!("{}blobs/blob{i}", "../".repeat(depth));
        #[cfg(unix)]
        std::os::unix::fs::symlink(target, &link).unwrap();
        #[cfg(not(unix))]
        {
            let _ = target;
            fs::copy(&blob, &link).unwrap();
        }
    }
    for (name, content) in refs {
        write(&repo.join("refs").join(name), content.as_bytes());
    }
}

fn cached(cache: &Path, revision: &str) -> Result<Repo, HubError> {
    Repo::cached_in(cache, "org/mini", Revision::new(revision))
}

fn is_invalid<T: std::fmt::Debug>(r: Result<T, HubError>) -> bool {
    matches!(r, Err(HubError::Invalid(_)))
}

fn is_not_found<T: std::fmt::Debug>(r: Result<T, HubError>) -> bool {
    matches!(r, Err(HubError::NotFound(_)))
}

#[test]
fn local_repositories() {
    let dir = scratch("local");
    write(&dir.join("config.json"), b"{}");
    write(&dir.join("sub/a.txt"), b"");
    let repo = Repo::local(&dir).unwrap();
    assert_eq!(repo.dir(), dir);
    assert_eq!(
        (repo.repo_id(), repo.revision(), repo.commit()),
        (None, None, None)
    );
    assert_eq!(repo.path("config.json").unwrap(), dir.join("config.json"));
    assert_eq!(repo.path("sub/a.txt").unwrap(), dir.join("sub/a.txt"));
    assert_eq!(repo.path("./sub//a.txt").unwrap(), dir.join("sub/a.txt"));
    // A directory isn't a file.
    assert!(is_not_found(repo.path("sub")));
    assert!(is_not_found(repo.path("missing.json")));
    let e = Repo::local(dir.join("config.json")).unwrap_err();
    assert!(e.to_string().contains("isn't a directory"), "{e}");
    assert!(is_not_found(Repo::local(dir.join("none"))));
}

/// huggingface_hub's `validate_repo_id` (check_m3_hub.py `REPO_IDS`).
#[test]
fn repository_ids() {
    let long = format!("{}/{}", "a".repeat(96), "b".repeat(96));
    let too_long = "a".repeat(97);
    let long_name = format!("org/{}", "b".repeat(97));
    let cases: &[(&str, bool)] = &[
        ("microsoft/resnet-50", true),
        ("gpt2", true),
        ("a/b", true),
        ("A_b.c-d/x_Y.z-9", true),
        ("_a/b_", true),
        (&long, true),
        (&too_long, false),
        (&long_name, false),
        ("", false),
        ("/", false),
        ("a/", false),
        ("/a", false),
        ("a/b/c", false),
        ("a b", false),
        ("-a/b", false),
        ("a/b-", false),
        (".a/b", false),
        ("a/b.", false),
        ("a--b/c", false),
        ("a/b..c", false),
        ("a/b.git", false),
        ("a/b.gitx", true),
        ("a/b:c", false),
        ("a\\b", false),
        ("a/b\n", false),
        ("é/b", false),
    ];
    let cache = scratch("repo_ids");
    for &(repo_id, valid) in cases {
        let r = Repo::cached_in(&cache, repo_id, Revision::main());
        // A valid id isn't in the empty cache.
        if valid {
            assert!(is_not_found(r), "{repo_id:?}");
        } else {
            assert!(is_invalid(r), "{repo_id:?}");
        }
    }
    // The repository directory joins the parts with `--`.
    fs::create_dir_all(cache.join("models--gpt2/snapshots/main")).unwrap();
    assert!(Repo::cached_in(&cache, "gpt2", Revision::main()).is_ok());
}

/// huggingface_hub's `_validate_relative_filename` (check_m3_hub.py `FILENAMES`).
#[test]
fn file_names() {
    let dir = scratch("file_names");
    let repo = Repo::local(&dir).unwrap();
    let cases = [
        ("config.json", true),
        ("sub/model.safetensors", true),
        ("./model.safetensors", true),
        ("a//b", true),
        ("..", false),
        ("../x", false),
        ("a/../b", false),
        ("a\\..\\b", false),
        ("/etc/x", false),
        ("\\x", false),
        ("C:x", false),
        ("a/C:x", false),
        ("\\\\host\\share", false),
        ("a/\\b", false),
        ("1:x", false),
        ("a:", false),
        ("..x", true),
        ("x..", true),
        ("", false),
        ("a\0b", false),
    ];
    for (name, valid) in cases {
        let r = repo.path(name);
        if valid {
            assert!(is_not_found(r), "{name:?}");
        } else {
            assert!(is_invalid(r), "{name:?}");
        }
    }
    // Invalid names are rejected even when the file they would name exists.
    write(&dir.join("sub/x"), b"");
    write(&dir.join("x"), b"");
    assert!(repo.path("sub/x").is_ok());
    assert!(is_invalid(repo.path("sub/../x")));
}

/// The offline resolution of revisions (check_m3_hub.py `revisions`).
#[test]
fn revisions() {
    let config: &[(&str, &[u8])] = &[("config.json", b"{}")];
    let cache = scratch("revisions");
    write_cache(&cache, config, &[("main", HASH)], HASH);
    let repo = cached(&cache, "main").unwrap();
    let snapshot = cache.join("models--org--mini/snapshots").join(HASH);
    assert_eq!(repo.dir(), snapshot);
    assert_eq!(repo.repo_id(), Some("org/mini"));
    assert_eq!(repo.revision(), Some(&Revision::main()));
    assert_eq!(repo.commit(), Some(HASH));
    assert_eq!(fs::read(repo.path("config.json").unwrap()).unwrap(), b"{}");
    assert_eq!(
        Repo::cached_in(&cache, "org/mini", Revision::default()).unwrap(),
        repo
    );

    // The ref's content stripped of whitespace.
    for content in [
        format!("{HASH}\n"),
        format!("{HASH}\r\n"),
        format!(" {HASH}"),
        format!("\n{HASH}\n\n"),
    ] {
        let cache = scratch("revisions_strip");
        write_cache(&cache, config, &[("main", &content)], HASH);
        assert_eq!(cached(&cache, "main").unwrap().commit(), Some(HASH));
    }
    // A branch with `/` is a file in a subdirectory of refs.
    let cache = scratch("revisions_branch");
    write_cache(&cache, config, &[("feature/x", HASH)], HASH);
    assert_eq!(cached(&cache, "feature/x").unwrap().dir(), {
        cache.join("models--org--mini/snapshots").join(HASH)
    });
    assert!(is_not_found(cached(&cache, "main")));
    // Without a ref, the revision names the snapshot directory, whatever it is.
    let cache = scratch("revisions_noref");
    write_cache(&cache, config, &[], HASH);
    assert_eq!(cached(&cache, HASH).unwrap().commit(), Some(HASH));
    assert!(is_not_found(cached(&cache, &HASH.to_uppercase())));
    assert!(is_not_found(cached(&cache, "main")));
    let cache = scratch("revisions_named");
    write_cache(&cache, config, &[], "main");
    assert_eq!(cached(&cache, "main").unwrap().commit(), Some("main"));
    let upper = HASH.to_uppercase();
    let cache = scratch("revisions_upper");
    write_cache(&cache, config, &[], &upper);
    assert_eq!(
        cached(&cache, &upper).unwrap().commit(),
        Some(upper.as_str())
    );
    let cache = scratch("revisions_abc");
    write_cache(&cache, config, &[("main", "abc")], "abc");
    assert_eq!(cached(&cache, "main").unwrap().commit(), Some("abc"));
    // A snapshot is one directory.
    let cache = scratch("revisions_slash");
    write_cache(&cache, config, &[("main", "x/y")], "x/y");
    assert!(is_invalid(cached(&cache, "main")));
    assert!(is_not_found(cached(&cache, "x/y")));
    // Missing ref, snapshot and repository.
    let cache = scratch("revisions_missing");
    write_cache(&cache, config, &[("main", HASH)], HASH);
    assert!(is_not_found(cached(&cache, "dev")));
    assert!(is_not_found(cached(&cache, &"f".repeat(40))));
    let cache = scratch("revisions_dangling");
    write_cache(&cache, config, &[("main", &"f".repeat(40))], HASH);
    let e = cached(&cache, "main").unwrap_err();
    assert!(matches!(e, HubError::NotFound(_)));
    assert!(
        e.to_string()
            .contains("hf download org/mini --revision main"),
        "{e}"
    );
    let e = Repo::cached_in(&cache, "org/none", Revision::main()).unwrap_err();
    assert!(matches!(e, HubError::NotFound(_)));
    assert!(e.to_string().contains("hf download org/none"), "{e}");
    // A ref that isn't text.
    let cache = scratch("revisions_binary");
    write_cache(&cache, config, &[], HASH);
    write(&cache.join("models--org--mini/refs/main"), b"\xff\xfe");
    assert!(is_invalid(cached(&cache, "main")));
    assert!(is_invalid(cached(&cache, "")));
}

#[test]
fn revisions_are_checked() {
    let cache = scratch("revisions_checked");
    write_cache(&cache, &[("config.json", b"{}")], &[("main", HASH)], HASH);
    // A file outside refs that a revision would reach.
    write(&cache.join("models--org--mini/outside"), HASH.as_bytes());
    for revision in [
        "",
        "/main",
        "main/",
        "a//b",
        ".",
        "..",
        "../outside",
        "a/./b",
        "a\\b",
        "a\0",
    ] {
        assert!(is_invalid(cached(&cache, revision)), "{revision:?}");
    }
    // Windows drives and prefixes, which would replace the cache directory when joined there
    // (R118): rejected before any path is made, also where the snapshot or ref exists here.
    for name in ["C:", "C:outside", "D:weights", "feature/C:x", "a:b"] {
        let cache = scratch("revisions_drives");
        write_cache(&cache, &[("config.json", b"{}")], &[], name);
        write(
            &cache.join("models--org--mini/refs").join(name),
            HASH.as_bytes(),
        );
        assert!(is_invalid(cached(&cache, name)), "{name:?}");
    }
    for content in ["C:", "C:outside", "D:weights", "x:y"] {
        let cache = scratch("revisions_drive_refs");
        write_cache(
            &cache,
            &[("config.json", b"{}")],
            &[("main", content)],
            content,
        );
        assert!(is_invalid(cached(&cache, "main")), "{content:?}");
    }
    assert_eq!(Revision::new("pr/1").as_str(), "pr/1");
    assert_eq!(Revision::new("v1.0").to_string(), "v1.0");
    assert_eq!(Revision::default(), Revision::main());
}

const SAFE: &str = "model.safetensors";
const SAFE_INDEX: &str = "model.safetensors.index.json";
const BIN: &str = "pytorch_model.bin";
const BIN_INDEX: &str = "pytorch_model.bin.index.json";

fn picked(repo: &Repo, format: Option<WeightFormat>) -> Option<&'static str> {
    match repo.weight_files(format, None) {
        Ok(WeightFiles {
            path,
            format: found,
            sharded,
            ..
        }) => {
            let name = path.file_name().unwrap().to_str().unwrap();
            let name = [SAFE, SAFE_INDEX, BIN, BIN_INDEX]
                .into_iter()
                .find(|n| *n == name)
                .unwrap();
            assert_eq!(
                found == WeightFormat::Safetensors,
                name.starts_with("model.")
            );
            assert_eq!(sharded, name.ends_with(".index.json"));
            Some(name)
        }
        Err(HubError::NotFound(_)) => None,
        Err(e) => panic!("{e}"),
    }
}

/// transformers' discovery, in a local directory and in the cache (check_m3_hub.py
/// `DISCOVERY`: the present files, then the file picked with no format, safetensors and PyTorch).
#[test]
fn weight_discovery() {
    type Case = (&'static [&'static str], [Option<&'static str>; 3]);
    let cases: &[Case] = &[
        (
            &[SAFE, SAFE_INDEX, BIN, BIN_INDEX],
            [Some(SAFE), Some(SAFE), Some(BIN)],
        ),
        (
            &[SAFE_INDEX, BIN, BIN_INDEX],
            [Some(SAFE_INDEX), Some(SAFE_INDEX), Some(BIN)],
        ),
        (&[SAFE, SAFE_INDEX], [Some(SAFE), Some(SAFE), None]),
        (&[BIN, BIN_INDEX], [Some(BIN), None, Some(BIN)]),
        (&[SAFE_INDEX], [Some(SAFE_INDEX), Some(SAFE_INDEX), None]),
        (&[BIN_INDEX], [Some(BIN_INDEX), None, Some(BIN_INDEX)]),
        (
            &[SAFE_INDEX, BIN_INDEX],
            [Some(SAFE_INDEX), Some(SAFE_INDEX), Some(BIN_INDEX)],
        ),
        (&[], [None, None, None]),
    ];
    let formats = [
        None,
        Some(WeightFormat::Safetensors),
        Some(WeightFormat::PyTorch),
    ];
    for (i, (present, expected)) in cases.iter().enumerate() {
        let files: Vec<(&str, &[u8])> = present.iter().map(|n| (*n, &b"x"[..])).collect();
        let dir = scratch(&format!("discovery_local_{i}"));
        for (name, data) in &files {
            write(&dir.join(name), data);
        }
        let cache = scratch(&format!("discovery_cache_{i}"));
        write_cache(&cache, &files, &[("main", HASH)], HASH);
        for repo in [Repo::local(&dir).unwrap(), cached(&cache, "main").unwrap()] {
            for (format, expected) in formats.iter().zip(expected) {
                assert_eq!(picked(&repo, *format), *expected, "{present:?} {format:?}");
            }
        }
    }
    // A directory with the name of a weight file is skipped.
    let dir = scratch("discovery_dir");
    fs::create_dir_all(dir.join(SAFE)).unwrap();
    write(&dir.join(BIN), b"");
    assert_eq!(picked(&Repo::local(&dir).unwrap(), None), Some(BIN));
    // The error lists the names looked for.
    let e = Repo::local(scratch("discovery_none"))
        .unwrap()
        .weight_files(Some(WeightFormat::PyTorch), None)
        .unwrap_err();
    assert!(
        e.to_string()
            .contains("none of pytorch_model.bin, pytorch_model.bin.index.json")
    );
}

/// `transformers_weights` (check_m3_hub.py `transformers_weights`).
#[test]
fn transformers_weights() {
    let files: &[(&str, &[u8])] = &[
        ("sub/w.safetensors", b"x"),
        ("w.safetensors.index.json", b"x"),
        (SAFE, b"x"),
        (BIN, b"x"),
        ("w.bin", b"x"),
    ];
    let dir = scratch("explicit_local");
    for (name, data) in files {
        write(&dir.join(name), data);
    }
    let cache = scratch("explicit_cache");
    write_cache(&cache, files, &[("main", HASH)], HASH);
    for repo in [Repo::local(&dir).unwrap(), cached(&cache, "main").unwrap()] {
        for format in [None, Some(WeightFormat::PyTorch)] {
            let w = repo
                .weight_files(format, Some("sub/w.safetensors"))
                .unwrap();
            assert_eq!(w.path, repo.dir().join("sub/w.safetensors"));
            assert_eq!((w.format, w.sharded), (WeightFormat::Safetensors, false));
        }
        let w = repo
            .weight_files(None, Some("w.safetensors.index.json"))
            .unwrap();
        assert!(w.sharded);
        for name in [
            "w.bin",
            "../w.safetensors",
            "/abs/w.safetensors",
            "sub/../w.safetensors",
        ] {
            assert!(is_invalid(repo.weight_files(None, Some(name))), "{name}");
        }
        assert!(is_not_found(
            repo.weight_files(None, Some("missing.safetensors"))
        ));
    }
}
