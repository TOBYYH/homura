//! Builds the ROCm layer (`csrc/`) into `libhomura_rocm.a` when the `rocm` feature is enabled.
//!
//! - ROCm is looked up in order: `ROCM_PATH`, `hipconfig --rocmpath`, `/opt/rocm`, `/usr`.
//! - `HOMURA_OFFLOAD_ARCH` lists the GPU architectures to compile for, separated by `;`
//!   (default `gfx1201`). Only the architectures in [`SUPPORTED`] are accepted.
//! - `HOMURA_ROCM_CHECK_ONLY=1` (and `DOCS_RS`) skip the native build, for `cargo check`,
//!   `clippy` and documentation on machines without ROCm. Nothing can be linked then.
//!
//! The objects are combined into one relocatable object in which only the `hmr_*` entry points
//! stay global, so that the C++ internals and template instantiations can't clash with symbols
//! of the program homura is linked into.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The architectures `HOMURA_OFFLOAD_ARCH` accepts. gfx1201 is tested on hardware; gfx1200 is
/// only compiled, and using it prints a warning (`src/backend/rocm/context.rs`).
const SUPPORTED: &[&str] = &["gfx1201", "gfx1200"];

const DEFAULT_ARCH: &str = "gfx1201";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    for var in [
        "ROCM_PATH",
        "HOMURA_OFFLOAD_ARCH",
        "HOMURA_ROCM_CHECK_ONLY",
        "DOCS_RS",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    if env::var_os("CARGO_FEATURE_ROCM").is_none() {
        return;
    }
    let archs = match offload_archs(env::var("HOMURA_OFFLOAD_ARCH").ok().as_deref()) {
        Ok(archs) => archs,
        Err(message) => fail(&message),
    };
    // The list as built (trimmed, without duplicates), for the Rust side's tests.
    println!(
        "cargo:rustc-env=HOMURA_ROCM_OFFLOAD_ARCHS={}",
        archs.join(";")
    );
    if env::var_os("DOCS_RS").is_some() || env::var("HOMURA_ROCM_CHECK_ONLY").as_deref() == Ok("1")
    {
        return;
    }
    let rocm = rocm_root();
    let tools = Tools::find(&rocm);
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let obj_dir = out.join("obj");
    std::fs::create_dir_all(&obj_dir).expect("create the object directory");
    let sources = sources(Path::new("csrc"));
    let objects = compile(&tools, &archs, &sources, &obj_dir);
    archive(&tools, &objects, &out);

    let lib = rocm.join("lib");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=homura_rocm");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    // Only for homura's own tests, examples and binaries: cargo doesn't pass link arguments to
    // dependent crates, whose programs need the ROCm library directory on the dynamic linker's
    // path (see the README). Their build scripts get it as `DEP_HOMURA_ROCM_LIB`, which
    // homura-models uses for its own tests.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    println!("cargo:lib={}", lib.display());
}

fn fail(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(1)
}

/// The architectures of `HOMURA_OFFLOAD_ARCH` (`None`: unset), checked against [`SUPPORTED`].
fn offload_archs(value: Option<&str>) -> Result<Vec<String>, String> {
    let value = value.unwrap_or(DEFAULT_ARCH);
    let mut archs: Vec<String> = Vec::new();
    for arch in value.split(';').map(str::trim) {
        if arch.is_empty() {
            continue;
        }
        if !SUPPORTED.contains(&arch) {
            let why = if arch.starts_with("gfx9") {
                " (wave64 GPUs are not supported: homura's kernels are written for wave32)"
            } else {
                ""
            };
            return Err(format!(
                "HOMURA_OFFLOAD_ARCH: {arch} is not supported{why}; supported architectures: {}",
                SUPPORTED.join(", ")
            ));
        }
        if !archs.iter().any(|a| a == arch) {
            archs.push(arch.to_string());
        }
    }
    if archs.is_empty() {
        return Err(format!(
            "HOMURA_OFFLOAD_ARCH is empty; set it to one or more of: {}",
            SUPPORTED.join(", ")
        ));
    }
    Ok(archs)
}

fn rocm_root() -> PathBuf {
    let has_hipcc = |root: &Path| root.join("bin").join("hipcc").is_file();
    if let Some(root) = env::var_os("ROCM_PATH") {
        let root = PathBuf::from(root);
        if has_hipcc(&root) {
            return root;
        }
        fail(&format!(
            "ROCM_PATH is {}, but it has no bin/hipcc",
            root.display()
        ));
    }
    if let Ok(out) = Command::new("hipconfig").arg("--rocmpath").output() {
        let root = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
        if out.status.success() && has_hipcc(&root) {
            return root;
        }
    }
    for root in ["/opt/rocm", "/usr"] {
        if has_hipcc(Path::new(root)) {
            return PathBuf::from(root);
        }
    }
    fail(
        "ROCm not found (looked at ROCM_PATH, hipconfig --rocmpath, /opt/rocm and /usr); install \
         ROCm, set ROCM_PATH, or build without the `rocm` feature",
    )
}

struct Tools {
    hipcc: PathBuf,
    ld: PathBuf,
    objcopy: PathBuf,
    ar: PathBuf,
}

impl Tools {
    /// hipcc and the LLVM tools of the ROCm installation, falling back to `PATH`.
    fn find(rocm: &Path) -> Tools {
        let llvm = rocm.join("lib").join("llvm").join("bin");
        let tool = |name: &str| {
            let bundled = llvm.join(name);
            if bundled.is_file() {
                return bundled;
            }
            let found = env::var_os("PATH").and_then(|path| {
                env::split_paths(&path)
                    .map(|dir| dir.join(name))
                    .find(|p| p.is_file())
            });
            found.unwrap_or_else(|| {
                fail(&format!(
                    "{name} not found in {} or on PATH",
                    llvm.display()
                ))
            })
        };
        Tools {
            hipcc: rocm.join("bin").join("hipcc"),
            ld: tool("ld.lld"),
            objcopy: tool("llvm-objcopy"),
            ar: tool("llvm-ar"),
        }
    }
}

/// The `.hip` files of `dir`, sorted.
fn sources(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read csrc")
        .map(|e| e.expect("read csrc").path())
        .filter(|p| p.extension().is_some_and(|e| e == "hip"))
        .collect();
    files.sort();
    files
}

/// Compiles every source with hipcc, in parallel.
fn compile(tools: &Tools, archs: &[String], sources: &[PathBuf], obj_dir: &Path) -> Vec<PathBuf> {
    let define = format!("-DHMR_OFFLOAD_ARCHS=\"{}\"", archs.join(";"));
    let jobs: Vec<_> = sources
        .iter()
        .map(|src| {
            let obj = obj_dir.join(src.file_stem().unwrap()).with_extension("o");
            let mut cmd = Command::new(&tools.hipcc);
            cmd.args(["-c", "-O3", "-std=c++17", "-fPIC", "-fvisibility=hidden"])
                .args(["-ffp-contract=off", "-Wall", "-Werror"])
                .args(archs.iter().map(|a| format!("--offload-arch={a}")))
                .arg(&define)
                .arg("-Icsrc")
                .arg(src)
                .arg("-o")
                .arg(&obj);
            let src = src.clone();
            std::thread::spawn(move || (src, obj, cmd.output()))
        })
        .collect();
    let mut objects = Vec::new();
    let mut failed = false;
    for job in jobs {
        let (src, obj, output) = job.join().expect("compiler thread");
        match output {
            Ok(o) if o.status.success() => objects.push(obj),
            Ok(o) => {
                eprintln!(
                    "hipcc failed on {}:\n{}{}",
                    src.display(),
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                );
                failed = true;
            }
            Err(e) => fail(&format!("cannot run {}: {e}", tools.hipcc.display())),
        }
    }
    if failed {
        fail("compiling the ROCm layer failed");
    }
    objects
}

/// Combines the objects into `libhomura_rocm.a` with only the `hmr_*` symbols global.
fn archive(tools: &Tools, objects: &[PathBuf], out: &Path) {
    let combined = out.join("homura_rocm_combined.o");
    let localized = out.join("homura_rocm.o");
    let lib = out.join("libhomura_rocm.a");
    run(Command::new(&tools.ld)
        .arg("-r")
        .args(objects)
        .arg("-o")
        .arg(&combined));
    run(Command::new(&tools.objcopy)
        .args(["--wildcard", "--keep-global-symbol=hmr_*"])
        .arg(&combined)
        .arg(&localized));
    let _ = std::fs::remove_file(&lib);
    run(Command::new(&tools.ar)
        .arg("rcsD")
        .arg(&lib)
        .arg(&localized));
}

fn run(cmd: &mut Command) {
    let output = cmd
        .output()
        .unwrap_or_else(|e| fail(&format!("cannot run {:?}: {e}", cmd.get_program())));
    if !output.status.success() {
        fail(&format!(
            "{:?} failed:\n{}{}",
            cmd,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
}
