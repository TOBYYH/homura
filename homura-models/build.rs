//! Lets homura-models' own tests and examples find the HIP runtime when homura is built with ROCm:
//! homura's build script passes the ROCm library directory as `DEP_HOMURA_ROCM_LIB`, which is
//! set only with the `rocm` feature. Programs that depend on homura-models need the directory on
//! the dynamic linker's path instead (see homura's README).

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_HOMURA_ROCM_LIB");
    if let Ok(lib) = std::env::var("DEP_HOMURA_ROCM_LIB") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    }
}
