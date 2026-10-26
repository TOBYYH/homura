//! `hub::cache_dir` against huggingface_hub 2.3.0's `constants.HF_HUB_CACHE` in the environments of
//! tools/golden/check_m3_hub.py `CACHE_DIRS`.
//!
//! This binary has one test, which changes the process environment: no other thread reads it
//! meanwhile.

use std::ffi::OsStr;

use homura_models::hub::cache_dir;

const VARIABLES: [&str; 10] = [
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HOME",
    "XDG_CACHE_HOME",
    "HOME",
    "V",
    "A",
    "B",
    "1",
    "A B",
];

#[test]
fn cache_dir_follows_huggingface_hub() {
    let cases: &[(&[(&str, &str)], &str)] = &[
        (&[], "/home/h/.cache/huggingface/hub"),
        (&[("XDG_CACHE_HOME", "/x")], "/x/huggingface/hub"),
        (&[("XDG_CACHE_HOME", "/x"), ("HF_HOME", "/hf")], "/hf/hub"),
        (
            &[("HF_HOME", "/hf"), ("HUGGINGFACE_HUB_CACHE", "/legacy")],
            "/legacy",
        ),
        (
            &[
                ("HF_HOME", "/hf"),
                ("HUGGINGFACE_HUB_CACHE", "/legacy"),
                ("HF_HUB_CACHE", "/new"),
            ],
            "/new",
        ),
        (&[("HF_HUB_CACHE", "~/c")], "/home/h/c"),
        (&[("HF_HUB_CACHE", "$V/c"), ("V", "/v")], "/v/c"),
        (&[("HF_HUB_CACHE", "${V}/c"), ("V", "/v")], "/v/c"),
        (&[("HF_HUB_CACHE", "$UNSET/c")], "$UNSET/c"),
        (&[("HUGGINGFACE_HUB_CACHE", "~/legacy")], "/home/h/legacy"),
        (
            &[("HUGGINGFACE_HUB_CACHE", "$V/legacy"), ("V", "/v")],
            "/v/legacy",
        ),
        (&[("HF_HOME", "~/hf")], "/home/h/hf/hub"),
        (&[("XDG_CACHE_HOME", "~/x")], "/home/h/x/huggingface/hub"),
        (
            &[("XDG_CACHE_HOME", "$V/x"), ("V", "/v")],
            "/v/x/huggingface/hub",
        ),
        (&[("HF_HOME", "$A"), ("A", "$B"), ("B", "/b")], "/b/hub"),
        (&[("HF_HUB_CACHE", "$A"), ("A", "$B"), ("B", "/b")], "$B"),
        (&[("HF_HUB_CACHE", "")], ""),
        (&[("HF_HOME", "")], "hub"),
        (&[("XDG_CACHE_HOME", "")], "huggingface/hub"),
        (&[("HF_HUB_CACHE", "~")], "/home/h"),
        // Python looks the user up; homura leaves the path as it is.
        (&[("HF_HUB_CACHE", "~user/c")], "~user/c"),
        (&[("HF_HUB_CACHE", "a~/b")], "a~/b"),
        (&[("HF_HUB_CACHE", "a/~")], "a/~"),
        (&[("HF_HUB_CACHE", "$1/x"), ("1", "/one")], "/one/x"),
        (&[("HF_HUB_CACHE", "${}/x")], "${}/x"),
        (&[("HF_HUB_CACHE", "${V/x"), ("V", "/v")], "${V/x"),
        (&[("HF_HUB_CACHE", "${V}/c}"), ("V", "/v")], "/v/c}"),
        (&[("HF_HUB_CACHE", "$V$V"), ("V", "/v")], "/v/v"),
        (&[("HF_HUB_CACHE", "$V_x/c"), ("V", "/v")], "$V_x/c"),
        (&[("HF_HUB_CACHE", "$V-x/c"), ("V", "/v")], "/v-x/c"),
        (&[("HF_HUB_CACHE", "${A B}/c"), ("A B", "/ab")], "/ab/c"),
        (&[("HF_HUB_CACHE", "/c$")], "/c$"),
        (
            &[("HF_HUB_CACHE", "~/c"), ("HOME", "/home/h/")],
            "/home/h/c",
        ),
        (&[("HF_HUB_CACHE", "~"), ("HOME", "/")], "/"),
        (&[("HF_HUB_CACHE", "~/c"), ("HOME", "/")], "/c"),
        (&[("HF_HUB_CACHE", "~/c"), ("HOME", "/h//")], "/h/c"),
        (&[("HF_HUB_CACHE", "~/c"), ("HOME", "")], "/c"),
        (&[("HF_HOME", "/hf/")], "/hf/hub"),
        (&[("HF_HOME", "/hf"), ("HUGGINGFACE_HUB_CACHE", "")], ""),
    ];
    for (env, expected) in cases {
        for name in VARIABLES {
            // SAFETY: the only test of this binary; no other thread uses the environment.
            unsafe { std::env::remove_var(name) };
        }
        // SAFETY: as above.
        unsafe { std::env::set_var("HOME", "/home/h") };
        for (name, value) in *env {
            // SAFETY: as above.
            unsafe { std::env::set_var(name, value) };
        }
        // As text: paths compare equal whatever their repeated separators.
        assert_eq!(cache_dir().as_os_str(), OsStr::new(expected), "{env:?}");
    }
}
