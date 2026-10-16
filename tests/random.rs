//! Random number generation. A single test, because the generator is global state.

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::{DType, Tensor};

#[test]
fn seeded_sampling() {
    homura::manual_seed(42);
    let a = homura::rand([1000]);
    let n = homura::randn([10001]);
    homura::manual_seed(42);
    let b = homura::rand([1000]);
    assert!(
        a.equal(&b).all().item::<bool>(),
        "same seed must give the same numbers"
    );
    assert!(!a.equal(homura::rand([1000])).all().item::<bool>());

    let (lo, hi) = (a.min().item::<f32>(), a.max().item::<f32>());
    assert!(
        (0.0..0.01).contains(&lo) && (0.99..1.0).contains(&hi),
        "{lo} {hi}"
    );
    let (mean, std) = (n.mean().item::<f32>(), n.std(1).item::<f32>());
    assert!(
        mean.abs() < 0.05 && (std - 1.0).abs() < 0.05,
        "mean {mean}, std {std}"
    );

    let r = homura::randint(-3, 4, [2000]);
    assert_eq!(r.dtype(), DType::I64);
    assert_eq!((r.min().item::<i64>(), r.max().item::<i64>()), (-3, 3));

    let h = homura::randn_with([100], DType::BF16);
    assert_eq!(h.dtype(), DType::BF16);
    let x = homura::zeros([3, 400]);
    x.transpose(0, 1).uniform_(2.0, 3.0);
    assert!(
        x.greater_equal(2.0)
            .logical_and(x.less(3.0))
            .all()
            .item::<bool>()
    );
    x.normal_(5.0, 0.1);
    assert!((x.mean().item::<f32>() - 5.0).abs() < 0.02);
    assert_eq!(homura::rand_like(&h).dtype(), DType::BF16);

    // Rejected calls don't advance the generator.
    homura::manual_seed(7);
    let expected = homura::rand([4]);
    homura::manual_seed(7);
    for invalid in [
        (|| {
            homura::zeros_with([4], DType::F64).uniform_(-1e308, 1e308);
        }) as fn(),
        || {
            homura::randint_with(70000, 70001, [4], DType::F16);
        },
    ] {
        assert!(std::panic::catch_unwind(invalid).is_err());
    }
    assert!(homura::rand([4]).equal(&expected).all().item::<bool>());
    rejected_fills_leave_the_generator_alone();
}

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

/// Review round 31 (R42): `uniform_` and `normal_` check everything before they draw samples,
/// so a rejected call doesn't advance the generator. Checked in grad mode (not recorded) and in
/// no-grad mode, on overlapping targets that are views or not, and for the errors that come
/// first: autograd's, then the arguments.
fn rejected_fills_leave_the_generator_alone() {
    const OVERLAP: &str = "unsupported operation: more than one element of the written-to tensor \
                           may refer to a single memory location. Please deep_clone() the tensor \
                           before performing the operation.";
    type Fill = fn(&Tensor);
    let fills: [(&str, Fill); 2] = [
        ("uniform_", |t| _ = t.uniform_(0.0, 1.0)),
        ("normal_", |t| _ = t.normal_(0.0, 1.0)),
    ];
    let check = |name: &str, target: &Tensor, message: &str, fill: &dyn Fn(&Tensor)| {
        homura::manual_seed(123);
        let expected = homura::rand([8]).to_vec::<f32>();
        homura::manual_seed(123);
        let data = target.to(DType::F32).to_vec::<f32>();
        // A graph that saved the target's data: a version change would make it fail.
        let probe = Tensor::from_slice(&[1.0f32], [1]);
        probe.requires_grad_(true);
        let saved = {
            let _grad_mode = homura::enable_grad();
            (&probe * &target.detach()).sum()
        };
        assert_eq!(panic_message(|| fill(target)), message, "{name}");
        assert_eq!(homura::rand([8]).to_vec::<f32>(), expected, "{name}");
        assert_eq!(target.to(DType::F32).to_vec::<f32>(), data, "{name}");
        saved.backward();
    };
    for (name, fill) in fills {
        for no_grad in [false, true] {
            let _mode = homura::set_grad_enabled(!no_grad);
            let view = homura::ones([1]).expand([4]);
            let detached = homura::ones([1]).expand([4]).detach();
            for target in [&view, &detached] {
                check(name, target, OVERLAP, &fill);
                assert!(target.grad_fn().is_none() && !target.requires_grad());
            }
            // Invalid arguments come before the overlap.
            let message = if name == "uniform_" {
                "uniform_ expects to return a [from, to) range, but found from=2 > to=1"
            } else {
                "normal expects std >= 0.0, but found std -1"
            };
            check(name, &detached, message, &|t| {
                if name == "uniform_" {
                    t.uniform_(2.0, 1.0);
                } else {
                    t.normal_(0.0, -1.0);
                }
            });
        }
        // Leaves and views that need recording are rejected first, also with invalid arguments.
        let x = homura::ones([4]);
        x.requires_grad_(true);
        let leaf = "a leaf Variable that requires grad is being used in an in-place operation.";
        check(name, &x, leaf, &fill);
        check(name, &x, leaf, &|t| _ = t.uniform_(2.0, 1.0));
        let view_of_leaf =
            "a view of a leaf Variable that requires grad is being used in an in-place operation.";
        check(name, &x.view([4]), view_of_leaf, &fill);
    }
}
