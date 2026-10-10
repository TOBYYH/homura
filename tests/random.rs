//! Random number generation. A single test, because the generator is global state.

use homura::DType;

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
}
