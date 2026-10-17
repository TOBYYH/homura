"""Check the reference values that tests/autograd_higher_order.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order. Inputs are exact binary fractions where the
results are exact, so those are compared for equality; elsewhere the values are compared with the
closed forms the Rust tests also use, or printed with full precision for them.
"""

import math

import torch

import reference

reference.check()

F64 = torch.float64


def t(*values, dtype=F64, grad=False):
    return torch.tensor(values, dtype=dtype, requires_grad=grad)


def grad(out, inputs, grad_outputs=None, create_graph=False, allow_unused=False):
    return torch.autograd.grad(
        out, inputs, grad_outputs, create_graph=create_graph, allow_unused=allow_unused
    )


# derivatives_of_a_cube_up_to_the_fourth
x = t(0.5, -1.25, 2.0, grad=True)
for name, y in [("product", x * x * x), ("pow", x.pow(3))]:
    (g1,) = grad(y.sum(), [x], create_graph=True)
    (g2,) = grad(g1.sum(), [x], create_graph=True)
    (g3,) = grad(g2.sum(), [x], create_graph=True)
    assert g1.tolist() == [0.75, 4.6875, 12.0], name
    assert g2.tolist() == [3.0, -7.5, 12.0], name
    assert g3.tolist() == [6.0, 6.0, 6.0], name
    if name == "product":
        # Each product's derivatives are sums of constants times the factors: the third has no
        # graph left.
        assert not g3.requires_grad and g3.grad_fn is None
    else:
        # pow's derivative multiplies by x.pow(2), x.pow(1) and then x.pow(0), which still depends
        # on x: the fourth derivative is a zero gradient, not an error.
        assert g3.grad_fn.name() == "MulBackward0"
        (g4,) = grad(g3.sum(), [x])
        assert g4.tolist() == [0.0, 0.0, 0.0]


# hessian_vector_products_of_an_elementwise_function
def d2(x):
    return 2 * math.sin(x) + 4 * x * math.cos(x) - x * x * math.sin(x)


def d3(x):
    return 6 * math.cos(x) - 6 * x * math.sin(x) - x * x * math.cos(x)


xs, vs, ws = [0.3, -1.1, 2.0, 0.7], [1.0, -0.5, 0.25, 2.0], [0.5, 1.5, -1.0, 0.75]
x = t(*xs, grad=True)
v = t(*vs, grad=True)
w = t(*ws)
(g,) = grad((x.sin() * x * x).sum(), [x], create_graph=True)
(hv,) = grad(g, [x], v, create_graph=True)
expected = [d2(a) * b for a, b in zip(xs, vs)]
assert torch.allclose(hv, t(*expected), rtol=1e-13, atol=0), (hv, expected)
# The product is linear in v (H w), and its derivative in x has the third derivatives.
hw_v, hw_x = grad(hv, [v, x], w)
assert torch.allclose(hw_v, t(*[d2(a) * c for a, c in zip(xs, ws)]), rtol=1e-13, atol=0)
assert torch.allclose(
    hw_x, t(*[d3(a) * b * c for a, b, c in zip(xs, vs, ws)]), rtol=1e-13, atol=0
)

# hessian_vector_products_of_a_quadratic_form
x = t(1.0, -2.0, 0.5, grad=True)
b = t(0.5, -1.0, 2.0)
v = t(0.5, 1.0, -1.0)
symmetric = torch.tensor([[2.0, -1.0, 0.5], [-1.0, 3.0, 0.25], [0.5, 0.25, 1.0]], dtype=F64)
general = torch.tensor([[2.0, -1.0, 0.5], [0.0, 3.0, 0.25], [1.5, -0.75, 1.0]], dtype=F64)
for m, g_expected, hv_expected in [
    (symmetric, [4.75, -7.875, 2.5], [-0.5, 2.25, -0.5]),
    # The Hessian of x^T M x / 2 is (M + M^T) / 2.
    (general, [4.0, -7.625, 4.0], [-0.5, 3.0, -0.75]),
]:
    f = 0.5 * x.dot(m.mv(x)) + b.dot(x)
    (g,) = grad(f, [x], create_graph=True)
    (hv,) = grad(g, [x], v)
    assert g.tolist() == g_expected and hv.tolist() == hv_expected, (g, hv)

# gradient_penalty_of_a_small_network
X = torch.tensor([[0.5, -1.0, 0.25], [1.5, 0.75, -0.5]], dtype=F64, requires_grad=True)
W1 = torch.tensor([[0.3, -0.6], [0.8, 0.1], [-0.4, 0.5]], dtype=F64, requires_grad=True)
b1 = t(0.1, -0.2, grad=True)
w2 = t(0.7, -1.3, grad=True)


def penalty(X, W1, b1, w2):
    y = ((X @ W1 + b1).tanh() @ w2).sum()
    (gx,) = grad(y, [X], create_graph=True)
    return (gx * gx).sum()


p = penalty(X, W1, b1, w2)
grads = grad(p, [X, W1, b1, w2])
# The values tests/autograd_higher_order.rs compares with (printed with every digit).
EXPECTED_PENALTY = 1.3059498758318049
EXPECTED = {
    "X": [-0.639555733871122, 0.7780865554975789, 0.4144641821735521,
          -0.23447136145361183, -0.08458010976176594, 0.21721491944734928],
    "W1": [0.9870142936783939, -0.5815705400743374, -0.6102296086399324,
           -1.8600872720033894, -0.4797688917101419, 1.8622800716570966],
    "b1": [0.6445120743314008, 1.7789678627069234],
    "w2": [1.1059625572918017, -1.4136353550456529],
}  # fmt: skip
print(f"penalty = {p.item()!r}")
assert math.isclose(p.item(), EXPECTED_PENALTY, rel_tol=1e-13)
for (name, expected), gr in zip(EXPECTED.items(), grads):
    print(f"d penalty / d {name} = {[repr(v) for v in gr.flatten().tolist()]}")
    assert torch.allclose(gr.flatten(), t(*expected), rtol=1e-13, atol=0), name


# two_stage_products_across_dtypes
for src, dst in [
    (torch.float32, torch.float64),
    (torch.float64, torch.float32),
    (torch.float64, torch.float16),
    (torch.float64, torch.bfloat16),
]:
    x = t(1.5, -0.75, 2.0, dtype=src, grad=True)
    u = t(0.5, 2.0, -1.0, dtype=dst, grad=True)
    w = t(1.0, 0.25, -2.0, dtype=src)
    y = x.to(dst).pow(2)
    (g,) = grad(y, [x], u, create_graph=True)
    assert g.dtype == src and g.tolist() == [1.5, -3.0, -4.0], (src, dst, g)
    gx, gu = grad(g, [x, u], w)
    assert gx.dtype == src and gx.tolist() == [1.0, 1.0, 4.0], (src, dst, gx)
    assert gu.dtype == dst and gu.tolist() == [3.0, -0.375, -8.0], (src, dst, gu)
    print(f"{src} -> {dst}: first derivative {g.grad_fn.name()}")

print("check_m1_higher_order: all reference assertions passed")
