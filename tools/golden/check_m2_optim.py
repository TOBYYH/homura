"""Asserts the PyTorch behavior that tests/optim.rs and tests/optim_default_dtype.rs encode for
the optimizers of M2 step 4a, in the order of the tests. The multi-step values are in
tests/golden/m2_optim.safetensors (tools/golden/gen_m2_optim.py).

Usage: python tools/golden/check_m2_optim.py (needs torch).
"""

import math
import warnings

import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim import optimizer as optimizer_module

import reference

reference.check()


def raises(f, exc, message):
    try:
        f()
    except exc as e:
        assert message in str(e), (str(e), message)
        return
    raise AssertionError(f"no {exc.__name__}: {message}")


def param(*values, dtype=torch.float64):
    return torch.nn.Parameter(torch.tensor(list(values), dtype=dtype))


def grad(p, *values):
    p.grad = torch.tensor(list(values), dtype=p.dtype)


# multi_step_golden: CPU runs the single-tensor implementations by default, with the fused
# multiply-add of add(alpha=).
assert optimizer_module._default_to_fused_or_foreach([param(1.0)], False, False) == (False, False)
p, g = param(1.0, 2.0), torch.tensor([0.3, -0.7], dtype=torch.float64)
p.grad = g.clone()
SGD([p], lr=0.1).step()
assert p.tolist() == [math.fma(gi, -0.1, pi) for pi, gi in zip([1.0, 2.0], g.tolist())]

# multi_step_golden / edge_values_golden: the rounding of PyTorch's CPU kernels (AVX2 build),
# which homura's fused optimizer kernel reproduces. add(alpha=) and a scalar lerp weight use
# fused multiply-adds, addcmul is contracted to fma(value * t1, t2, self), addcdiv isn't fused;
# F16 / BF16 compute in F32 and round once, add(alpha=) with alpha rounded to the dtype first.
assert torch.backends.cpu.get_cpu_capability() == "AVX2"
torch.manual_seed(3)


def f32(x):
    return torch.tensor(x, dtype=torch.float64).float().double().item()


for n in (12, 37):
    s_, t1, t2 = (torch.randn(n, dtype=torch.float64) for _ in range(3))
    t2 = t2.abs() + 0.5
    S, A, B = s_.tolist(), t1.tolist(), t2.tolist()
    assert s_.add(t1, alpha=0.37).tolist() == [math.fma(y, 0.37, x) for x, y in zip(S, A)]
    assert s_.lerp(t1, 0.1).tolist() == [math.fma(0.1, y - x, x) for x, y in zip(S, A)]
    assert s_.lerp(t1, 0.9).tolist() == [math.fma(0.9 - 1, y - x, y) for x, y in zip(S, A)]
    assert s_.addcmul(t1, t2, value=0.37).tolist() == [
        math.fma(0.37 * y, z, x) for x, y, z in zip(S, A, B)]
    assert s_.addcdiv(t1, t2, value=0.37).tolist() == [x + 0.37 * y / z for x, y, z in zip(S, A, B)]
for dtype in (torch.float16, torch.bfloat16):
    rd = lambda x: torch.tensor(x, dtype=torch.float64).to(dtype).double().item()
    alpha = rd(f32(0.37))
    s_, t1 = (torch.randn(37, dtype=torch.float64).to(dtype) for _ in range(2))
    S, A = s_.double().tolist(), t1.double().tolist()
    out = s_.add(t1, alpha=0.37).double().tolist()
    # The vectorized loop takes 32 elements at a time; the scalar loop for the other 5 rounds
    # alpha * t1 to the dtype before the addition.
    assert out[:32] == [rd(f32(math.fma(y, alpha, x))) for x, y in zip(S[:32], A[:32])]
    assert out[32:] == [rd(x + rd(alpha * y)) for x, y in zip(S[32:], A[32:])]
# The reviewer's R65 examples (32 elements): rounding the products first loses or overflows them.
p = torch.nn.Parameter(torch.zeros(32, dtype=torch.float16))
p.grad = torch.full((32,), 1e-5, dtype=torch.float16)
Adam([p], lr=1e-3, eps=1e-4).step()
assert torch.all(p == -0.00010132789611816406)
p = torch.nn.Parameter(torch.full((32,), 60000.0, dtype=torch.float16))
p.grad = torch.full((32,), 2.0, dtype=torch.float16)
SGD([p], lr=60000.0).step()
assert torch.all(p == -60000.0)
p = torch.nn.Parameter(torch.full((32,), 1e308, dtype=torch.float64))
p.grad = torch.full((32,), 2.0, dtype=torch.float64)
SGD([p], lr=1e308).step()
assert torch.all(p == -1e308)

# coefficients_that_overflow_their_type_are_rejected (R66): the CPU kernels convert a finite
# alpha of add() to the dtype and the value of addcdiv to its computation type with an overflow
# check. PyTorch raises in the middle of the step (homura before it), so only the parameters
# are compared here.
def step_error(cls, dtype, kw, g=1.0, device="cpu", extra=None):
    p = torch.nn.Parameter(torch.ones(32, dtype=dtype, device=device))
    p.grad = torch.full((32,), g, dtype=dtype, device=device)
    o = cls([p], **kw)
    try:
        o.step()
    except RuntimeError as e:
        return str(e), p, o
    return None, p, o


over = {torch.float16: "c10::Half", torch.bfloat16: "c10::BFloat16", torch.float32: "float"}
for dtype, big in ((torch.float16, 1e5), (torch.bfloat16, 1e40), (torch.float32, 1e40)):
    for lr, wd, g in ((big, 0.0, 1.0), (0.1, big, 1.0), (big, 0.0, 0.0)):
        e, p, o = step_error(SGD, dtype, dict(lr=lr, weight_decay=wd, momentum=0.9), g)
        assert e == f"value cannot be converted to type {over[dtype]} without overflow", e
        assert torch.all(p == 1)
e, p, _ = step_error(SGD, torch.float16, dict(lr=0.1, momentum=1e5, nesterov=True))
assert "without overflow" in e and torch.all(p == 1)
p = torch.nn.Parameter(torch.ones(32, dtype=torch.float16))
p.grad = torch.ones(32, dtype=torch.float16)
o = SGD([p], lr=0.1, momentum=0.9, dampening=-1e5)
o.step()
raises(lambda: o.step(), RuntimeError, "value cannot be converted to type c10::Half without overflow")
a, b = (torch.nn.Parameter(torch.ones(32, dtype=torch.float16)) for _ in range(2))
a.grad, b.grad = torch.ones(32, dtype=torch.float16), torch.ones(32, dtype=torch.float16)
o = SGD([{"params": [a]}, {"params": [b], "lr": 1e5}], lr=0.1)
raises(lambda: o.step(), RuntimeError, "without overflow")
assert torch.all(a < 1) and torch.all(b == 1)  # PyTorch updated the first group
for dtype in (torch.float32, torch.float16):
    e, p, o = step_error(Adam, dtype, dict(lr=1e38))
    assert e == "value cannot be converted to type float without overflow" and torch.all(p == 1)
    assert o.state[p]["step"].item() == 1  # PyTorch counted the step and updated the moments
    e, p, _ = step_error(AdamW, dtype, dict(lr=3e38, weight_decay=0.0))
    assert e == "value cannot be converted to type float without overflow" and torch.all(p == 1)
for lr, fits in ((2.4e38, True), (2.7e38, False)):
    p = torch.nn.Parameter(torch.ones(32))
    p.grad = torch.ones(32)
    o = Adam([p], lr=0.1)
    o.step()
    o.param_groups[0]["lr"] = lr
    o.param_groups[0]["betas"] = (0.5, 0.999)
    if fits:
        o.step()
        assert o.state[p]["step"].item() == 2
    else:
        raises(lambda: o.step(), RuntimeError, "value cannot be converted to type float without overflow")
e, p, _ = step_error(Adam, torch.float16, dict(lr=0.1, weight_decay=1e5))
assert e == "value cannot be converted to type c10::Half without overflow" and torch.all(p == 1)
# controls
e, p, _ = step_error(SGD, torch.float16, dict(lr=65504.0))
assert e is None and torch.all(p == -65504)
e, p, _ = step_error(SGD, torch.float16, dict(lr=math.inf))
assert e is None and torch.all(torch.isinf(p))
e, p, _ = step_error(SGD, torch.float16, dict(lr=math.nan))
assert e is None and torch.all(torch.isnan(p))
e, p, _ = step_error(SGD, torch.float16, dict(lr=0.1, momentum=1e5))
assert e is None and torch.all(p == 0.89990234375)
e, p, _ = step_error(AdamW, torch.float32, dict(lr=1.0, weight_decay=1e40))
assert e is None and torch.all(p == -math.inf)
e, p, _ = step_error(Adam, torch.float32, dict(lr=0.1, eps=1e40))
assert e is None and torch.all(p == 1)
e, p, _ = step_error(Adam, torch.float16, dict(lr=1e5))
assert e is None and torch.all(p == -math.inf)
p = torch.nn.Parameter(torch.ones(32, dtype=torch.float16))
SGD([p], lr=1e5).step()
assert torch.all(p == 1)
e, _, _ = step_error(SGD, torch.float16, dict(lr=1e5), device="meta")
assert e is None
e, p, o = step_error(Adam, torch.float16, dict(lr=1e38), device="meta")
assert e is None and o.state[p]["step"].item() == 1

# non_contiguous_parameters_and_gradients: the result doesn't depend on the layout.
data = torch.randn(5, 7, dtype=torch.float64)
g = torch.randn(7, 5, dtype=torch.float64)
p1 = torch.nn.Parameter(data.clone().t())
p2 = torch.nn.Parameter(data.t().contiguous())
p1.grad, p2.grad = g.t().contiguous().t(), g.clone()
assert not p1.is_contiguous() and not p1.grad.is_contiguous()
for q in (p1, p2):
    SGD([q], lr=0.1, momentum=0.9, weight_decay=0.1).step()
assert torch.equal(p1, p2)

# options_mut_keeps_the_state_with_its_parameter: PyTorch keys the state by parameter, so even
# reordering the groups keeps it with the parameter (homura doesn't let the groups change).
a, b = param(1.0), param(2.0)
o = SGD([{"params": [a]}, {"params": [b]}], lr=0.1, momentum=0.9)
grad(a, 1.0)
grad(b, 10.0)
o.step()
o.param_groups[0], o.param_groups[1] = o.param_groups[1], o.param_groups[0]
o.step()
assert o.state[a]["momentum_buffer"].item() == 1.9 and o.state[b]["momentum_buffer"].item() == 19.0
a, b = param(1.0), param(2.0)
o = SGD([{"params": [a]}, {"params": [b]}], lr=0.1, momentum=0.9)
grad(a, 1.0)
grad(b, 10.0)
o.step()
o.param_groups[1]["lr"] = 0.01
o.param_groups[0]["momentum"] = 0.5
o.step()
assert o.state[a]["momentum_buffer"].item() == 1.5 and o.state[b]["momentum_buffer"].item() == 19.0
assert abs(a.item() - (1.0 - 0.1 - 0.15)) < 1e-15 and abs(b.item() - (2.0 - 1.0 - 0.19)) < 1e-15

# defaults_are_pytorchs
d = SGD([param(1.0)], lr=0.1).defaults
assert [d[k] for k in ("momentum", "dampening", "weight_decay", "nesterov", "maximize")] == [
    0, 0, 0, False, False]
d = Adam([param(1.0)], lr=0.1).defaults
assert (d["betas"], d["eps"], d["weight_decay"], d["amsgrad"], d["maximize"]) == (
    (0.9, 0.999), 1e-8, 0, False, False)
assert d["decoupled_weight_decay"] is False
d = AdamW([param(1.0)], lr=0.1).defaults
assert (d["weight_decay"], d["decoupled_weight_decay"]) == (1e-2, True)

# options_are_checked_with_pytorchs_messages (Python prints 1.0 and nan where Rust prints 1
# and NaN)
raises(lambda: SGD([param(1.0)], lr=-0.5), ValueError, "Invalid learning rate: -0.5")
raises(lambda: SGD([param(1.0)], lr=0.1, momentum=-0.5), ValueError, "Invalid momentum value: -0.5")
raises(lambda: SGD([param(1.0)], lr=0.1, weight_decay=-0.5), ValueError,
       "Invalid weight_decay value: -0.5")
raises(lambda: Adam([param(1.0)], lr=-0.5), ValueError, "Invalid learning rate: -0.5")
raises(lambda: Adam([param(1.0)], eps=-0.5), ValueError, "Invalid epsilon value: -0.5")
raises(lambda: Adam([param(1.0)], betas=(1.0, 0.9)), ValueError,
       "Invalid beta parameter at index 0: 1.0")
raises(lambda: Adam([param(1.0)], betas=(0.9, -0.1)), ValueError,
       "Invalid beta parameter at index 1: -0.1")
raises(lambda: Adam([param(1.0)], weight_decay=-0.5), ValueError, "Invalid weight_decay value: -0.5")
raises(lambda: AdamW([param(1.0)], weight_decay=-0.5), ValueError,
       "Invalid weight_decay value: -0.5")
raises(lambda: Adam([param(1.0)], lr=math.nan), ValueError, "Invalid learning rate: nan")
raises(lambda: AdamW([param(1.0)], betas=(math.nan, 0.9)), ValueError,
       "Invalid beta parameter at index 0: nan")
raises(lambda: Adam([param(1.0)], eps=math.nan), ValueError, "Invalid epsilon value: nan")
SGD([param(1.0)], lr=math.nan, momentum=5.0, dampening=-3.0)
nesterov = "Nesterov momentum requires a momentum and zero dampening"
raises(lambda: SGD([param(1.0)], lr=0.1, nesterov=True), ValueError, nesterov)
raises(lambda: SGD([param(1.0)], lr=0.1, nesterov=True, momentum=0.9, dampening=0.1), ValueError,
       nesterov)
SGD([param(1.0)], lr=0.1, nesterov=True, momentum=0.9)
# PyTorch checks the nesterov combination only in SGD(): a group added with dampening is
# accepted (homura's add_param_group panics).
o = SGD([param(1.0)], lr=0.1, nesterov=True, momentum=0.9)
o.add_param_group({"params": [param(1.0)], "dampening": 0.5})

# step_checks_every_group_before_updating: PyTorch doesn't check options set later.
p = param(1.0)
o = SGD([p], lr=0.1)
o.param_groups[0]["lr"] = -1.0
grad(p, 1.0)
o.step()
assert p.item() == 2.0

# parameters_are_checked
raises(lambda: SGD([], lr=0.1), ValueError, "optimizer got an empty parameter list")
a, b = param(1.0), param(2.0)
o = SGD([a], lr=0.1)
o.add_param_group({"params": []})
cross = "some parameters appear in more than one parameter group"
raises(lambda: o.add_param_group({"params": [b, a]}), ValueError, cross)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    twice = Adam([a, a], lr=0.1)
assert "duplicate parameters" in str(w[0].message)
c = param(1.0)
c.grad = torch.ones(1, dtype=torch.float64)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    SGD([c, c], lr=0.1).step()
assert abs(c.item() - 0.8) < 1e-15  # updated twice


def make_non_leaf(p):
    p.requires_grad_(False)
    x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    p.add_(x)
    assert not p.is_leaf and p.grad_fn is not None


c = param(1.0)
make_non_leaf(c)
raises(lambda: SGD([c], lr=0.1), ValueError, "can't optimize a non-leaf Tensor")
x = torch.ones(1, requires_grad=True)
y = x * 2
y.retain_grad()
SGD([y], lr=0.1)  # a non-leaf that retains its grad is accepted (homura has no such parameter)
# PyTorch checks leaves only when it adds a group: a step updates a parameter that became a
# non-leaf (homura's step panics).
d, e = param(1.0), param(1.0)
o = SGD([d, e], lr=0.1)
grad(d, 1.0)
grad(e, 1.0)
make_non_leaf(e)
o.step()
assert d.item() == 0.9
ip = torch.nn.Parameter(torch.tensor([1, 2]), requires_grad=False)
o = Adam([ip], lr=0.1)
o.step()
assert len(o.state) == 0 and ip.tolist() == [1, 2]

# sgd_momentum_buffer
p = param(1.0)
o = SGD([p], lr=0.1, momentum=0.9, dampening=0.5, weight_decay=0.1)
grad(p, 2.0)
o.step()
buf = o.state[p]["momentum_buffer"]
assert buf.item() == 2.0 + 0.1 and abs(p.item() - (1.0 - 0.1 * 2.1)) < 1e-15
p.grad.fill_(7.0)
assert buf.item() == 2.1
g = 7.0 + 0.1 * p.item()
o.step()
assert abs(buf.item() - (0.9 * 2.1 + 0.5 * g)) < 1e-15
p = param(1.0)
o = SGD([p], lr=0.1, momentum=0.9, weight_decay=0.1, maximize=True)
grad(p, 2.0)
o.step()
assert o.state[p]["momentum_buffer"].item() == -2.0 + 0.1
p = param(1.0)
o = SGD([p], lr=0.1, momentum=0.9, nesterov=True)
grad(p, 2.0)
o.step()
o.step()
assert abs(p.item() - (1.0 - 0.38 - 0.542)) < 1e-15
p = param(1.0)
o = SGD([p], lr=0.1)
grad(p, 2.0)
o.step()
assert len(o.state) == 0
o.param_groups[0]["momentum"] = 0.9
o.step()
assert o.state[p]["momentum_buffer"].item() == 2.0
o.param_groups[0]["momentum"] = 0.0
before = p.item()
o.step()
assert o.state[p]["momentum_buffer"].item() == 2.0 and abs(p.item() - (before - 0.2)) < 1e-15
p = param(1.0)
o = SGD([p], lr=0.1, momentum=0.9)
q = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
p.grad = q * 1.0
o.step()
buf = o.state[p]["momentum_buffer"]
assert buf.grad_fn is None and not buf.requires_grad and p.is_leaf and p.grad_fn is None

# none_and_zero_gradients_and_frozen_parameters
p, r = param(1.0), param(1.0)
o = SGD([p, r], lr=0.1, weight_decay=0.5)
grad(p, 0.0)
o.step()
assert p.item() == 1.0 - 0.1 * 0.5 and r.item() == 1.0
p.requires_grad_(False)
o.step()
assert p.item() < 0.95 - 0.04
o.zero_grad()
assert p.grad is None and r.grad is None
frozen = p.item()
o.step()
assert p.item() == frozen
p, r = param(1.0), param(1.0)
o = AdamW([p, r], lr=0.1, weight_decay=0.5)
grad(p, 0.0)
o.step()
assert p.item() == 1.0 - 0.1 * 0.5 and r.item() == 1.0 and r not in o.state
assert o.state[p]["step"].item() == 1.0
p = param(1.0)
o = Adam([p], lr=0.1, weight_decay=0.5)
grad(p, 0.0)
o.step()
assert abs(p.item() - 0.9) < 1e-7

# adam_state
for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
    p = torch.nn.Parameter(torch.ones(2, dtype=dtype))
    o = Adam([p], lr=0.1)
    p.grad = torch.ones(2, dtype=dtype)
    o.step()
    s = o.state[p]
    assert s["step"].dtype == torch.float32 and s["step"].device.type == "cpu"
    assert s["step"].shape == () and set(s) == {"step", "exp_avg", "exp_avg_sq"}
    assert s["exp_avg"].dtype == dtype and s["exp_avg_sq"].dtype == dtype
p = param(1.0)
o = Adam([p], lr=0.1, amsgrad=True)
grad(p, 1.0)
o.step()
assert "max_exp_avg_sq" in o.state[p]
# amsgrad turned on later: KeyError when the group's state is collected, after the updates of
# earlier groups (here the failing group is the first, so nothing changed).
p, r = param(1.0), param(1.0)
o = Adam([p], lr=0.1)
grad(p, 1.0)
o.step()
o.add_param_group({"params": [r], "amsgrad": True})
grad(r, 1.0)
o.step()
o.param_groups[0]["amsgrad"] = True
raises(lambda: o.step(), KeyError, "max_exp_avg_sq")
o.param_groups[0]["amsgrad"] = False
o.step()
assert o.state[p]["step"].item() == 3.0  # the failed step didn't count

# meta_parameters
p = torch.nn.Parameter(torch.ones(3, device="meta"))
p.grad = torch.ones(3, device="meta")
o = Adam([p], lr=0.1, amsgrad=True, weight_decay=0.1)
o.step()
s = o.state[p]
assert s["step"].device.type == "cpu" and s["step"].item() == 1.0
assert s["exp_avg"].device.type == "meta" and s["exp_avg_sq"].device.type == "meta"
o = SGD([p], lr=0.1, momentum=0.9, nesterov=True)
o.step()
o.step()
assert o.state[p]["momentum_buffer"].device.type == "meta" and p.is_leaf

# step_records_nothing_and_bumps_versions
p = param(1.0, 2.0)
o = AdamW([p], lr=0.1, amsgrad=True)
saved = p * p
(p * 2.0).sum().backward()
with torch.enable_grad():
    o.step()
for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
    assert not o.state[p][k].requires_grad and o.state[p][k].grad_fn is None
assert p.is_leaf and p.requires_grad
raises(lambda: saved.sum().backward(), RuntimeError, "has been modified by an inplace operation")


# shared_parameters_are_updated_once
class Tied(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.head = w
        self.tail = w


w = param(1.0, 2.0)
model = Tied(w)
assert len(list(model.parameters())) == 1
o = SGD(model.parameters(), lr=0.1)
x = torch.ones(2, dtype=torch.float64)
((model.head * x).sum() + (model.tail * x).sum()).backward()
assert w.grad.tolist() == [2.0, 2.0]
o.step()
assert model.tail.tolist() == [0.8, 1.8]

# moving_the_model_after_the_state_was_created
model = torch.nn.Linear(2, 2)
o = Adam(model.parameters(), lr=0.1)
model.to(torch.float64)
model(torch.ones(1, 2, dtype=torch.float64)).sum().backward()
o.step()
assert o.state[model.weight]["exp_avg"].dtype == torch.float64
model.to(torch.float32)
model(torch.ones(1, 2)).sum().backward()
raises(lambda: o.step(), RuntimeError, "expected dtype double for `end` but got dtype float")
p, r = param(1.0), param(1.0)
o = SGD([p, r], lr=0.1, momentum=0.9)
grad(p, 1.0)
grad(r, 1.0)
o.step()
r.data = r.data.float()
r.grad = torch.ones(1)
# PyTorch's SGD goes on with the F64 buffer of the F32 parameter (homura's step panics).
o.step()
assert o.state[r]["momentum_buffer"].dtype == torch.float64 and r.dtype == torch.float32

# optim_default_dtype: adam_step_follows_the_default_dtype
torch.set_default_dtype(torch.float64)
p = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
o = Adam([p], lr=0.1)
p.grad = torch.ones(2, dtype=torch.float32)
o.step()
assert o.state[p]["step"].dtype == torch.float64 and o.state[p]["step"].device.type == "cpu"
torch.set_default_dtype(torch.bfloat16)
q = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
o.add_param_group({"params": [q]})
q.grad = torch.ones(2, dtype=torch.float32)
o.step()
assert o.state[q]["step"].dtype == torch.float32 and o.state[p]["step"].item() == 2.0
torch.set_default_dtype(torch.float32)

print("check_m2_optim: all reference behavior asserted")
