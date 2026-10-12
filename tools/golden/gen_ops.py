"""Generates tests/golden/ops.safetensors: inputs and PyTorch outputs for the M0 operators.

Each case is stored as "<case>/<name>" entries. The Rust tests in tests/golden_ops.rs run the same
operation on the inputs and compare with the outputs, so case names and arguments must stay in
sync with that file.

Usage: python tools/golden/gen_ops.py (needs torch and safetensors).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "ops.safetensors"

reference.check()
torch.manual_seed(0)
data = {}


def put(case, **tensors):
    for name, t in tensors.items():
        key = f"{case}/{name}"
        assert key not in data, key
        data[key] = t.detach().contiguous().clone()


def randn(*shape, dtype=torch.float32):
    return torch.randn(*shape, dtype=torch.float64).to(dtype)


def rand(*shape, dtype=torch.float32):
    return torch.rand(*shape, dtype=torch.float64).to(dtype)


nan, inf = float("nan"), float("inf")

# ---------------------------------------------------------------- unary
x = randn(3, 4)
xp = rand(3, 4) * 3 + 0.1
for name in ["neg", "abs", "exp", "sin", "cos", "tanh", "sigmoid", "relu", "reciprocal"]:
    put(f"unary.{name}", x=x, out=getattr(torch, name)(x))
for name in ["log", "sqrt", "rsqrt"]:
    put(f"unary.{name}", x=xp, out=getattr(torch, name)(xp))
put("unary.silu", x=x, out=F.silu(x))
put("unary.gelu", x=x, out=F.gelu(x))
put("unary.gelu_tanh", x=x, out=F.gelu(x, approximate="tanh"))

wide = torch.linspace(-12, 12, 49)
put("unary_wide.gelu", x=wide, out=F.gelu(wide))
put("unary_wide.gelu_tanh", x=wide, out=F.gelu(wide, approximate="tanh"))
put("unary_wide.silu", x=wide, out=F.silu(wide))
put("unary_wide.sigmoid", x=wide, out=torch.sigmoid(wide))
put("unary_wide.tanh", x=wide, out=torch.tanh(wide))
put("unary_wide.exp", x=wide, out=torch.exp(wide))

special = torch.tensor([nan, inf, -inf, 0.0, -0.0, 1.5, -2.5, 4.0])
for name in ["neg", "abs", "exp", "log", "sqrt", "tanh", "sigmoid", "relu"]:
    put(f"unary_special.{name}", x=special, out=getattr(torch, name)(special))
put("unary_special.gelu", x=special, out=F.gelu(special))
put("unary_special.silu", x=special, out=F.silu(special))

for dt, tag in [(torch.float16, "f16"), (torch.bfloat16, "bf16"), (torch.float64, "f64")]:
    xd, xpd = x.to(dt), xp.to(dt)
    for name in ["exp", "tanh", "sigmoid", "neg", "relu"]:
        put(f"unary_{tag}.{name}", x=xd, out=getattr(torch, name)(xd))
    put(f"unary_{tag}.silu", x=xd, out=F.silu(xd))
    put(f"unary_{tag}.gelu", x=xd, out=F.gelu(xd))
    put(f"unary_{tag}.sqrt", x=xpd, out=torch.sqrt(xpd))
    put(f"unary_{tag}.log", x=xpd, out=torch.log(xpd))

xi = torch.randint(-5, 6, (3, 4))
put("unary_int.exp", x=xi, out=torch.exp(xi))
put("unary_int.sqrt", x=xi.abs(), out=torch.sqrt(xi.abs()))
put("unary_int.abs", x=xi, out=torch.abs(xi))
put("unary_int.neg", x=xi, out=torch.neg(xi))
put("unary_int.relu", x=xi, out=torch.relu(xi))
put("unary_int.pow3", x=xi, out=xi.pow(3))
put("unary_int.clamp_float", x=xi, out=xi.clamp(-1.5, 2.5))
put("unary_int.clamp_int", x=xi, out=xi.clamp(-1, 2))

put("pow.scalar2", x=x, out=x.pow(2))
put("pow.scalar3", x=x, out=x.pow(3))
put("pow.half", x=xp, out=xp.pow(0.5))
put("pow.neg", x=xp, out=xp.pow(-1.5))
put("clamp.both", x=x, out=x.clamp(-0.5, 0.5))
put("clamp.min", x=x, out=x.clamp_min(0.2))
put("clamp.max", x=x, out=x.clamp_max(-0.1))
put("clamp.special", x=special, out=special.clamp(-1.0, 1.0))
put("clamp.inverted", x=x, out=x.clamp(0.5, -0.5))
lnot = torch.tensor([0.0, 1.0, -2.0, nan, -0.0])
put("logical_not.float", x=lnot, out=torch.logical_not(lnot))

# ---------------------------------------------------------------- binary
a, b = randn(3, 1, 4), randn(2, 4)
put("binary.add", a=a, b=b, out=a + b)
put("binary.sub", a=a, b=b, out=a - b)
put("binary.mul", a=a, b=b, out=a * b)
put("binary.div", a=a, b=b, out=a / b)
put("binary.maximum", a=a, b=b, out=torch.maximum(a, b))
put("binary.minimum", a=a, b=b, out=torch.minimum(a, b))
pa, pb = rand(3, 1, 4) + 0.5, randn(2, 4)
put("binary.pow", a=pa, b=pb, out=torch.pow(pa, pb))
put("binary.rsub", x=x, out=1.0 - x)
put("binary.rdiv", x=xp, out=2.0 / xp)
put("binary.add_scalar", x=x, out=x + 2.5)

ca = torch.tensor([1.0, nan, 3.0, 2.0, -inf, nan])
cb = torch.tensor([1.0, 1.0, nan, 3.0, -inf, nan])
put("nan.maximum", a=ca, b=cb, out=torch.maximum(ca, cb))
put("nan.minimum", a=ca, b=cb, out=torch.minimum(ca, cb))
for name in ["gt", "ge", "lt", "le", "eq", "ne"]:
    put(f"compare.{name}", a=ca, b=cb, out=getattr(torch, name)(ca, cb))
ci = torch.tensor([1, 2, 3, 4, -5, 0], dtype=torch.int32)
put("compare.mixed_lt", a=ci, b=cb, out=ci < cb)
put("compare.scalar_ge", a=ci, out=ci >= 2.5)

la = torch.tensor([0.0, 1.0, nan, -2.0, 0.0])
lb = torch.tensor([1, 0, 0, 3, 0], dtype=torch.int64)
put("logical.and", a=la, b=lb, out=torch.logical_and(la, lb))
put("logical.or", a=la, b=lb, out=torch.logical_or(la, lb))

ia = torch.randint(-10, 10, (4,), dtype=torch.int32)
fb = randn(4, dtype=torch.float16)
put("promote.i32_f16", a=ia, b=fb, out=ia + fb)
ua = torch.randint(0, 256, (5,), dtype=torch.uint8)
sb = torch.randint(-128, 128, (5,), dtype=torch.int8)
put("promote.u8_i8", a=ua, b=sb, out=ua + sb)
li = torch.randint(-10, 10, (4,))
put("promote.i64_float_scalar", a=li, out=li * 2.5)
fh = randn(4, dtype=torch.float16)
put("promote.f16_float_scalar", a=fh, out=fh * 2.5)
da = torch.randint(-10, 10, (6,))
db = torch.tensor([3, -4, 7, 2, -3, 5])
put("promote.int_div", a=da, b=db, out=da / db)
ba, bb = torch.tensor([True, False, True]), torch.tensor([True, False, False])
put("promote.bool_add", a=ba, b=bb, out=ba + bb)
put("promote.bool_mul", a=ba, b=bb, out=ba * bb)
put("promote.bool_int_scalar", a=ba, out=ba + 1)
f32v, f64z = randn(3), torch.tensor(0.25, dtype=torch.float64)
put("promote.f32_f64_zero_dim", a=f32v, b=f64z, out=f32v + f64z)
put("promote.i64_f64_zero_dim", a=li, b=f64z, out=li + f64z)
put("promote.u8_minus_one", a=ua, out=ua - 1)
o1 = torch.tensor([127, -128, 100], dtype=torch.int8)
o2 = torch.tensor([1, -1, 100], dtype=torch.int8)
put("promote.i8_overflow", a=o1, b=o2, out=o1 + o2)
put("promote.i8_mul_overflow", a=o1, b=o2, out=o1 * o2)
z1, z0 = torch.tensor([1.0, -1.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])
put("binary.div_zero", a=z1, b=z0, out=z1 / z0)
for dt, tag in [(torch.float16, "f16"), (torch.bfloat16, "bf16")]:
    ad, bd = a.to(dt), b.to(dt)
    put(f"binary_{tag}.add", a=ad, b=bd, out=ad + bd)
    put(f"binary_{tag}.mul", a=ad, b=bd, out=ad * bd)
    put(f"binary_{tag}.div", a=ad, b=bd, out=ad / bd)

cond = randn(3, 4) > 0
wx, wy = randn(3, 4), randn(4)
put("where.tensors", cond=cond, x=wx, y=wy, out=torch.where(cond, wx, wy))
put("where.scalar", cond=cond, x=wx, out=torch.where(cond, wx, 0.5))
wi = torch.randint(-5, 5, (3, 4))
put("where.promote", cond=cond, x=wi, y=wy, out=torch.where(cond, wi, wy))
mask = randn(4) > 0
put("masked_fill.broadcast", x=wx, mask=mask, out=wx.masked_fill(mask, -1e9))
put("masked_fill.int", x=wi, mask=cond, out=wi.masked_fill(cond, 7))

# ---------------------------------------------------------------- reductions
r = randn(3, 4, 5)
put("reduce.sum", x=r, out=r.sum())
put("reduce.sum_dims_keepdim", x=r, out=r.sum((0, 2), keepdim=True))
put("reduce.sum_last", x=r, out=r.sum(-1))
put("reduce.sum_negative_dims", x=r, out=r.sum((-1, -3)))
put("reduce.mean", x=r, out=r.mean())
put("reduce.mean_dim_keepdim", x=r, out=r.mean(1, keepdim=True))
rp = rand(3, 4, 5) + 0.5
put("reduce.prod", x=rp, out=rp.prod())
put("reduce.prod_dim", x=rp, out=rp.prod(1))
put("reduce.max", x=r, out=r.max())
put("reduce.min", x=r, out=r.min())
v, i = r.max(dim=1)
put("reduce.max_dim", x=r, values=v, indices=i)
v, i = r.min(dim=-1, keepdim=True)
put("reduce.min_dim_keepdim", x=r, values=v, indices=i)
put("reduce.argmax", x=r, out=r.argmax())
put("reduce.argmin_dim", x=r, out=r.argmin(dim=2))
ties = torch.tensor([[1, 3, 3, 2], [5, 5, 5, 0], [-1, -2, -2, -1]])
put("ties.argmax_dim", x=ties, out=ties.argmax(dim=1))
put("ties.argmin_dim", x=ties, out=ties.argmin(dim=1))
v, i = ties.max(dim=1)
put("ties.max_dim", x=ties, values=v, indices=i)
v, i = ties.min(dim=0)
put("ties.min_dim", x=ties, values=v, indices=i)
put("ties.argmax", x=ties, out=ties.argmax())
tn = torch.tensor([[1.0, nan, 3.0], [nan, nan, 0.0], [2.0, 1.0, 0.0]])
v, i = tn.max(dim=1)
put("nan.max_dim", x=tn, values=v, indices=i)
v, i = tn.min(dim=0)
put("nan.min_dim", x=tn, values=v, indices=i)
put("nan.argmax_dim", x=tn, out=tn.argmax(dim=1))
put("nan.max", x=tn, out=tn.max())
put("nan.sum_dim", x=tn, out=tn.sum(dim=0))
put("var.default", x=r, out=r.var())
put("var.dim_c0_keepdim", x=r, out=r.var(dim=1, correction=0, keepdim=True))
put("std.dims", x=r, out=r.std(dim=(0, 2)))
put("std.dim_c2", x=r, out=r.std(dim=2, correction=2))
one = torch.tensor([2.0])
put("var.single", x=one, out=one.var())
ab = randn(3, 4) > 0
put("bool.all", x=ab, out=ab.all())
put("bool.any_dim", x=ab, out=ab.any(dim=1))
put("bool.all_dim_keepdim", x=ab, out=ab.all(dim=0, keepdim=True))
put("bool.sum", x=ab, out=ab.sum())
fa = torch.tensor([1.0, nan, 0.0])
put("float.all", x=fa, out=fa.all())
put("float.any", x=fa, out=fa.any())
ri = torch.randint(-100, 100, (4, 5), dtype=torch.int32)
put("int.sum", x=ri, out=ri.sum())
put("int.sum_dim", x=ri, out=ri.sum(dim=0))
put("int.prod_dim", x=ri, out=ri.prod(dim=1))
put("int.max_dim", x=ri, values=ri.max(dim=1).values, indices=ri.max(dim=1).indices)
for dt, tag in [(torch.float16, "f16"), (torch.bfloat16, "bf16")]:
    rd = r.to(dt)
    put(f"reduce_{tag}.sum_dim", x=rd, out=rd.sum(dim=-1))
    put(f"reduce_{tag}.mean", x=rd, out=rd.mean())
    put(f"reduce_{tag}.var_dim", x=rd, out=rd.var(dim=1))

# ---------------------------------------------------------------- matmul
for case, (sa, sb) in {
    "dot": ((4,), (4,)),
    "mm": ((3, 4), (4, 5)),
    "vm": ((4,), (4, 5)),
    "mv": ((3, 4), (4,)),
    "batched": ((2, 1, 3, 4), (5, 4, 2)),
    "batched_vec": ((2, 3, 4), (4,)),
    "vec_batched": ((4,), (2, 4, 3)),
}.items():
    ma, mb = randn(*sa), randn(*sb)
    put(f"matmul.{case}", a=ma, b=mb, out=torch.matmul(ma, mb))
mia, mib = torch.randint(-5, 5, (3, 4)), torch.randint(-5, 5, (4, 2))
put("matmul.int", a=mia, b=mib, out=mia @ mib)
mha, mhb = randn(3, 8, dtype=torch.float16), randn(8, 2, dtype=torch.float16)
put("matmul.f16", a=mha, b=mhb, out=mha @ mhb)
mta, mtb = randn(4, 3), randn(5, 4)
put("matmul.transposed", a=mta, b=mtb, out=mta.t() @ mtb.t())

# ---------------------------------------------------------------- composite
s = randn(2, 3, 5) * 3
put("softmax.last", x=s, out=s.softmax(-1))
put("softmax.first", x=s, out=s.softmax(0))
put("log_softmax.mid", x=s, out=s.log_softmax(1))
sh = s.half()
put("softmax.f16", x=sh, out=sh.softmax(-1))
sinf = torch.tensor([[0.0, -inf, 1.0], [-inf, -inf, -inf], [1000.0, 0.0, -1000.0]])
put("softmax.inf", x=sinf, out=sinf.softmax(-1))
put("log_softmax.inf", x=sinf, out=sinf.log_softmax(-1))

inp, m1, m2 = randn(3, 5), randn(3, 4), randn(4, 5)
put("addmm.full", input=inp, m1=m1, m2=m2, out=torch.addmm(inp, m1, m2, beta=0.5, alpha=2.0))
row = randn(5)
put("addmm.bias", input=row, m1=m1, m2=m2, out=torch.addmm(row, m1, m2))
nan_row = torch.full((5,), nan)
put("addmm.beta0", input=nan_row, m1=m1, m2=m2, out=torch.addmm(nan_row, m1, m2, beta=0))
mv_in, mv_m, mv_v = randn(3), randn(3, 4), randn(4)
put("addmv", input=mv_in, mat=mv_m, vec=mv_v, out=torch.addmv(mv_in, mv_m, mv_v, beta=2, alpha=0.5))
ar_in, v1, v2 = randn(3, 4), randn(3), randn(4)
put("addr", input=ar_in, v1=v1, v2=v2, out=torch.addr(ar_in, v1, v2, beta=1, alpha=-1))
bi, b1, b2 = randn(2, 3, 5), randn(2, 3, 4), randn(2, 4, 5)
put("baddbmm", input=bi, b1=b1, b2=b2, out=torch.baddbmm(bi, b1, b2, beta=0.5, alpha=1.5))
abi = randn(3, 5)
put("addbmm", input=abi, b1=b1, b2=b2, out=torch.addbmm(abi, b1, b2))
ci_, t1, t2 = randn(3, 4), randn(3, 4), randn(4)
put("addcmul", input=ci_, t1=t1, t2=t2, out=torch.addcmul(ci_, t1, t2, value=0.5))
t2p = rand(3, 4) + 0.5
put("addcdiv", input=ci_, t1=t1, t2=t2p, out=torch.addcdiv(ci_, t1, t2p, value=-2))
ls, le = randn(3, 4), randn(3, 4)
put("lerp.low", start=ls, end=le, out=torch.lerp(ls, le, 0.3))
put("lerp.high", start=ls, end=le, out=torch.lerp(ls, le, 0.7))
lw = rand(3, 4) * 1.4 - 0.2
put("lerp.tensor", start=ls, end=le, weight=lw, out=torch.lerp(ls, le, lw))

# ---------------------------------------------------------------- shape ops
c1, c2, c3 = randn(2, 3), randn(2, 1), randn(2, 2, dtype=torch.float64)
put("concat.promote", a=c1, b=c2, c=c3, out=torch.cat([c1, c2, c3], dim=1))
s1, s2 = randn(2, 3), randn(2, 3)
put("stack.last", a=s1, b=s2, out=torch.stack([s1, s2], dim=-1))
put("stack.first", a=s1, b=s2, out=torch.stack([s1, s2], dim=0))
sp = randn(3, 7)
for k, t in enumerate(sp.split(3, dim=1)):
    put("split.size3", **({"x": sp} if k == 0 else {}), **{f"out{k}": t})
for k, t in enumerate(sp.split([2, 4, 1], dim=1)):
    put("split_with_sizes", **({"x": sp} if k == 0 else {}), **{f"out{k}": t})
for k, t in enumerate(sp.chunk(3, dim=1)):
    put("chunk.three", **{f"out{k}": t})
tr = randn(2, 4, 5)
put("triu.batched", x=tr, out=tr.triu(1))
put("tril.batched", x=tr, out=tr.tril(-1))
put("triu.square", x=tr[0], out=tr[0].triu())
rep = randn(2, 3)
put("repeat.extra_dims", x=rep, out=rep.repeat(2, 1, 3))
fl = randn(2, 3, 4)
put("flip.dims", x=fl, out=fl.flip((0, 2)))

# ---------------------------------------------------------------- views and indexing
v3 = randn(3, 4, 5)
put("index.slice", x=v3, out=v3[:, 1:3])
put("index.ellipsis_last", x=v3, out=v3[..., -1])
put("index.newaxis", x=v3, out=v3[0, None])
put("index.step", x=v3, out=v3[:, ::2])
put("index.negative_start", x=v3, out=v3[-2:])
put("index.mixed", x=v3, out=v3[1:, -3:-1, ::3])
put("index.clamped", x=v3, out=v3[:, -10:10, 2:100])
idx = torch.tensor([2, -1, 0])
put("index.tensor", x=v3, idx=idx, out=v3[idx])
idx2 = torch.tensor([[0, 3], [1, 1]])
put("index.tensor_2d", x=v3, idx=idx2, out=v3[:, idx2])
put("index.tensor_after_select", x=v3, idx=idx2, out=v3[1, :, idx2])
put("index_select", x=v3, idx=torch.tensor([3, 0, 0]), out=v3.index_select(1, torch.tensor([3, 0, 0])))
put("view.permute", x=v3, out=v3.permute(2, 0, 1))
put("view.transpose", x=v3, out=v3.transpose(0, 2))
put("view.narrow", x=v3, out=v3.narrow(1, 1, 2))
put("view.select", x=v3, out=v3.select(2, -1))
put("view.flatten", x=v3, out=v3.flatten(1, 2))
put("view.reshape_copy", x=v3, out=v3.transpose(0, 1).reshape(4, 15))
put("view.expand", x=v3[:, :1], out=v3[:, :1].expand(-1, 4, 5))

# ---------------------------------------------------------------- creation and casting
put("create.arange_int", out=torch.arange(10))
put("create.arange_float_step", out=torch.arange(0, 1, 0.1))
put("create.arange_negative_step", out=torch.arange(5, 0, -2))
put("create.arange_float_end", out=torch.arange(0.0, 1.0, 0.3))
put("create.linspace", out=torch.linspace(0, 1, 7))
put("create.linspace_neg", out=torch.linspace(-3, 10, 6))
put("create.linspace_one", out=torch.linspace(2, 5, 1))
put("create.linspace_empty", out=torch.linspace(2, 5, 0))
put("create.eye", out=torch.eye(3))
put("create.full_int", out=torch.full((2, 3), 7))
put("create.full_float", out=torch.full((2,), 2.5))
cf = torch.tensor([-2.7, -0.5, 0.0, 0.5, 1.5, 2.5, 100.9])
put("cast.f32_i32", x=cf, out=cf.to(torch.int32))
put("cast.f32_i64", x=cf, out=cf.to(torch.int64))
put("cast.f32_bool", x=cf, out=cf.to(torch.bool))
put("cast.f32_f16", x=cf, out=cf.to(torch.float16))
put("cast.f32_bf16", x=cf, out=cf.to(torch.bfloat16))
h = torch.tensor([1 + 2**-11, 1 + 3 * 2**-11, 65519.0, 65520.0, 1e-8, 6e-8, -1e5])
put("cast.f16_rounding", x=h, out=h.to(torch.float16))
bh = torch.tensor([1 + 2**-8, 1 + 3 * 2**-8, 3.0e38, 1e-40, -7.77])
put("cast.bf16_rounding", x=bh, out=bh.to(torch.bfloat16))
ci64 = torch.tensor([2**24 + 1, -3, 0])
put("cast.i64_f32", x=ci64, out=ci64.float())
put("cast.bool_f32", x=ba, out=ba.float())
put("cast.f16_f32", x=fh, out=fh.float())

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(data, str(OUT))
print(f"wrote {len(data)} tensors to {OUT} ({OUT.stat().st_size} bytes)")
