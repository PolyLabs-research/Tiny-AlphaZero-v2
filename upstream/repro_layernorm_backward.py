# Minimal reproducer: LayerNorm weight/bias gradients on ROCm vs an fp64 CPU reference.
# Needs only torch. Prints relative errors; any value > 1e-3 is wrong.
import sys

import torch

torch.manual_seed(0)


def rel(a, b):
    return ((a.double().cpu() - b).norm() / b.norm()).item()


def run(n, d, dtype=torch.float32):
    x = torch.randn(n, d, dtype=torch.float64)
    g = torch.randn(n, d, dtype=torch.float64)
    ref = torch.nn.LayerNorm(d).double()
    xr = x.clone().requires_grad_(True)
    ref(xr).backward(g)
    ln = torch.nn.LayerNorm(d).to("cuda", dtype)
    xc = x.to("cuda", dtype).requires_grad_(True)
    ln(xc).backward(g.to("cuda", dtype))
    # The same gradients from plain ops on the same device: no fp64, no CPU involved.
    xhat = torch.nn.functional.layer_norm(xc.detach().float(), (d,))
    gc = g.to("cuda", torch.float32)
    w_plain, b_plain = (gc * xhat).sum(0), gc.sum(0)
    return (
        rel(ln.weight.grad, ref.weight.grad),
        rel(ln.bias.grad, ref.bias.grad),
        rel(xc.grad, xr.grad),
        ((ln.weight.grad.float() - w_plain).norm() / w_plain.norm()).item(),
        ((ln.bias.grad.float() - b_plain).norm() / b_plain.norm()).item(),
    )


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    print(torch.__version__, getattr(torch.version, "hip", None), torch.cuda.get_device_name(0))
    print(
        f"width {d}: rows | weight.grad | bias.grad | input.grad (vs fp64 CPU) | weight.grad, bias.grad (vs plain ops on device)"
    )
    for n in (64, 128, 192, 256, 1024, 4096, 8192, 16384, 32768, 65536, 73728, 131072):
        r = run(n, d)
        flag = "  <-- WRONG" if max(r[0], r[1]) > 1e-3 else ""
        print(f"  {n:7d} | {r[0]:.2e} | {r[1]:.2e} | {r[2]:.2e} | {r[3]:.2e}, {r[4]:.2e}{flag}")
