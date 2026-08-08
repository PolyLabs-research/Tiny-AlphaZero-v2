"""Isolate compute from the dataloader: time train_step on a synthetic on-device batch."""
import argparse, time, torch
from tinyalphazero.config import load_config
from tinyalphazero.model import ChessNet
from tinyalphazero.train import train_step

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--compile", action="store_true")
args = p.parse_args()

cfg = load_config("configs/phase1_c.yaml")
dev = torch.device("cuda")
net = ChessNet(cfg.model).to(dev)
if args.compile:
    net = torch.compile(net)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
B = args.batch
g = torch.Generator().manual_seed(0)
def make_batch():
    return {
        "tokens": torch.randint(0, 43, (B, 72), generator=g),
        "from_sq": torch.randint(0, 64, (B,), generator=g),
        "to_sq": torch.randint(0, 64, (B,), generator=g),
        "promo": torch.zeros(B, dtype=torch.long),
        "legals": torch.randint(0, 256, (B, 512), generator=g, dtype=torch.uint8),
        "outcome": torch.randint(-1, 2, (B,), generator=g),
        "value_weight": torch.full((B,), 0.25),
    }


for _ in range(10):  # warmup (compile needs it)
    train_step(make_batch(), net, opt, dev, cfg.train, use_bf16=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(30):
    train_step(make_batch(), net, opt, dev, cfg.train, use_bf16=True)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / 30 * 1000
print(f"batch={B} compile={args.compile}  {ms:.0f} ms/step  ({B/ms*1000:.0f} samples/s)")