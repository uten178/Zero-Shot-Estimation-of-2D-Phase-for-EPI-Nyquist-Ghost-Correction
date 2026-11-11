"""
STEP 1  -  physics-only screen for the starting point of the phase estimate (manuscript 3.3).

A GRE reference scan has no RO+/RO- alternation, so it cannot tell us the polarity phase.
We therefore try a coarse grid of (p0, px) values (py = 0) and keep the one that predicts
the held-out k-space lines best.  No network is involved here.

Output: results/baseline.json   ->  [p0, px, py] used as the starting point of step 2
"""
import json
import os

import numpy as np
import torch

import config
import d2_tools as t

device = t.pick_device()
os.makedirs(config.RESULT_DIR, exist_ok=True)
D = t.prepare(config, device)
slices, X, Y = D["slices"], D["X"], D["Y"]
print(f"{len(slices)} slice(s) loaded, device = {device}")

# use a few slices spread over the whole stack; each uses the training pattern assigned to it
chosen = t.representative_indices(len(slices), config.SCREEN_NSLICES)

best = None
table = []
for p0 in config.SCREEN_P0:
    for px in config.SCREEN_PX:
        for py in config.SCREEN_PY:
            theta = t.theta_map(torch.tensor([p0, px, py], dtype=torch.float32, device=device), X, Y)
            losses = []
            with torch.no_grad():
                for i in chosen:
                    s = slices[i]
                    pattern = D["patterns"][D["assign"][i]]
                    loss, _ = t.heldout_loss(s["csm"], theta, s["ksp"], pattern, config.PCSENSE_LAMBDA, config.SCREEN_CG)
                    losses.append(loss.item())
            total = float(np.mean(losses))
            table.append((p0, px, py, total))
            if best is None or total < best[3]:
                best = (p0, px, py, total)
    print(f"  p0 = {p0:+.2f} done, best so far: p0={best[0]:+.2f} px={best[1]:+.0f}  loss={best[3]:.4f}")

baseline = [float(best[0]), float(best[1]), float(best[2])]
print("selected starting point [p0, px, py] =", baseline)
with open(os.path.join(config.RESULT_DIR, "baseline.json"), "w") as f:
    json.dump(dict(baseline=baseline, screen=table), f, indent=1)
