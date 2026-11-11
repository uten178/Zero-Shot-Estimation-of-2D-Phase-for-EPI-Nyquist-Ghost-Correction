"""
STEP 2  -  weight-shared zero-shot estimation of the 2D linear phase  [p0, px, py]  per slice
           (manuscript Section 3.4 and Figure 1).

  * ONE network (JointPCNet) is trained on all slices together (shared weights);
    it returns three numbers per slice = shared global phase + slice-specific change.
  * the loss uses only the measured data: reconstruct (PC-SENSE) from the fit lines of both
    polarities, predict the held-out lines, compare (0.7 * L2 + 0.3 * L1), plus three weak
    terms (ghost energy, smoothness across slices, polarity swap).
  * four fixed split patterns are used for training (slice number mod 4), one separate pattern
    for validation (checkpoint choice / early stopping) and one untouched pattern for the test.
  * after training, every slice gets four estimates (one per training pattern):  median = estimate,
    spread (MAD) = a stability check.
  * a short per-slice polish follows; it is kept only if the validation loss improves.

Output (folder results/):  theta_zs.csv, theta_zs.npz, training_log.csv, zs_model.pt
"""
import copy
import csv
import json
import math
import os
import time

import numpy as np
import torch

import config
import d2_tools as t

torch.manual_seed(config.SEED)
rng = np.random.default_rng(config.SEED)
device = t.pick_device()
D = t.prepare(config, device)
slices, X, Y, patterns, assign = D["slices"], D["X"], D["Y"], D["patterns"], D["assign"]
n_slices, VAL, TEST = len(slices), D["val_id"], D["test_id"]
lam = config.PCSENSE_LAMBDA
numbers = [s["number"] for s in slices]
basin = json.load(open(os.path.join(config.RESULT_DIR, "baseline.json")))["baseline"]
print(f"{n_slices} slices | device {device} | starting point {basin}")


def slice_loss(i, params, pattern_id, n_cg, with_ghost=True):
    """Physics loss (+ ghost term) of slice i for the three numbers `params` and one split pattern."""
    s = slices[i]
    theta = t.theta_map(params, X, Y)
    return t.heldout_loss(s["csm"], theta, s["ksp"], patterns[pattern_id], lam, n_cg,
                          s["mask"] if with_ghost else None, s["ghost"] if with_ghost else None)


net = t.JointPCNet(2 * D["NC"] + 2, basin, config).to(device)
opt = torch.optim.AdamW(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.EPOCHS)
val_slices = t.representative_indices(n_slices, config.VAL_N_SLICES)


def network_theta(ids, pattern_ids):
    xp, xm = t.batch_inputs(slices, ids, patterns, pattern_ids)
    return net(xp, xm)


def validation_loss():
    """Mean physics loss on the VALIDATION pattern (never used for gradients)."""
    net.eval()
    with torch.no_grad():
        theta = network_theta(val_slices, [VAL] * len(val_slices))
        vals = [slice_loss(int(i), theta[j], VAL, config.VAL_CG_ITERS)[0].item() for j, i in enumerate(val_slices)]
    return float(np.mean(vals))


# ---------------------------------------------------------------------------------- training
best_val, best_state, bad_checks, log = np.inf, copy.deepcopy(net.state_dict()), 0, []
step = 0
t0 = time.time()
for epoch in range(1, config.EPOCHS + 1):
    net.train()
    n_use = min(n_slices, config.SLICES_PER_EPOCH)
    chosen = rng.permutation(n_slices)[:n_use]
    phys_list = []
    for start in range(0, n_use, config.BATCH_SLICES):
        batch = chosen[start:start + config.BATCH_SLICES]
        pattern_ids = assign[batch]
        opt.zero_grad()
        theta = network_theta(batch, pattern_ids)
        phys, ghost = [], []
        for j, i in enumerate(batch):
            p, g = slice_loss(int(i), theta[j], int(pattern_ids[j]), config.TRAIN_CG_ITERS)
            phys.append(p)
            ghost.append(g)
        l_phys, l_ghost = torch.stack(phys).mean(), torch.stack(ghost).mean()

        l_coh = torch.zeros((), device=device)
        l_swap = torch.zeros((), device=device)
        if step % config.REG_EVERY == 0:
            # coherence: the three numbers of neighbouring slices should change smoothly
            all_ids = np.arange(n_slices)
            l_coh = t.second_order_slice_loss(network_theta(all_ids, assign), net.glob_scale)
            # polarity swap: exchanging RO+ and RO- must flip the sign of the output
            xp, xm = t.batch_inputs(slices, batch, patterns, pattern_ids)
            theta_swapped = net(xm, xp, swapped=True)
            l_swap = (((theta + theta_swapped) / net.glob_scale[None, :]) ** 2).mean()

        loss = l_phys + config.LAMBDA_GHOST * l_ghost + config.LAMBDA_COH * l_coh + config.LAMBDA_SWAP * l_swap
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.GRAD_CLIP)
        opt.step()
        phys_list.append(l_phys.item())
        step += 1
    sched.step()

    if epoch == 1 or epoch % config.VAL_EVERY == 0 or epoch == config.EPOCHS:
        val = validation_loss()
        flag = ""
        if val < best_val:
            best_val, best_state, bad_checks, flag = val, copy.deepcopy(net.state_dict()), 0, "  <- best"
        else:
            bad_checks += 1
        g = net.global_theta().detach().cpu().numpy()
        log.append((epoch, float(np.mean(phys_list)), val))
        print(f"epoch {epoch:3d}  train {np.mean(phys_list):.4f}  val {val:.4f}  "
              f"shared [p0,px,py] = {g[0]:+.3f} {g[1]:+.3f} {g[2]:+.3f}{flag}")
        if epoch >= config.MIN_EPOCHS and bad_checks >= config.EARLY_STOP_CHECKS:
            print("early stop: validation did not improve")
            break

net.load_state_dict(best_state)
net.eval()
print(f"training time {time.time() - t0:.0f} s, best validation loss {best_val:.4f}")

# ---------------------------------------------------------------------------------- four estimates per slice -> median + MAD
all_ids = np.arange(n_slices)
with torch.no_grad():
    per_pattern = np.stack([network_theta(all_ids, [q] * n_slices).cpu().numpy()
                            for q in range(config.N_TRAIN_PATTERNS)], axis=1)      # (slices, 4, 3)


def unwrap_about_first(v):
    """p0 is an angle: put the four values on the same branch before taking the median."""
    return v[0] + np.angle(np.exp(1j * (v - v[0])))


theta_median = np.zeros((n_slices, 3), np.float32)
theta_mad = np.zeros((n_slices, 3), np.float32)
for i in range(n_slices):
    v = per_pattern[i].astype(float)
    v[:, 0] = unwrap_about_first(v[:, 0])
    theta_median[i] = np.median(v, axis=0)
    theta_mad[i] = 1.4826 * np.median(np.abs(v - np.median(v, axis=0)), axis=0)
unstable = np.any(theta_mad > np.asarray(config.MAD_WARN)[None, :], axis=1)
print("median spread (MAD) over the 4 patterns [p0, px, py] =", np.round(np.median(theta_mad, axis=0), 3))
print("slices above the spread threshold:", [numbers[i] for i in np.where(unstable)[0]])


def validation_value(i, params, n_cg):
    with torch.no_grad():
        p = torch.as_tensor(params, dtype=torch.float32, device=device)
        return slice_loss(i, p, VAL, n_cg)[0].item()


# ---------------------------------------------------------------------------------- per-slice polish (accepted only if validation improves)
theta_final = theta_median.copy()
accepted = np.zeros(n_slices, bool)
val_before = np.zeros(n_slices)
val_after = np.zeros(n_slices)
bounds = torch.tensor(config.REFINE_BOUNDS, dtype=torch.float32, device=device)
for i in range(n_slices):
    start = theta_median[i]
    val_before[i] = validation_value(i, start, config.REFINE_VAL_CG_ITERS)
    best_v, best_theta = val_before[i], start.copy()
    if config.REFINE_STEPS > 0:
        # the three numbers are optimised in a bounded form:  theta = bounds * tanh(raw)
        raw = torch.atanh((torch.as_tensor(start, device=device) / bounds).clamp(-0.98, 0.98)).clone().requires_grad_(True)
        o = torch.optim.Adam([raw], lr=config.REFINE_LR)
        for k in range(1, config.REFINE_STEPS + 1):
            q = (k - 1) % config.N_TRAIN_PATTERNS            # cycle over the four TRAINING patterns
            phys, ghost = slice_loss(i, bounds * torch.tanh(raw), q, config.REFINE_CG_ITERS)
            o.zero_grad()
            (phys + config.REFINE_GHOST_WEIGHT * ghost).backward()
            torch.nn.utils.clip_grad_norm_([raw], 1.0)
            o.step()
            if k % config.REFINE_VAL_EVERY == 0 or k == config.REFINE_STEPS:
                candidate = (bounds * torch.tanh(raw)).detach().cpu().numpy()
                v = validation_value(i, candidate, config.REFINE_VAL_CG_ITERS)
                if v < best_v:
                    best_v, best_theta = v, candidate.copy()
    rel = (val_before[i] - best_v) / max(abs(val_before[i]), 1e-12)
    if rel >= config.MIN_REL_VAL_IMPROVEMENT:
        theta_final[i], accepted[i], val_after[i] = best_theta, True, best_v
    else:
        val_after[i] = val_before[i]
    print(f"slice {numbers[i]:3d}: validation {val_before[i]:.5f} -> {val_after[i]:.5f}  "
          f"{'polish accepted' if accepted[i] else 'polish rejected (network value kept)'}  "
          f"[p0, px, py] = {np.round(theta_final[i], 3)}")

# ---------------------------------------------------------------------------------- untouched TEST pattern
test_pc = np.zeros(n_slices)
test_no_pc = np.zeros(n_slices)
with torch.no_grad():
    for i in range(n_slices):
        test_pc[i] = slice_loss(i, torch.as_tensor(theta_final[i], device=device), TEST, config.TEST_CG_ITERS)[0].item()
        test_no_pc[i] = slice_loss(i, torch.zeros(3, device=device), TEST, config.TEST_CG_ITERS)[0].item()
print(f"test pattern, mean loss:  no phase correction {test_no_pc.mean():.4f}   2D phase {test_pc.mean():.4f}")

# ---------------------------------------------------------------------------------- save
os.makedirs(config.RESULT_DIR, exist_ok=True)
np.savez(os.path.join(config.RESULT_DIR, "theta_zs.npz"), slices=numbers, theta=theta_final,
         theta_cnn_median=theta_median, theta_split_patterns=per_pattern, theta_split_mad=theta_mad,
         unstable_split=unstable, refine_accepted=accepted, val_before=val_before, val_after=val_after,
         test_no_pc=test_no_pc, test_pc=test_pc)
with open(os.path.join(config.RESULT_DIR, "theta_zs.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["slice", "p0_rad", "px_rad_per_halfFOV", "py_rad_per_halfFOV", "mad_p0", "mad_px", "mad_py",
                "unstable_split", "polish_accepted", "val_before", "val_after", "test_no_pc", "test_pc"])
    for i, n in enumerate(numbers):
        w.writerow([n] + [f"{v:.4f}" for v in theta_final[i]] + [f"{v:.4f}" for v in theta_mad[i]] +
                   [int(unstable[i]), int(accepted[i]), f"{val_before[i]:.5f}", f"{val_after[i]:.5f}",
                    f"{test_no_pc[i]:.5f}", f"{test_pc[i]:.5f}"])
with open(os.path.join(config.RESULT_DIR, "training_log.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_loss", "val_loss"])
    w.writerows(log)
torch.save(net.state_dict(), os.path.join(config.RESULT_DIR, "zs_model.pt"))
print("median [p0, px, py] =", np.round(np.median(theta_final, axis=0), 3))
print("polish accepted for", int(accepted.sum()), "of", n_slices, "slices")
print("saved results/theta_zs.csv")
