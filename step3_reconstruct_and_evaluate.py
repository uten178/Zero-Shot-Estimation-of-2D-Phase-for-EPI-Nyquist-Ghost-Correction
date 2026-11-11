"""
STEP 3  -  final reconstruction with PEC-GRAPPA and ghost-to-signal ratio (manuscript 3.5 and 3.6).

For every slice, PEC-GRAPPA is run three times, with the same kernel size and the same code;
only the phase handed to it differs:

   none    phase = 0                                (matched zero-phase control)
   nav1d   phase from a straight-line fit of the 1D navigator data   (skipped if no navigator file)
   zs2d    the zero-shot 2D phase [p0, px, py] from step 2

Coil images are combined with the ESPIRiT maps (matched filter).  PC-SENSE is NOT the final
reconstruction; it is only used here as a quick data-consistency check (is the residual of the
measured data lower with the phase than without?).  A slice is called "stable" if
  - the four split patterns agreed (small MAD),
  - the test pattern loss is lower with the 2D phase than without,
  - the full-data PC-SENSE residual is lower with the 2D phase than without.

Output (folder results/): slice_results.csv, summary.txt, images.npz, figure_slice_<n>.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

import config
import d2_tools as t

device = t.pick_device()
D = t.prepare(config, device)
slices, X, Y = D["slices"], D["X"], D["Y"]
N, NC = D["N"], D["NC"]
lines_plus, lines_minus = D["lines_plus"], D["lines_minus"]
lam = config.PCSENSE_LAMBDA

zs = np.load(os.path.join(config.RESULT_DIR, "theta_zs.npz"))
row_of = {int(n): i for i, n in enumerate(zs["slices"])}

nav = None
if os.path.exists(config.NAV_FILE):
    nav = sio.loadmat(config.NAV_FILE)["raw_phase"]
    print("navigator file found:", nav.shape)
else:
    print("no navigator file -> method nav1d skipped")

# PEC-GRAPPA convention of the research code:  RO+ (pos1) = 'ghost' polarity, RO- (pos2) = reference.
# Our symmetric model gives RO+ minus RO- = -Theta, therefore  psi = -Theta  on the ACS pixels.
acs_shape = slices[0]["acs"].shape[1:]
hx, px_max = t.choose_pec_halfwidth(zs["theta"], acs_shape, NC, lines_minus, lines_plus, config)
print(f"PEC-GRAPPA: max |px| = {px_max:.2f} rad -> RO+/RO- shift {px_max / np.pi:.2f} samples; kernel half-width along kx = {hx}")


def pcsense_check(s, params, full_cg):
    """PC-SENSE from ALL acquired lines. Returns the image, the relative data residual and the ghost energy ratio."""
    theta = t.theta_map(torch.as_tensor(params, dtype=torch.float32, device=device), X, Y)
    lp, lm = torch.as_tensor(lines_plus), torch.as_tensor(lines_minus)
    with torch.no_grad():
        op = t.PCSense(s["csm_final"], theta, lp, lm, lam)
        rho = op.solve(s["ksp"], full_cg)
        kp, km = op.forward(rho)
        num = ((kp - s["ksp"][:, :, lp]).abs() ** 2).sum() + ((km - s["ksp"][:, :, lm]).abs() ** 2).sum()
        den = (s["ksp"][:, :, lp].abs() ** 2).sum() + (s["ksp"][:, :, lm].abs() ** 2).sum()
        a2 = rho.abs() ** 2
        ghost = a2[s["ghost"]].sum() / (a2[s["mask"]].sum() + 1e-8)
    return rho.cpu().numpy(), float(torch.sqrt(num / den)), float(ghost)


def nav_params(s):
    """Navigator phase -> [p0, px, 0]. The sign convention between navigator and this repo is not
    assumed: both signs are tried and the one that predicts the held-out lines better is used."""
    intercept, slope = t.navigator_fit(nav, s["number"] - 1, config.NAV_REP)
    best = None
    for sign in (+1, -1):
        p = t.nav_to_theta_params(intercept, slope, N, sign)
        theta = t.theta_map(torch.tensor(p, dtype=torch.float32, device=device), X, Y)
        with torch.no_grad():
            loss = t.heldout_loss(s["csm"], theta, s["ksp"], D["patterns"][D["val_id"]], lam, config.VAL_CG_ITERS)[0].item()
        if best is None or loss < best[0]:
            best = (loss, p)
    return best[1]


def pec_image(s, params):
    """PEC-GRAPPA image of one slice for a given phase [p0, px, py]."""
    ksp = np.zeros((N, N, NC), np.complex128)                    # only the acquired lines
    k = s["ksp"].cpu().numpy()
    for lines in (lines_plus, lines_minus):
        ksp[:, lines, :] = np.transpose(k[:, :, lines], (1, 2, 0))
    acs = np.transpose(s["acs"], (1, 2, 0))
    psi = -t.theta_on_acs_grid(params, acs_shape, N, N)
    csm = np.transpose(s["csm_final"].cpu().numpy(), (1, 2, 0))
    return t.pec_grappa_recon(ksp, acs, psi, lines_minus, lines_plus, config, hx, csm)


rows, stack = [], {}
for s in slices:
    n = s["number"]
    i = row_of[n]
    theta_zs = zs["theta"][i]

    # --- PC-SENSE data-consistency check (no phase vs 2D phase)
    _, res_no, ghost_no = pcsense_check(s, np.zeros(3), config.FINAL_NO_PC_CG_ITERS)
    _, res_pc, ghost_pc = pcsense_check(s, theta_zs, config.FINAL_CG_ITERS)
    stable = (not bool(zs["unstable_split"][i])) and (zs["test_pc"][i] < zs["test_no_pc"][i]) and (res_pc < res_no)

    # --- PEC-GRAPPA, three phases
    methods = {"none": np.zeros(3), "zs2d": theta_zs}
    if nav is not None and (n - 1) < nav.shape[1]:
        methods["nav1d"] = nav_params(s)
    row = dict(slice=n, p0=theta_zs[0], px=theta_zs[1], py=theta_zs[2],
               mad_p0=zs["theta_split_mad"][i][0], mad_px=zs["theta_split_mad"][i][1], mad_py=zs["theta_split_mad"][i][2],
               polish_accepted=int(zs["refine_accepted"][i]), val_before=zs["val_before"][i], val_after=zs["val_after"][i],
               test_no_pc=zs["test_no_pc"][i], test_pc=zs["test_pc"][i],
               pcsense_residual_no_pc=res_no, pcsense_residual_pc=res_pc, pcsense_ghost_no_pc=ghost_no, pcsense_ghost_pc=ghost_pc,
               stable=int(stable))
    for name, params in methods.items():
        img, diag = pec_image(s, params)
        row[f"GSR_{name}"] = t.gsr_percent(img, s["mask_np"], s["ghost_np"], s["background"])
        row[f"GSRplain_{name}"] = t.gsr_plain(img, s["mask_np"], s["ghost_np"])
        if name == "zs2d":
            row.update(pec_coverage=diag["coverage"], pec_filled=diag["filled"], pec_period=diag["period"],
                       pec_cal_nrmse=diag["cal_nrmse"], pec_cal_nrmse_max=diag["cal_nrmse_max"])
        stack.setdefault(name, []).append(img.astype(np.float32))
    print(f"slice {n:3d}  " + "  ".join(f"{k}={v:.2f}%" for k, v in row.items() if k.startswith("GSR_")) +
          f"   [p0,px,py]={np.round(theta_zs, 2)}  stable={bool(stable)}")
    rows.append(row)

# ---------------------------------------------------------------------------------- tables
cols = list(rows[0].keys())
with open(os.path.join(config.RESULT_DIR, "slice_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: (v if k == "slice" else f"{float(v):.5g}") for k, v in r.items()})
np.savez_compressed(os.path.join(config.RESULT_DIR, "images.npz"), slices=[s["number"] for s in slices],
                    **{k: np.stack(v) for k, v in stack.items()})

with open(os.path.join(config.RESULT_DIR, "summary.txt"), "w") as f:
    lines = [f"slices: {len(rows)}    stable: {sum(r['stable'] for r in rows)} / {len(rows)}",
             f"PEC-GRAPPA kernel half-width along kx: {hx}"]
    for c in [c for c in cols if c.startswith("GSR_")] + [c for c in cols if c.startswith("GSRplain_")]:
        v = np.array([r[c] for r in rows])
        lines.append(f"{c:16s} mean {v.mean():7.2f} %   median {np.median(v):7.2f} %")
    for line in lines:
        print(line)
        f.write(line + "\n")

# ---------------------------------------------------------------------------------- figures (first 6 slices)
for i, s in enumerate(slices[:6]):
    names = list(stack)
    fig, ax = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.4))
    ax = np.atleast_1d(ax)
    vmax = np.abs(stack["zs2d"][i]).max() * 0.7
    for a, name in zip(ax, names):
        a.imshow(np.abs(stack[name][i]).T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
        a.set_title(f"PEC-GRAPPA, {name}   %GSR {rows[i]['GSR_' + name]:.1f}")
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULT_DIR, f"figure_slice_{s['number']}.png"), dpi=110)
    plt.close()
print("done -> results/")
