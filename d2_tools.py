"""
d2_tools.py  --  the building blocks used by the step scripts.

Everything is written in a plain style: small functions, one job each, comments in words.
Array convention (torch part):   k-space / images have shape (coils, readout, phase-enc)
                                 = (NC, N, N);  axis 1 = readout (x),  axis 2 = phase encoding (y)
Array convention (PEC-GRAPPA, numpy part):   (readout, phase-enc, coils)   as in the MATLAB files

Model (symmetric gauge):   RO+ lines see  exp(-i*Theta/2),   RO- lines see  exp(+i*Theta/2)
                           Theta(x,y) = p0 + px*x + py*y,   x,y in [-1, 1]   (rad)

Sections
   1 loading                       6 PC-SENSE (physics model used during estimation)
   2 Fourier helpers               7 self-supervised line split + loss
   3 coil maps, masks (ESPIRiT)    8 the network (JointPCNet)
   4 1D navigator comparison       9 PEC-GRAPPA (final reconstruction)
   5 the phase model              10 image quality (GSR)  +  put-together helper
"""
import glob
import math
import os
import re

import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# 1. LOADING
# ======================================================================================

def find_slice_files(data_dir, wanted=None):
    """Return {slice_number: path} for every Slice_<n>.mat found in data_dir."""
    files = {}
    for path in glob.glob(os.path.join(data_dir, "Slice_*.mat")):
        m = re.search(r"Slice_(\d+)\.mat$", path)
        if m:
            files[int(m.group(1))] = path
    if wanted is not None:
        files = {n: p for n, p in files.items() if n in wanted}
    if not files:
        raise FileNotFoundError(f"No Slice_*.mat files found in {data_dir}. See data/README.md")
    return dict(sorted(files.items()))


def load_slice(path):
    """Read one slice file. Returns a dictionary of numpy arrays."""
    m = sio.loadmat(path)
    ksp = np.transpose(m["KOE_CC"], (2, 0, 1))        # (NC, N, N)  undersampled EPI k-space
    acs = np.transpose(m["kcal_cc"], (2, 0, 1))       # (NC, 64, 32) GRE reference scan
    plus = np.unique(m["pos1"].ravel().astype(int))   # phase-encode lines of RO+ (MATLAB: 1-based)
    minus = np.unique(m["pos2"].ravel().astype(int))  # phase-encode lines of RO-
    if min(plus.min(), minus.min()) >= 1:
        plus, minus = plus - 1, minus - 1             # -> 0-based
    # scale: 99th percentile of the ACQUIRED samples of the EPI data; the ACS by its own 99th percentile
    acquired = np.concatenate([np.abs(ksp[:, :, plus]).ravel(), np.abs(ksp[:, :, minus]).ravel()])
    ksp = ksp / max(np.percentile(acquired, 99.0), 1e-12)
    acs = acs / max(np.percentile(np.abs(acs), 99.0), 1e-12)
    return dict(ksp=ksp.astype(np.complex64), acs=acs.astype(np.complex64), lines_plus=plus, lines_minus=minus)


# ======================================================================================
# 2. FOURIER HELPERS (centered, orthonormal)
# ======================================================================================

def fft2c(x):
    """torch: acts on the last two axes."""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(k):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def np_fft2c(x):
    """numpy: acts on the FIRST two axes (readout, phase-enc, coils)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(0, 1)), axes=(0, 1), norm="ortho"), axes=(0, 1))


def np_ifft2c(k):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(0, 1)), axes=(0, 1), norm="ortho"), axes=(0, 1))


# ======================================================================================
# 3. COIL MAPS (ESPIRiT) AND MASKS FROM THE REFERENCE SCAN (ACS)
# ======================================================================================

def espirit_maps(acs, N, calib_width=32, kernel=8, n_vectors=80, chunk=8):
    """
    ESPIRiT sensitivities from ONE calibration block (the GRE ACS of D2).
      1. zero-pad the ACS into an N x N k-space,
      2. cut the central calib_width x calib_width block and build the block-Hankel
         calibration matrix (every kernel x kernel patch = one row),
      3. SVD -> keep the first n_vectors right singular vectors (signal space),
      4. at every pixel: eigen-decomposition of the 'G^H G' matrix; the eigenvector of the
         largest eigenvalue is the sensitivity vector (unit norm over coils).
    Returns maps (NC, N, N) (unit norm, NOT masked) and the largest eigenvalue map (N, N).
    Written after espirit_2cal / espirit in the research code; numpy only.
    """
    nc, nr, npe = acs.shape
    g = np.zeros((N, N, nc), np.complex64)
    g[(N - nr) // 2:(N - nr) // 2 + nr, (N - npe) // 2:(N - npe) // 2 + npe, :] = np.transpose(acs, (1, 2, 0))
    W, B = int(calib_width), int(kernel)
    x1, y1 = (N - W) // 2, (N - W) // 2
    cal = g[x1:x1 + W, y1:y1 + W, :]
    n = W - B + 1
    A = np.empty((n * n, B * B * nc), np.complex64)
    row = 0
    for yy in range(n):
        for xx in range(n):
            A[row] = cal[xx:xx + B, yy:yy + B, :].reshape(-1, order="F")
            row += 1
    _, s, Vh = np.linalg.svd(A, full_matrices=False)
    V = Vh.conj().T[:, :min(int(n_vectors), Vh.shape[0])]
    GhG = np.zeros((N, N, nc, nc), np.complex64)
    bx, by = (N - B) // 2, (N - B) // 2
    for k0 in range(0, V.shape[1], int(chunk)):
        ks = V[:, k0:k0 + int(chunk)]
        nk = ks.shape[1]
        buf = np.zeros((nk, N, N, nc), np.complex64)
        buf[:, bx:bx + B, by:by + B, :] = np.stack([ks[:, j].reshape(B, B, nc, order="F") for j in range(nk)], axis=0)
        S = np.fft.fftshift(np.fft.fft(np.fft.fftshift(buf, axes=1), axis=1), axes=1)
        S = np.fft.fftshift(np.fft.fft(np.fft.fftshift(S, axes=2), axis=2), axes=2) / B
        GhG += np.einsum("kxyc,kxyd->xycd", S.conj(), S, optimize=True)
    E, Vec = np.linalg.eigh(GhG.reshape(-1, nc, nc))
    eigs = E.reshape(N, N, nc)[..., -1].astype(np.float32)
    maps = Vec.reshape(N, N, nc, nc)[..., -1].astype(np.complex64)
    maps = maps * np.exp(-1j * np.angle(maps[..., -1:]))                       # fix the phase gauge
    maps = maps / np.maximum(np.sqrt((np.abs(maps) ** 2).sum(-1, keepdims=True)), 1e-8)
    return np.transpose(maps, (2, 0, 1)).astype(np.complex64), eigs


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def coil_maps_and_masks(acs, N, cfg):
    """
    From the small ACS block:
      * ESPIRiT maps  (unmasked, unit norm)  and the support mask (largest eigenvalue > threshold),
      * the low-resolution image (root-sum-of-squares of the zero-padded ACS),
      * object mask   = bright part of that image AND inside the ESPIRiT support,
      * ghost region  = object shifted by FOV/2 along phase encoding, minus the object (and a margin),
      * background    = far from object and ghost (used to estimate the noise level).
    """
    maps_unmasked, eigs = espirit_maps(acs, N, cfg.ESPIRIT_WIDTH, cfg.ESPIRIT_KERNEL, cfg.ESPIRIT_VECTORS)
    esp_mask = (eigs > cfg.ESPIRIT_MASK_THRESH).astype(np.float32)
    if esp_mask.mean() < 0.005:                       # mask collapsed -> no mask (as in the research code)
        esp_mask = np.ones_like(esp_mask)

    nc, nr, npe = acs.shape
    padded = np.zeros((nc, N, N), np.complex64)
    padded[:, (N - nr) // 2:(N - nr) // 2 + nr, (N - npe) // 2:(N - npe) // 2 + npe] = acs
    rss = np.sqrt((np.abs(ifft2c(torch.from_numpy(padded)).numpy()) ** 2).sum(0) + 1e-12)

    mask = (rss > cfg.OBJECT_FRAC * np.percentile(rss, 99.5)) & (esp_mask > 0)
    mask = largest_component(ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=2)))
    if mask.sum() < 100:                              # fallback: bright part only
        mask = rss > cfg.OBJECT_FRAC * np.percentile(rss, 99.5)
        mask = largest_component(ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=2)))
    shifted = np.roll(mask, N // 2, axis=1)
    ghost = shifted & ~ndi.binary_dilation(mask, iterations=cfg.GHOST_DILATE)
    if ghost.sum() < 20:
        ghost = shifted & ~mask
    background = ~ndi.binary_dilation(mask | shifted, iterations=cfg.GHOST_DILATE)
    maps_masked = maps_unmasked * esp_mask[None]
    return dict(csm_unmasked=maps_unmasked.astype(np.complex64), csm_masked=maps_masked.astype(np.complex64),
                esp_mask=esp_mask, mask=mask, ghost=ghost, background=background, rss=rss)


def combine_coils(coil_images, csm=None):
    """
    Coil combination of the PEC-GRAPPA coil images (coils on the LAST axis, numpy):
        with maps :  | sum_c conj(C_c) I_c / sum_c |C_c|^2 |     (sensitivity weighted)
        without   :  sqrt( sum_c |I_c|^2 )                       (RSS, has a noise floor)
    """
    if csm is None:
        return np.sqrt((np.abs(coil_images) ** 2).sum(-1))
    num = (np.conj(csm) * coil_images).sum(-1)
    den = (np.abs(csm) ** 2).sum(-1)
    return np.abs(num / np.maximum(den, 1e-8))


# ======================================================================================
# 4. 1D NAVIGATOR (optional comparison): straight-line fit of the RO+/RO- phase
# ======================================================================================

def navigator_fit(raw_phase, slice_index, rep, mag_frac=0.1):
    """
    raw_phase : complex array (readout, slices, repetitions) = coil-combined RO+ x conj(RO-)
    Returns (intercept, slope) of a magnitude-weighted straight-line fit to the unwrapped
    phase;  x = 0 ... N-1 (pixel index).  slope is rad per pixel.
    """
    pd = raw_phase[:, slice_index, rep]
    mag = np.abs(pd)
    keep = mag > mag_frac * mag.max()
    x = np.arange(len(pd), dtype=float)
    ph = np.unwrap(np.angle(pd))
    w = mag[keep]
    A = np.stack([np.ones(keep.sum()), x[keep]], axis=1)
    beta = np.linalg.pinv(A.T @ (w[:, None] * A)) @ (A.T @ (w * ph[keep]))
    return beta[0], beta[1]


def nav_to_theta_params(intercept, slope, N, sign):
    """Convert (intercept, slope/pixel) to [p0, px, py] in the convention of this repo."""
    half = (N - 1) / 2.0
    p0 = intercept + slope * half          # value in the middle of the FOV
    px = slope * half                      # rad per half-FOV
    return sign * np.array([p0, px, 0.0])


# ======================================================================================
# 5. THE PHASE MODEL
# ======================================================================================

def coordinate_grids(N, device="cpu"):
    """x runs along readout (axis 0 of an N x N image), y along phase encoding (axis 1)."""
    g = (torch.arange(N, dtype=torch.float32, device=device) - (N - 1) / 2.0) / ((N - 1) / 2.0)
    return g[:, None].expand(N, N), g[None, :].expand(N, N)


def theta_map(params, X, Y):
    """params (..., 3) = [p0, px, py]  ->  Theta map (..., N, N)"""
    p0, px, py = params[..., 0], params[..., 1], params[..., 2]
    return p0[..., None, None] + px[..., None, None] * X + py[..., None, None] * Y


# ======================================================================================
# 6. PC-SENSE: forward operator and (preconditioned) conjugate-gradient solver
# ======================================================================================

class PCSense:
    """
    Phase-corrected SENSE for one slice.
        k+ = P+ F{ C rho exp(-i Theta/2) },   k- = P- F{ C rho exp(+i Theta/2) }
    lines_plus / lines_minus select the phase-encode lines that are USED in a given solve.
    A small Tikhonov term lam*||rho||^2 keeps the solve stable (no spatial smoothing).
    """

    def __init__(self, csm, theta, lines_plus, lines_minus, lam):
        self.C = csm
        self.Cc = csm.conj()
        self.ep_plus = torch.polar(torch.ones_like(theta), -theta / 2)
        self.ep_minus = torch.polar(torch.ones_like(theta), +theta / 2)
        self.lines_plus = lines_plus
        self.lines_minus = lines_minus
        self.lam = lam
        N = csm.shape[-1]
        density = (len(lines_plus) + len(lines_minus)) / N      # fraction of lines used
        self.diag = density * (csm.abs() ** 2).sum(0) + lam      # simple (Jacobi) preconditioner

    def forward(self, rho):
        kp = fft2c(rho[None] * self.ep_plus[None] * self.C)[:, :, self.lines_plus]
        km = fft2c(rho[None] * self.ep_minus[None] * self.C)[:, :, self.lines_minus]
        return kp, km

    def adjoint(self, kp, km):
        out = 0
        for k, lines, ep in ((kp, self.lines_plus, self.ep_plus), (km, self.lines_minus, self.ep_minus)):
            full = torch.zeros(self.C.shape, dtype=self.C.dtype, device=self.C.device)
            full[:, :, lines] = k
            out = out + ep.conj() * (self.Cc * ifft2c(full)).sum(0)
        return out

    def normal(self, rho):
        kp, km = self.forward(rho)
        return self.adjoint(kp, km) + self.lam * rho

    def solve(self, ksp, n_iter):
        """Preconditioned conjugate gradient, fixed number of iterations (differentiable)."""
        b = self.adjoint(ksp[:, :, self.lines_plus], ksp[:, :, self.lines_minus])
        x = torch.zeros_like(b)
        r = b
        z = r / self.diag.clamp_min(1e-6)
        p = z
        rz = torch.vdot(r.flatten(), z.flatten()).real
        for _ in range(int(n_iter)):
            Ap = self.normal(p)
            alpha = rz / torch.vdot(p.flatten(), Ap.flatten()).real.clamp_min(1e-8)
            x = x + alpha * p
            r = r - alpha * Ap
            z = r / self.diag.clamp_min(1e-6)
            rz_new = torch.vdot(r.flatten(), z.flatten()).real
            p = z + (rz_new / rz.clamp_min(1e-8)) * p
            rz = rz_new
        return x


# ======================================================================================
# 7. SELF-SUPERVISED SPLIT OF THE LINES (SSDU idea) AND THE LOSS
# ======================================================================================

def stratified_split(lines, hold_frac, rng):
    """Pick hold_frac of the lines as loss lines, spread evenly over the k-space (one per bin)."""
    idx = np.sort(np.asarray(lines, dtype=int))
    n = len(idx)
    n_hold = max(1, int(round(hold_frac * n)))
    edges = np.linspace(0, n, n_hold + 1).astype(int)
    pick = [int(rng.integers(a, b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    pick = np.unique(pick)
    rest = np.setdiff1d(np.arange(n), pick)
    if len(pick) < n_hold:
        pick = np.sort(np.r_[pick, rng.choice(rest, n_hold - len(pick), replace=False)])
    elif len(pick) > n_hold:
        pick = np.sort(rng.choice(pick, n_hold, replace=False))
    hold = np.sort(idx[pick])
    fit = np.sort(np.setdiff1d(idx, hold))
    return fit, hold


def make_split_patterns(lines_plus, lines_minus, n_train, hold_frac, seed):
    """
    n_train training patterns + 1 validation pattern + 1 test pattern (always in this order).
    Each pattern: lines used to RECONSTRUCT (fit) and lines used to compute the LOSS (hold),
    separately for RO+ and RO-.  'avail' marks, on the acquired lines, which are available.
    """
    rng0 = np.random.default_rng(seed)
    patterns = []
    for q in range(n_train + 2):
        rng = np.random.default_rng(int(rng0.integers(0, 2 ** 31 - 1)))
        fit_p, hold_p = stratified_split(lines_plus, hold_frac, rng)
        fit_m, hold_m = stratified_split(lines_minus, hold_frac, rng)
        patterns.append(dict(
            fit_p=torch.as_tensor(fit_p), hold_p=torch.as_tensor(hold_p),
            fit_m=torch.as_tensor(fit_m), hold_m=torch.as_tensor(hold_m),
            avail_p=torch.tensor([0.0 if l in set(hold_p.tolist()) else 1.0 for l in lines_plus]),
            avail_m=torch.tensor([0.0 if l in set(hold_m.tolist()) else 1.0 for l in lines_minus])))
    return patterns


def heldout_loss(csm, theta, ksp, split, lam, n_cg, mask=None, ghost=None, want_image=False):
    """
    The physics loss of the manuscript, Eqs. (6)-(9):
      1. reconstruct from the FIT lines (PC-SENSE, theta = hypothesis),
      2. predict the held-out LOSS lines,
      3. compare with the measured loss lines:  0.7 * normalised L2 + 0.3 * normalised L1,
         averaged over the two polarities.
    Also returns the ghost term of Eq. (10)  (ghost energy / object energy of the image).
    """
    op = PCSense(csm, theta, split["fit_p"], split["fit_m"], lam)
    rho = op.solve(ksp, n_cg)
    eps = 1e-8
    total = 0
    for ep, hold in ((op.ep_plus, split["hold_p"]), (op.ep_minus, split["hold_m"])):
        pred = fft2c(rho[None] * ep[None] * csm)[:, :, hold]
        meas = ksp[:, :, hold]
        diff = pred - meas
        r2 = torch.sqrt((diff.abs() ** 2).sum() / ((meas.abs() ** 2).sum() + eps) + eps)
        r1 = diff.abs().sum() / (meas.abs().sum() + eps)
        total = total + 0.5 * (0.7 * r2 + 0.3 * r1)
    ghost_term = torch.zeros((), device=theta.device)
    if ghost is not None:
        a2 = rho.abs() ** 2
        ghost_term = a2[ghost].sum() / (a2[mask].sum() + eps)
    if want_image:
        return total, ghost_term, rho
    return total, ghost_term


def second_order_slice_loss(theta_all, glob_scale):
    """Coherence term: second difference of the three numbers across neighbouring slices."""
    if theta_all.shape[0] < 3:
        return torch.zeros((), device=theta_all.device)
    q = theta_all / glob_scale[None, :]
    return ((q[2:] - 2 * q[1:-1] + q[:-2]) ** 2).mean()


def representative_indices(n, k):
    """k indices spread evenly over 0 ... n-1."""
    if n <= k:
        return np.arange(n, dtype=int)
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))


# ======================================================================================
# 8. THE NETWORK  (JointPCNet: shared encoders for RO+ and RO-, compact regression head)
# ======================================================================================

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        y = self.c2(F.relu(self.c1(x)))
        return F.relu(x + y)


class SharedEncoder(nn.Module):
    """Raw k-space lines (coils as channels) -> one feature vector (mean and max pooling)."""

    def __init__(self, c_in, base=32, n_res=3):
        super().__init__()
        self.inp = nn.Conv2d(c_in, base, 5, padding=2)
        self.res = nn.Sequential(*[ResBlock(base) for _ in range(n_res)])
        self.down1 = nn.Conv2d(base, 2 * base, 3, stride=(2, 1), padding=1)
        self.down2 = nn.Conv2d(2 * base, 2 * base, 3, stride=(2, 1), padding=1)
        self.out_dim = 4 * base

    def forward(self, x):
        z = F.relu(self.inp(x))
        z = self.res(z)
        z = F.relu(self.down1(z))
        z = F.relu(self.down2(z))
        return torch.cat([z.mean(dim=(-2, -1)), z.amax(dim=(-2, -1))], dim=1)


class JointPCNet(nn.Module):
    """
    Input : RO+ and RO- k-space lines of one slice (real, imaginary, ky position, 'available' flag).
    Output: three numbers [p0, px, py] =  shared global phase (same for all slices, starts at the
            screening basin)  +  slice-specific residual (bounded by tanh, starts exactly at zero).
    """

    def __init__(self, c_in, basin, cfg):
        super().__init__()
        self.enc = SharedEncoder(c_in, cfg.BASE_FILTERS, cfg.RESIDUAL_BLOCKS)
        d = self.enc.out_dim
        self.head = nn.Sequential(nn.Linear(4 * d, cfg.HEAD_HIDDEN), nn.ReLU(),
                                  nn.Linear(cfg.HEAD_HIDDEN, 64), nn.ReLU(), nn.Linear(64, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        glob_scale = torch.tensor(cfg.GLOBAL_SCALES, dtype=torch.float32)
        start = torch.as_tensor(basin, dtype=torch.float32) / glob_scale
        self.global_raw = nn.Parameter(torch.atanh(start.clamp(-0.98, 0.98)))
        self.register_buffer("glob_scale", glob_scale)
        self.register_buffer("res_scale", torch.tensor(cfg.RESIDUAL_SCALES, dtype=torch.float32))

    def global_theta(self):
        return torch.tanh(self.global_raw) * self.glob_scale

    def forward(self, xp, xm, swapped=False):
        fp, fm = self.enc(xp), self.enc(xm)
        z = torch.cat([fp, fm, fp - fm, fp * fm], dim=1)
        residual = torch.tanh(self.head(z)) * self.res_scale[None, :]
        base = self.global_theta()[None, :]
        if swapped:                      # RO+ and RO- exchanged -> the shared phase changes sign
            return -base + residual
        return base + residual


def slice_features(ksp, lines):
    """Fixed part of the network input for one polarity: (2*NC + 1, N, n_lines) = Re, Im, ky position."""
    k = ksp[:, :, lines]
    N = ksp.shape[-1]
    ky = ((lines.to(torch.float32) - (N - 1) / 2.0) / (N / 2.0))[None, None, :].expand(1, ksp.shape[1], len(lines))
    return torch.cat([k.real, k.imag, ky.to(k.device)], dim=0)


def network_input(slc, pattern):
    """Held-out lines are zeroed; a last channel tells the network which lines are available."""
    out = []
    for feat, avail in ((slc["feat_plus"], pattern["avail_p"]), (slc["feat_minus"], pattern["avail_m"])):
        a = avail.to(feat.device)
        nch = feat.shape[0] - 1
        x = feat.clone()
        x[:nch] = x[:nch] * a[None, None, :]
        out.append(torch.cat([x, a[None, None, :].expand(1, feat.shape[1], len(a))], dim=0))
    return out        # [x_plus, x_minus]


def batch_inputs(slices, ids, patterns, pattern_ids):
    xp, xm = [], []
    for i, q in zip(ids, pattern_ids):
        a, b = network_input(slices[int(i)], patterns[int(q)])
        xp.append(a)
        xm.append(b)
    return torch.stack(xp), torch.stack(xm)


# ======================================================================================
# 9. PEC-GRAPPA  (final reconstruction; works on k-space, coil maps are NOT used for the kernels)
#    numpy arrays here are (readout, phase-enc, coils)
# ======================================================================================

def pec_sampling_period(lines_ref, lines_gh):
    """Period of the combined line pattern (e.g. 4) and the residues of the two polarities."""
    diffs = []
    for lines in (np.sort(lines_ref), np.sort(lines_gh)):
        d = np.diff(lines)
        diffs.extend(d[d > 0].tolist())
    period = int(np.gcd.reduce(np.asarray(diffs, dtype=int)))
    if period < 2:
        vals, cnt = np.unique(np.asarray(diffs, dtype=int), return_counts=True)
        period = int(vals[np.argmax(cnt)])
    r_ref, r_gh = int(lines_ref[0] % period), int(lines_gh[0] % period)
    if r_ref == r_gh:
        raise ValueError("RO+ and RO- lines fall on the same residue - PEC-GRAPPA needs interleaved lines")
    if not (np.all(lines_ref % period == r_ref) and np.all(lines_gh % period == r_gh)):
        raise ValueError("RO+ / RO- lines are not a regular pattern")
    return period, r_ref, r_gh


def pec_source_offsets(target_residue, period, source_residues, n_source):
    """The n_source nearest acquired lines (as ky offsets) around a line of a given residue."""
    cand = [d for d in range(-3 * period, 3 * period + 1) if (target_residue + d) % period in source_residues]
    cand = sorted(cand, key=lambda d: (abs(d), d))
    if len(cand) < n_source:
        raise ValueError("not enough source lines for PEC-GRAPPA")
    return np.asarray(sorted(cand[:n_source]), dtype=int)


def theta_on_acs_grid(theta, acs_shape, NX, NY):
    """The SAME phase plane Theta evaluated on the pixels of the (small) ACS block."""
    p0, px, py = [float(v) for v in theta]
    nxa, nya = acs_shape
    xe = NX / 2.0 + (np.arange(nxa) - nxa / 2.0) * (NX / nxa)
    ye = NY / 2.0 + (np.arange(nya) - nya / 2.0) * (NY / nya)
    xn = (xe - (NX - 1) / 2.0) / ((NX - 1) / 2.0)
    yn = (ye - (NY - 1) / 2.0) / ((NY - 1) / 2.0)
    XX, YY = np.meshgrid(xn, yn, indexing="ij")
    return p0 + px * XX + py * YY


def _window(src, row, hx, offsets):
    """Features of one k-space row: for every kx the values at kx+offsets, all coils.
    Returns (n_kx, (2*hx+1)*NC) for kx = hx ... nx-hx-1."""
    nx = src.shape[0]
    cols = [src[hx + o: nx - hx + o, row, :] for o in offsets]
    return np.stack(cols, axis=1).reshape(nx - 2 * hx, -1)


def pec_calibrate(acs, psi, lines_ref, lines_gh, cfg, hx):
    """
    PEC-GRAPPA calibration from the ACS (readout, ky, coils) and the phase psi on the ACS pixels.
    The ACS has no polarity alternation, so the 'ghost-polarity' ACS is made from it:
        ACS_GH = FFT{ IFFT{ACS} * exp(+i psi) }.
    For each residue of the line pattern, a ridge-regularised least-squares fit predicts the
    ACS from neighbouring ACS lines (reference or ghost polarity, as they would be acquired).
    """
    acs = np.asarray(acs, dtype=np.complex128)
    nxa, nya, nc = acs.shape
    period, r_ref, r_gh = pec_sampling_period(lines_ref, lines_gh)
    source_res = {r_ref, r_gh}
    acs_gh = np_fft2c(np_ifft2c(acs) * np.exp(1j * psi)[..., None])
    kx_off = np.arange(-hx, hx + 1)

    weights, ky_offsets, nrmse = {}, {}, {}
    for res in range(period):
        n_src = cfg.PEC_N_SOURCE_ACQUIRED if res in source_res else cfg.PEC_N_SOURCE_MISSING
        dys = pec_source_offsets(res, period, source_res, n_src)
        ky_offsets[res] = dys
        my = int(np.abs(dys).max())
        X_rows, Y_rows = [], []
        for ky in range(my, nya - my):
            feats = []
            for dy in dys:
                src = acs if (res + int(dy)) % period == r_ref else acs_gh
                feats.append(_window(src, ky + int(dy), hx, kx_off))
            X_rows.append(np.concatenate(feats, axis=1))
            Y_rows.append(acs[hx:nxa - hx, ky, :])
        Xc, Yc = np.concatenate(X_rows), np.concatenate(Y_rows)
        gram = Xc.conj().T @ Xc
        reg = (cfg.PEC_LAMBDA ** 2) * max(float(np.linalg.eigvalsh(gram)[-1].real), 1e-30)
        W = np.linalg.solve(gram + reg * np.eye(gram.shape[0]), Xc.conj().T @ Yc)
        weights[res] = W
        nrmse[res] = float(np.linalg.norm(Xc @ W - Yc) / (np.linalg.norm(Yc) + 1e-30))
    return dict(weights=weights, ky_offsets=ky_offsets, kx_off=kx_off, period=period, nrmse=nrmse)


def pec_grappa_recon(ksp, acs, psi, lines_ref, lines_gh, cfg, hx, csm=None):
    """
    FINAL reconstruction.  ksp (readout, ky, coils) contains only the acquired lines.
    Every missing / acquired line is predicted from neighbouring acquired lines of BOTH
    polarities with the calibrated weights; the phase psi is what makes the two polarities
    'compatible'.  Returns the combined magnitude image, the coil images and diagnostics.
    """
    ksp = np.asarray(ksp, dtype=np.complex128)
    nx, ny, nc = ksp.shape
    spec = pec_calibrate(acs, psi, lines_ref, lines_gh, cfg, hx)
    period = spec["period"]
    acquired = np.zeros(ny, dtype=bool)
    acquired[lines_ref] = True
    acquired[lines_gh] = True

    kout = np.zeros_like(ksp)
    predicted = np.zeros((nx, ny), dtype=bool)
    for ky in range(ny):
        res = ky % period
        src_rows = ky + spec["ky_offsets"][res]
        if src_rows.min() < 0 or src_rows.max() >= ny or not acquired[src_rows].all():
            continue
        feats = np.concatenate([_window(ksp, int(r), hx, spec["kx_off"]) for r in src_rows], axis=1)
        kout[hx:nx - hx, ky, :] = feats @ spec["weights"][res]
        predicted[hx:nx - hx, ky] = True

    n_filled = 0
    if cfg.PEC_FILL_ACQUIRED:                     # where PEC could not predict, keep the acquired sample
        acq2d = np.zeros((nx, ny), dtype=bool)
        acq2d[:, acquired] = True
        fill = acq2d & ~predicted
        kout[fill, :] = ksp[fill, :]
        n_filled = int(fill.sum())

    coil = np_ifft2c(kout)
    image = combine_coils(coil, csm if cfg.PEC_COMBINE == "espirit" else None)
    diag = dict(coverage=float(predicted.mean()), filled=n_filled, period=period,
                cal_nrmse=float(np.mean(list(spec["nrmse"].values()))),
                cal_nrmse_max=float(np.max(list(spec["nrmse"].values()))))
    return image, diag


def choose_pec_halfwidth(theta_all, acs_shape, nc, lines_ref, lines_gh, cfg):
    """
    A phase px*x (x from -1 to 1) shifts RO+ against RO- by |px|/pi samples along kx. The kernel window
    must span that shift. Widen the window if needed, but not beyond what the ACS can determine.
    """
    hx = cfg.PEC_KX_HALFWIDTH
    px_max = float(np.max(np.abs(np.asarray(theta_all)[:, 1])))
    if cfg.PEC_KX_AUTO:
        need = int(math.ceil(px_max / math.pi)) + cfg.PEC_KX_MARGIN
        hx = int(np.clip(need, cfg.PEC_KX_HALFWIDTH, cfg.PEC_KX_HALFWIDTH_MAX))
        period, r_ref, r_gh = pec_sampling_period(lines_ref, lines_gh)
        dys = pec_source_offsets(0, period, {r_ref, r_gh}, cfg.PEC_N_SOURCE_MISSING)
        my = int(np.abs(dys).max())
        nxa, nya = acs_shape
        while hx > cfg.PEC_KX_HALFWIDTH:
            equations = max(0, nya - 2 * my) * max(0, nxa - 2 * hx)
            unknowns = (2 * hx + 1) * cfg.PEC_N_SOURCE_MISSING * nc
            if equations / unknowns >= cfg.PEC_MIN_EQ_PER_UNKNOWN:
                break
            hx -= 1
    return hx, px_max


# ======================================================================================
# 10. IMAGE QUALITY: ghost-to-signal ratio, and the put-everything-together helper
# ======================================================================================

def gsr_percent(image, mask, ghost, background):
    """
    %GSR = 100 * (G - sigma) / (S - sigma)   (manuscript, Section 3.6)
      G = mean |image| in the ghost region, S = mean |image| in the object,
      sigma = noise level = standard deviation of |image| in object-free, ghost-free background.
    (The manuscript draws the border rows by hand; here the background region is found automatically.)
    """
    a = np.abs(image)
    sigma = a[background].std() if background.any() else 0.0
    return 100.0 * (a[ghost].mean() - sigma) / (a[mask].mean() - sigma)


def gsr_plain(image, mask, ghost):
    """Simple ratio  100 * mean|ghost| / mean|object|  (no noise correction)."""
    a = np.abs(image)
    return 100.0 * a[ghost].mean() / a[mask].mean()


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare(cfg, device):
    """
    Load the requested slices, compute coil maps + masks, the split patterns and the network inputs.
    Returns a dictionary with
        slices   list of per-slice dictionaries,   N, NC, X, Y,  lines_plus, lines_minus,
        patterns the 4 training + validation + test patterns,
        assign   which training pattern each slice uses (slice index mod 4).
    """
    files = find_slice_files(cfg.DATA_DIR, cfg.SLICES)
    slices = []
    for number, path in files.items():
        d = load_slice(path)
        N = d["ksp"].shape[1]
        m = coil_maps_and_masks(d["acs"], N, cfg)
        csm_est = m["csm_masked"] if cfg.CSM_MODE_ESTIMATION == "masked" else m["csm_unmasked"]
        csm_fin = m["csm_masked"] if cfg.CSM_MODE_FINAL == "masked" else m["csm_unmasked"]
        ksp = torch.from_numpy(d["ksp"]).to(device)
        lp, lm = torch.as_tensor(d["lines_plus"]), torch.as_tensor(d["lines_minus"])
        slices.append(dict(
            number=number, N=N, ksp=ksp, acs=d["acs"],
            csm=torch.from_numpy(csm_est).to(device), csm_final=torch.from_numpy(csm_fin).to(device),
            mask=torch.from_numpy(m["mask"]).to(device), ghost=torch.from_numpy(m["ghost"]).to(device),
            mask_np=m["mask"], ghost_np=m["ghost"], background=m["background"], rss=m["rss"],
            esp_mask=m["esp_mask"],
            feat_plus=slice_features(ksp, lp), feat_minus=slice_features(ksp, lm)))
        if len(slices) > 1:
            if not (np.array_equal(d["lines_plus"], slices[0]["lines_plus"]) and np.array_equal(d["lines_minus"], slices[0]["lines_minus"])):
                raise ValueError("all slices must use the same RO+/RO- line pattern")
        slices[-1]["lines_plus"], slices[-1]["lines_minus"] = d["lines_plus"], d["lines_minus"]
    N = slices[0]["N"]
    X, Y = coordinate_grids(N, device)
    patterns = make_split_patterns(slices[0]["lines_plus"], slices[0]["lines_minus"],
                                   cfg.N_TRAIN_PATTERNS, cfg.HOLDOUT_FRAC, cfg.SPLIT_SEED)
    assign = np.arange(len(slices)) % cfg.N_TRAIN_PATTERNS
    return dict(slices=slices, N=N, NC=slices[0]["ksp"].shape[0], X=X, Y=Y, patterns=patterns, assign=assign,
                lines_plus=slices[0]["lines_plus"], lines_minus=slices[0]["lines_minus"],
                val_id=cfg.N_TRAIN_PATTERNS, test_id=cfg.N_TRAIN_PATTERNS + 1)
