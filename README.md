# Zero-Shot-Estimation-of-2D-Phase-for-EPI-Nyquist-Ghost-Correction
To estimate slice- and frame-specific 2D polarity phase errors for EPI Nyquist ghost correction from the acquired scan alone, without training data or navigators, and to test whether tracking this phase matters in resting-state fMRI data with a dual-polarity calibration.

# Data for D2

D2 = 0.8 mm isotropic, single-band EPI, R = 4, GRE-based reference scan, 40 slices, 7T.

Download the files from the shared Google Drive folder
<https://drive.google.com/drive/folders/1E6E8OYuNI7BdjIe9p-60mBBr7UA0HFCA> and put them **in this folder**:

```
data/Slice_1.mat ... data/Slice_40.mat     one file per slice
data/raw_phase_0p8_ipat4.mat               1D navigator phase (optional, only for the comparison)
```

Each `Slice_<n>.mat` contains

| variable  | shape        | meaning                                                         |
|-----------|--------------|-----------------------------------------------------------------|
| `KOE_CC`  | 260 x 260 x 12 | undersampled EPI k-space (readout x phase-encode x coil), coil-compressed |
| `kcal_cc` | 64 x 32 x 12   | fully sampled GRE reference scan (ACS), central k-space          |
| `pos1`    | 1 x 33         | phase-encode lines of the RO+ readouts (1-based)                |
| `pos2`    | 1 x 32         | phase-encode lines of the RO- readouts (1-based)                |

`raw_phase_0p8_ipat4.mat` contains `raw_phase` (260 x 40 x 10, complex): readout x slice x repetition.


# Zero-shot 2D phase correction for EPI Nyquist ghosts — D2 pipeline

This repository contains the pipeline for the **D2** data set of the manuscript
(0.8 mm isotropic single-band EPI, R = 4, GRE-based reference scan, 40 slices, 7T).
It follows the manuscript: weight-shared zero-shot phase estimation (Section 3.4) and
**PEC-GRAPPA** as the final reconstruction (Section 3.5), compared with no correction and with a 1D navigator.

The code is deliberately plain: three small step scripts, one settings file and one file of building blocks.
Nothing is hidden in command-line options; if you want to change something, change `config.py`.

## What it does

EPI reads k-space lines with alternating readout polarity (RO+ / RO−). A phase mismatch between the two
polarities produces a Nyquist (N/2) ghost. The relative phase is modelled as a plane

    Theta(x, y) = p0 + px * x + py * y          (x, y from -1 to 1, in rad)

and **p0, px, py are estimated for every slice directly from the acquired data** (no training database,
no reference images, no navigator):

1. **Step 1** – physics-only screen for the starting point (a GRE reference scan has no polarity information,
   so the baseline is zero and a coarse grid of p0, px picks the starting basin).
2. **Step 2** – one network (`JointPCNet`: shared RO+ / RO− encoders with residual blocks, compact regression head)
   is shared over all slices and trained self-supervised: reconstruct with phase-corrected SENSE from the fit lines,
   predict the held-out lines, minimise 0.7·L2 + 0.3·L1 plus three weak terms (ghost energy, smoothness over slices,
   polarity swap). Then:
   * the **median of four estimates** (one per training split pattern) is taken, and their spread (MAD) is a stability check;
   * a short **per-slice polish** follows and is kept only if the validation loss improves;
   * an **untouched test pattern** reports the loss with and without the 2D phase.
3. **Step 3** – final images with **PEC-GRAPPA** (phase = 0 control, 1D navigator, zero-shot 2D phase), coil images
   combined with ESPIRiT maps, %GSR for every slice, and a PC-SENSE data-consistency check that feeds the `stable` flag.

## How to run it (step by step)

### 0. What you need
* A computer with **Python 3.9 or newer** (check with `python --version`; on some systems the command is `python3`).
* About **2 GB of free disk space** for the 40 slices and the results, and **8 GB RAM** or more.
* A GPU is **optional**. Without a GPU, one slice takes under a minute for all three steps; 40 slices should take roughly one to two hours (estimated from the one-slice timing, not measured)
  (step 2 and the PEC-GRAPPA step are the slow parts). With an NVIDIA GPU (and the CUDA version of PyTorch) it is much faster and is used automatically.

### 1. Get the code
Either download the repository as a zip from GitHub (green **Code** button → **Download ZIP**) and unzip it,
or, if you use git:

```bash
git clone https://github.com/uten178/Zero-Shot-Estimation-of-2D-Phase-for-EPI-Nyquist-Ghost-Correction.git
cd Zero-Shot-Estimation-of-2D-Phase-for-EPI-Nyquist-Ghost-Correction
```

### 2. Install the Python packages (once)
Open a terminal (Windows: "Anaconda Prompt" or "Command Prompt") **inside the repository folder** and run

```bash
python -m venv venv                 # optional but recommended: a clean environment
venv\Scripts\activate               # Windows      (Mac/Linux:  source venv/bin/activate)
pip install -r requirements.txt
```

If you use Anaconda you can instead run `conda create -n zs2dpc python=3.11`, `conda activate zs2dpc`, then `pip install -r requirements.txt`.
For a GPU, install PyTorch with CUDA first, following the selector on <https://pytorch.org>.

### 3. Put the data in the `data` folder
Download the files from the shared Drive folder (link in `data/README.md`) and place them **directly in `data/`**:

```
data/Slice_1.mat
data/Slice_2.mat
...
data/Slice_40.mat
data/raw_phase_0p8_ipat4.mat        (optional: only needed for the 1D-navigator comparison)
```

### 4. (Optional) test on a few slices first
Open `config.py` in any text editor and change the line

```python
SLICES = None            # all slices found in data/
```
to, for example,
```python
SLICES = [18, 25, 38]    # only these slice numbers
```
This finishes in a few minutes on a CPU. Set it back to `None` for the full run. (A single slice also works, but then the network has
nothing to share between slices; use at least a handful.)

### 5. Run everything
```bash
python run_all.py
```
This runs the three steps one after the other. You can also run them one by one (each needs the output of the one before):

```bash
python step1_screen_baseline.py            # ~10 s   -> results/baseline.json
python step2_train_zero_shot.py            # slow    -> results/theta_zs.csv  (estimated p0, px, py per slice)
python step3_reconstruct_and_evaluate.py   # ~15 s per slice -> results/slice_results.csv, images, figures
```

Running from **Spyder**: open the script, set the working directory to the repository folder and press Run (F5).

### 6. What you should see
* Step 1 prints the selected starting point, e.g. `selected starting point [p0, px, py] = [0.0, 8.0, 0.0]`.
* Step 2 prints a line every 5 epochs (training and validation loss; the validation loss should go down), then one line per slice
  showing whether the polish was accepted, and finally the median phase, e.g. `median [p0, px, py] = [-0.17 9.53 -0.07]`.
  Training stops early when the validation loss no longer improves.
* Step 3 prints one line per slice with the %GSR for `none`, `nav1d` and `zs2d`, then a summary. The corrected image (`zs2d`) should have
  a clearly lower %GSR than `none`.
* Everything is written to the folder `results/` (see "Outputs" below). Open `results/figure_slice_<n>.png` to look at the images.
  `docs/example_slice25.png` shows what slice 25 should look like.

### 7. Change a setting
Everything is in `config.py`; each line has a comment. Common changes: `SLICES`, `EPOCHS`, `LEARNING_RATE`, `PEC_KX_HALFWIDTH`,
`PEC_COMBINE` (`"espirit"` or `"rss"`). Run again afterwards; results in `results/` are overwritten.

### Common problems
| message / symptom | what to do |
|---|---|
| `No Slice_*.mat files found` | the `.mat` files are not directly in `data/`, or their names differ (they must be `Slice_<number>.mat`) |
| `FileNotFoundError ... baseline.json` or `theta_zs.npz` | the earlier step was not run; use `python run_all.py` or run the steps in order |
| `ModuleNotFoundError: No module named 'torch'` (or `scipy`, ...) | the packages are not installed in the environment you are using: repeat step 2 |
| `python` is not recognised | try `python3`, or open the Anaconda Prompt |
| `all slices must use the same RO+/RO- line pattern` | the slice files come from different protocols; only D2 slices can be mixed |
| out of memory | close other programs, or process fewer slices at a time (`SLICES = [...]`) |
| step 2 is very slow | expected on a CPU for 40 slices; reduce `EPOCHS` or `SLICES_PER_EPOCH` in `config.py` for a quick look |
| runs, but results look odd for a single slice | with one slice the network cannot share information between slices; use several slices |

## Files

| file | purpose |
|------|---------|
| `config.py` | every setting (paths, slices, epochs, loss weights, PEC-GRAPPA options) |
| `d2_tools.py` | loading, ESPIRiT maps, phase model, PC-SENSE, split patterns, loss, network, PEC-GRAPPA, GSR |
| `step1_screen_baseline.py` | starting point for the phase |
| `step2_train_zero_shot.py` | network training, median/MAD over split patterns, polish, test → `results/theta_zs.csv` |
| `step3_reconstruct_and_evaluate.py` | PEC-GRAPPA images, %GSR table, stable flag, figures |
| `run_all.py` | runs the three steps |
| `data/README.md` | what the data files contain and where to get them |

## Where the manuscript is in the code

| manuscript | code |
|---|---|
| 2.2 phase model, symmetric gauge | `theta_map`, `PCSense` in `d2_tools.py` |
| 3.2 ESPIRiT maps from the ACS | `espirit_maps`, `coil_maps_and_masks` |
| 3.3 zero baseline + physics screen (p0 ∈ {−π/2, 0, π/2}, px −12…12 step 4, py = 0) | `step1_screen_baseline.py`, `SCREEN_*` |
| 3.4 four deterministic patterns, 20 % loss lines, separate validation and test pattern | `make_split_patterns`, `HOLDOUT_FRAC`, `N_TRAIN_PATTERNS` |
| Eqs. (6)–(9) physics loss | `heldout_loss` |
| Eq. (10) ghost term, Eq. (11) swap term, coherence (second difference over slices), Eq. (12) weights 0.02 / 0.005 / 0.01 | `step2_train_zero_shot.py`, `LAMBDA_*` |
| JointPCNet, SharedEncoder (Fig. 1 B, C) | `JointPCNet`, `SharedEncoder` |
| AdamW, weight decay 1e-6, cosine decay, up to 90 epochs, best-validation checkpoint | `step2_train_zero_shot.py`, `EPOCHS`, `LEARNING_RATE` |
| 3.5 PEC-GRAPPA with a matched zero-phase control | `pec_calibrate`, `pec_grappa_recon`, `step3_...py` |
| 3.5 1D navigator comparison | `navigator_fit`, `nav_to_theta_params` |
| 3.6 %GSR = 100 (G − σN) / (S − σN) | `gsr_percent` |

## Example (slice 25, tested)

![example](docs/example_slice25.png)

Left to right: PEC-GRAPPA with phase = 0, with the 1D-navigator phase, and with the zero-shot 2D phase
(same kernel, same code; only the phase differs). Estimated phase of this slice: p0 ≈ −0.17 rad, px ≈ 9.5 rad per half-FOV,
py ≈ −0.07; the four split patterns agree within 0.05 rad (MAD of px). %GSR: 1.74 → **0.53** (2D phase) versus 1.70 (1D navigator).
All numbers of this run are in `docs/example_slice25_results.csv`. This was run on **one slice only**
(on 40 slices the weight sharing and the slice-coherence term come into play; please run the full data set).

## Outputs (folder `results/`)

* `baseline.json` – starting point from step 1
* `theta_zs.csv` – estimated `p0`, `px`, `py` per slice, MAD over the four patterns, `unstable_split`, `polish_accepted`,
  validation loss before/after the polish, test loss without/with phase
* `training_log.csv`, `zs_model.pt` – training history and the trained network
* `slice_results.csv` – everything per slice, including %GSR (`GSR_*`, noise-corrected as in the manuscript) and the plain
  ratio (`GSRplain_*`), PEC-GRAPPA diagnostics and the `stable` flag
* `summary.txt` – mean / median %GSR, number of stable slices
* `images.npz` – PEC-GRAPPA magnitude images for each method
* `figure_slice_<n>.png` – side-by-side images (first 6 slices)

## Conventions

* Arrays are (coil, readout, phase-encode) in the torch code, (readout, phase-encode, coil) in the PEC-GRAPPA code.
* Symmetric gauge: RO+ lines see `exp(-i·Theta/2)`, RO− lines see `exp(+i·Theta/2)`.
* `px` is in rad per half field of view along the readout; divide by π to get the RO+/RO− shift in samples.
* `pos1` lines are RO+ ("ghost" polarity in the PEC-GRAPPA convention), `pos2` lines are RO− (reference polarity).
  The phase given to PEC-GRAPPA is therefore `−Theta` on the ACS pixels.
* Coil maps: ESPIRiT from the single GRE ACS block, support mask = largest eigenvalue > 0.9. **Masked** maps are used while estimating
  the phase, **unmasked** maps for the PC-SENSE check and for the coil combination. PEC-GRAPPA itself uses no coil maps.



## Citation / license

MIT license (see `LICENSE`; put your name in the first line). Please cite the manuscript when it is published.
