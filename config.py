"""
config.py  --  ALL settings of the D2 pipeline are here.
Edit this file, then run  python run_all.py   (or the step scripts one by one).

D2 = 0.8 mm isotropic, single-band EPI, R = 4, GRE-based reference scan (ACS),
     40 slices, 7T, coil-compressed channels (Table 2 of the manuscript).

The numbers below follow the manuscript (Sections 3.3 - 3.5) and the research script
ZS_2dpc_ESPIRiTmask_PECGRAPPA.py.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- where is the data?
DATA_DIR    = os.path.join(HERE, "data")        # put Slice_1.mat, Slice_2.mat, ... here
NAV_FILE    = os.path.join(DATA_DIR, "raw_phase_0p8_ipat4.mat")   # 1D navigator (optional)
RESULT_DIR  = os.path.join(HERE, "results")

# Which slices to process. None = every file called Slice_*.mat in DATA_DIR.
# Example: SLICES = [18, 25, 38]
SLICES = None
SEED   = 1234

# ---------------------------------------------------------------- coil maps (ESPIRiT, from the GRE reference scan)
ESPIRIT_WIDTH       = 32      # calibration block (pixels, square)
ESPIRIT_KERNEL      = 8
ESPIRIT_VECTORS     = 80
ESPIRIT_MASK_THRESH = 0.90    # support mask = largest ESPIRiT eigenvalue > 0.9 ("hard" mask)
CSM_MODE_ESTIMATION = "masked"     # maps used while estimating the phase (steps 1-2)
CSM_MODE_FINAL      = "unmasked"   # maps used for the PC-SENSE check and the coil combination (step 3)
OBJECT_FRAC         = 0.12    # object mask = ACS image > 12 % of its 99.5th percentile (and inside ESPIRiT support)
GHOST_DILATE        = 7       # pixels kept free around the object when defining the "ghost region"

# ---------------------------------------------------------------- self-supervised line split (manuscript 3.4)
HOLDOUT_FRAC     = 0.20       # 20 % of the lines of each polarity are the "loss lines"
N_TRAIN_PATTERNS = 4          # four deterministic training patterns ...
SPLIT_SEED       = 20260903   # ... plus one validation and one test pattern (same seed -> same patterns)

# ---------------------------------------------------------------- step 1: starting point (manuscript 3.3)
# A GRE reference scan has no polarity information, so the baseline is zero and a coarse
# physics-only screen picks the starting basin of the shared phase.
SCREEN_P0      = [-3.14159 / 2, 0.0, 3.14159 / 2]      # rad
SCREEN_PX      = [-12, -8, -4, 0, 4, 8, 12]            # rad per half-FOV (readout)
SCREEN_PY      = [0.0]                                 # rad per half-FOV (phase encoding)
SCREEN_NSLICES = 8            # slices used for the screen (spread over the stack)
SCREEN_CG      = 2

# ---------------------------------------------------------------- step 2: weight-shared zero-shot network (manuscript 3.4)
BASE_FILTERS     = 32
RESIDUAL_BLOCKS  = 3
HEAD_HIDDEN      = 96
RESIDUAL_SCALES  = [0.75, 3.0, 2.0]     # largest slice-specific change   (p0, px, py)
GLOBAL_SCALES    = [3.14159, 14.0, 8.0] # largest shared change           (p0, px, py)

EPOCHS           = 90                   # "up to 90 epochs"; the best validation checkpoint is kept
SLICES_PER_EPOCH = 20                   # random subset of slices per epoch
BATCH_SLICES     = 4
LEARNING_RATE    = 2e-4                 # AdamW + cosine decay. (The manuscript text says 1e-4; the research script used 2e-4.)
WEIGHT_DECAY     = 1e-6
GRAD_CLIP        = 1.0
TRAIN_CG_ITERS   = 4                    # conjugate-gradient iterations inside a training step
LAMBDA_GHOST     = 0.02                 # weights of the three weak extra loss terms, Eq. (12)
LAMBDA_COH       = 0.005                # (coherence over neighbouring slices, second difference)
LAMBDA_SWAP      = 0.01                 # (polarity swap)
REG_EVERY        = 2                    # coherence and swap terms are evaluated every 2nd step

VAL_EVERY          = 5
VAL_N_SLICES       = 10                 # validation uses this many slices spread over the stack
VAL_CG_ITERS       = 8
EARLY_STOP_CHECKS  = 6                  # stop if validation did not improve for 6 checks ...
MIN_EPOCHS         = 25                 # ... but not before this epoch

MAD_WARN = [0.35, 0.75, 0.60]           # rad; a slice is "unstable" if the spread over the 4 patterns exceeds this (p0, px, py)

# per-slice polish of the three numbers after training (accepted only if the validation loss improves)
REFINE_STEPS           = 18             # 0 = off
REFINE_LR              = 0.025
REFINE_CG_ITERS        = 5
REFINE_GHOST_WEIGHT    = 0.01
REFINE_VAL_EVERY       = 6
REFINE_VAL_CG_ITERS    = 8
REFINE_BOUNDS          = [3.14159, 18.0, 10.0]   # range of (p0, px, py) during the polish
MIN_REL_VAL_IMPROVEMENT = 1e-4

TEST_CG_ITERS    = 10                   # untouched test pattern: reported, never used for training or selection

# ---------------------------------------------------------------- step 3: final reconstruction (manuscript 3.5)
PCSENSE_LAMBDA   = 1e-4                 # small Tikhonov weight of PC-SENSE (estimation AND the check images)
FINAL_CG_ITERS   = 45                   # PC-SENSE check images (with phase / without phase)
FINAL_NO_PC_CG_ITERS = 35

# PEC-GRAPPA (a k-space method; the phase estimate only enters here, never in the training)
PEC_LAMBDA           = 3e-2     # ridge weight (relative to the largest eigenvalue of the calibration matrix)
PEC_KX_HALFWIDTH     = 2        # kernel is (2*halfwidth + 1) points wide along readout (kx)
PEC_KX_AUTO          = True     # widen the window if max|px| needs it (RO+/RO- shift = |px|/pi samples)
PEC_KX_MARGIN        = 1
PEC_KX_HALFWIDTH_MAX = 6
PEC_MIN_EQ_PER_UNKNOWN = 2.0    # shrink the window if the ACS cannot determine the weights
PEC_N_SOURCE_MISSING  = 4       # source lines for a missing line
PEC_N_SOURCE_ACQUIRED = 3       # source lines for an acquired line
PEC_FILL_ACQUIRED    = True     # where PEC cannot predict, keep the acquired sample
PEC_COMBINE          = "espirit"   # coil combination of the PEC-GRAPPA coil images: "espirit" (matched filter) or "rss"
NAV_REP              = 0        # which repetition of the navigator file belongs to the data
