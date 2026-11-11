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


