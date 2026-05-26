# DSAA-ViT

This repository provides representative PAM-4 eye-diagram samples, corresponding labels, and preprocessing scripts for the paper:

**DSAA-ViT: Density-Guided Split Axial Attention for PAM-4 Eye-Diagram Measurement**

## Repository Structure

```text
DSAA-ViT/
├─ scripts/
│  └─ preprocess_eye.py
├─ sample_data/
│  ├─ eyeMatrix_17.csv
│  ├─ label_17.csv
│  ├─ eyeMatrix_821.csv
│  ├─ label_821.csv
│  └─ ...
├─ requirements.txt
└─ README.md
```

## Sample Data

Each sample contains:

- `eyeMatrix_<sample_id>.csv`: exported eye-diagram density matrix.
- `label_<sample_id>.csv`: corresponding eye-parameter labels.

For example:

```text
eyeMatrix_17.csv  -> density matrix of sample 17
label_17.csv      -> label file of sample 17
```

If the input file is named `eyeMatrix.csv`, the sample ID is set to 0.

The nine labels are ordered as: $A_l,\ A_m,\ A_u,\ W_l,\ W_m,\ W_u,\ H_l,\ H_m,\ H_u$.

where `A`, `W`, and `H` denote eye amplitude, eye width, and eye height, respectively. The subscripts `l`, `m`, and `u` denote the lower, middle, and upper eyes. The first column of each label file contains the parameter values, and the second column provides the corresponding units.

## Preprocessing

The preprocessing follows the procedure described in the paper:

1. Zero-density pixels are mapped to white.
2. Nonzero density values are uniformly quantized into 15 levels.
3. Each level is assigned to a predefined RGB colormap.
4. The rendered RGB image is resized to 224 × 224 using bicubic interpolation.
5. A patch-aligned density prior is extracted from the raw density matrix.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```


Run preprocessing for one sample:

```bash
python scripts/preprocess_eye.py --input "sample_data/eyeMatrix_17.csv" --out-dir "outputs"
```

The script generates:

- a resized 224 × 224 RGB input image;
- a rendered RGB eye-diagram figure with colorbar;
- a patch-aligned density prior;
- a density-prior visualization figure.