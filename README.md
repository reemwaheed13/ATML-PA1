# ATML-PA1 — Learning Beyond the IID, Closed-Set Setting

Programming Assignment 1 (EE-5102 / CS-6304). Four tasks on inductive biases and
representations (Task 1), unsupervised domain adaptation (Task 2), domain
generalization (Task 3), and open-set recognition (Task 4).

I followed the repository structure outlined in the manual to the best of my ability.
The report is entirely self-written; AI was used only for LaTeX formatting of tables
and figures. I like to write, and my sentences tend to be on the longer side.

---

## Environment

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies (see `requirements.txt` for exact versions): PyTorch, torchvision,
`open_clip_torch`, scikit-learn, umap-learn, scipy, pandas, matplotlib, PyYAML, tqdm,
Pillow. A CUDA GPU is recommended; all scripts fall back to CPU.

Every experiment uses **seed 6304**. Frozen splits/indices are stored under
`shared/splits/` and `results/*/*_splits.json` so results are reproducible.

## Datasets

Raw datasets are **not** committed. Obtain them once and point the scripts at them:

- **STL-10** (Task 1) — downloaded automatically by `torchvision` into `./data`.
- **CIFAR-10 / CIFAR-100** (Task 4) — downloaded automatically by `torchvision`.
  CIFAR-100 is used **evaluation-only** as the unknown set.
- **PACS** (Tasks 2 & 3) — download separately and place the four domain folders
  (`photo`, `art_painting`, `cartoon`, `sketch`) under one root. Pass it with
  `--pacs_root <path>` or set the `PACS_ROOT` environment variable.

## Repository structure

```
common/          seed + shared metric helpers
shared/          PACS dataset, protocol, and frozen splits (Tasks 2 & 3)
task1/           inductive biases: data/ models/ analysis/ scripts/
task2/           UDA: configs/ models/ methods/ evaluation/ train.py evaluate_final.py
task3/           DG:  configs/ methods/ evaluation/ train.py evaluate_sketch.py
task4/           OSR: configs/ data/ models/ methods/ scores/ evaluation/ train.py ...
results/         machine-readable results (JSON + CSV) for every task
report/figures/  figures used in the report
requirements.txt
```

---

## Reproducing each task

All commands are run from the repository root. Target/unknown labels are read **only**
at the final evaluation step of each task, behind an explicit `--confirm_*` flag, and
never influence training or model selection.

### Task 1 — Inductive biases and representations (STL-10)

```bash
python -m task1.data.make_subset            # frozen 500-image subset + train/val/test split
python -m task1.scripts.run_task1           # cache features, train 3 heads, clean baseline
python -m task1.data.make_cue_conflicts_v2  # generate cue-conflict (stylized) images
python -m task1.analysis.evaluate_bias      # interventions + translation curve (head predictions)
python -m task1.analysis.feature_similarity # cosine representation stability
python -m task1.analysis.representation     # t-SNE clean-vs-transformed figures
python -m task1.analysis.zeroshot_bias      # CLIP zero-shot vs trained head (parity-checked)
```

### Task 2 — Unsupervised Domain Adaptation (PACS, target = Sketch)

```bash
python -m shared.pacs_protocol --root <pacs>              # build frozen splits (once; shared with Task 3)
python -m task2.train --config configs/source_only.yaml   # ERM baseline (reused by Task 3)
python -m task2.train --config configs/dan.yaml
python -m task2.train --config configs/dann.yaml
python -m task2.train --config configs/cdan.yaml
python -m task2.train --config configs/dan_lambda0.1.yaml # + dan_lambda10.yaml (design study)
# Disclosed convergence variants (gated off by default; see report):
python -m task2.train --config configs/dan_warmup.yaml
python -m task2.train --config configs/dann_fix.yaml
python -m task2.train --config configs/cdan_fix.yaml
python -m task2.evaluate_final --confirm_frozen --pacs_root <pacs>   # reads Sketch labels once
```

### Task 3 — Domain Generalization (PACS, unseen = Sketch)

```bash
# ERM is NOT retrained: Task 3 reuses checkpoints/task2/source_only/best.pt.
python -m task3.train --config configs/dan_dg.yaml            # DAN-DG (collapses at lambda=1; reported as-is)
python -m task3.train --config configs/dan_dg_lambda0.1.yaml  # + dan_dg_lambda10.yaml (design study)
python -m task3.train --config configs/sam.yaml               # SAM, rho=0.05
python -m task3.train --config configs/dan_dg_warmup.yaml     # disclosed convergence variant
python -m task3.evaluate_sketch --confirm_frozen --pacs_root <pacs>   # reads Sketch labels once
```

### Task 4 — Open-Set Recognition (known = CIFAR-10, unknown = CIFAR-100)

```bash
python -m task4.data.make_splits --data_root <d>
python -m task4.train --config configs/vanilla.yaml          # train FIRST
python -m task4.train --config configs/gcsc.yaml
python -m task4.train --config configs/proser.yaml           # initialized from vanilla best.pt
python -m task4.extract_outputs --model vanilla --confirm_unknowns --data_root <d>
python -m task4.extract_outputs --model gcsc   --confirm_unknowns --data_root <d>
python -m task4.extract_outputs --model proser --confirm_unknowns --data_root <d>
python -m task4.evaluate_osr                                 # MSP / MLS / Energy / Mahalanobis + OSR metrics
```

## Results

All reported numbers trace to machine-readable files under `results/<task>/`
(`*_final.json` and the flat `*_summary.csv` rollups). Figures used in the report are
in `report/figures/`.

---

## External code and attribution

This repository implements each method's objective from the referenced papers; the
following external assets, libraries, and algorithms are reused and attributed here.

**Pretrained models / weights**
- torchvision ResNet-50 (`IMAGENET1K_V2`), ResNet-18 (`IMAGENET1K_V1`), ViT-B/16 (`IMAGENET1K_V1`).
- OpenAI CLIP ViT-B/32 via `open_clip_torch` (`pretrained='openai'`).

**Datasets** — STL-10, CIFAR-10, CIFAR-100 via `torchvision.datasets`; PACS via `ImageFolder`.

**Libraries** — PyTorch, torchvision, NumPy, SciPy, pandas, matplotlib; scikit-learn
(t-SNE and the logistic-regression separability probes); umap-learn.

**Algorithms / methods implemented from the literature**
- Cue-conflict generation: neural style transfer using VGG-19 features (torchvision),
  AdaIN content preservation (Huang & Belongie, 2017) and Gram-matrix style loss
  (Gatys et al., 2016).
- DAN / MMD alignment (Long et al., 2015); DANN with gradient-reversal (Ganin et al., 2016);
  CDAN (Long et al., 2018).
- SAM — Sharpness-Aware Minimization (Foret et al., 2021).
- OSR: MSP (Hendrycks & Gimpel, 2017); Energy / max-logit (Liu et al., 2020);
  Mahalanobis distance score (Lee et al., 2018); PROSER (Zhou et al., 2021); GCSC and
  the OSR evaluation protocol per the assignment specification.

All method implementations, training loops, and evaluation code in `task1/`–`task4/`,
`common/`, and `shared/` were written for this assignment.
