# VALID-Aug: leakage-aware evaluation of data augmentation for flow-based NIDS

Code, configurations and results for the article

> H.-T. Nguyen, D.-T. Huynh, T.-T. Huynh. *When does data augmentation help flow-based network intrusion detection? A leakage-aware study of semantic validity and class trade-offs.* Manuscript, 2026.

The study compares nine published augmentation methods and a constraint-aware diffusion augmenter (**VALID-Aug**, used as a probe) on five flow datasets, under four splitting protocols and six classifier families. Every table and figure of the article can be regenerated from the files in `runs/` (Section 9).

The pipeline is resumable and organised in phases A–H (Section 7). It can be tested end to end on synthetic toy data (`scripts/make_toy_data.py`); toy results only show that the code runs and carry no meaning.

---

## 1. Installation

```bash
conda create -n validaug python=3.11 -y && conda activate validaug
pip install -r requirements.txt
# Pascal/Volta GPUs (P40 sm_61, V100 sm_70) need a CUDA 12.6 build of PyTorch, e.g.
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # same GPU numbering as nvidia-smi
```

Smoke test (CPU or GPU, a few minutes):

```bash
python scripts/make_toy_data.py
python scripts/prepare_data.py --datasets toy_nf toy_nf_b toy_cic
python scripts/run_grid.py --exp configs/exp_smoke.yaml
```

## 2. Data and preprocessing

The datasets are not redistributed. Download links, the expected folder layout and the requested citations are given in [`data/README.md`](data/README.md); the paths are set in `configs/datasets.yaml`. Two feature families are supported:

| Family | Datasets | Multi-class label | Time (for splitting) | Host (group split) |
|---|---|---|---|---|
| `netflow` | NF-UNSW-NB15-v3, NF-ToN-IoT-v3, NF-BoT-IoT-v3, NF-CSE-CIC-IDS2018-v3 (v2 also readable) | `Attack` | `FLOW_START_MILLISECONDS` (v3 only) | `IPV4_SRC_ADDR` |
| `cicflowmeter` | CIC-IDS2017, corrected release (Engelen et al.); original CIC-IDS2017/2018 | `Label` | `Timestamp` | `Src IP` |

### Data-independent steps (`validaug/io.py`, applied before splitting)

1. Column names are normalised to snake_case. CICFlowMeter aliases are mapped to canonical names (leading spaces in the original 2017 release, `Total Fwd Packet` vs `Total Fwd Packets`, parquet names). The duplicated `Fwd Header Length.1` is dropped.
2. Labels `BENIGN`, `Benign` and `normal` become `Benign`. The "… - Attempted" labels of the corrected release follow `attempted: drop | benign | keep` (default `drop`). Optional `label_map` and `drop_classes` per dataset.
3. Identifiers are removed to avoid leakage: source and destination IP, source port, timestamps, Flow ID, `DNS_QUERY_ID`. Time and host are kept in the auxiliary columns `__time` and `__host`, which are used for splitting only and never as features.
4. The destination port becomes categorical (a fixed list of common ports plus three IANA ranges; fixed in code, not learned).
5. TCP flag bitmasks (`TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS`) are split into eight binary bits (FIN … CWR), so that constraints such as "only TCP flows carry flags" can be written explicitly.
6. `L7_PROTO` values such as `7.178` are mapped to the master protocol (`7`); categorical `6.0` and `6` are unified.
7. Infinite values are kept at this stage and replaced later by the finite training maximum or minimum; counts of inf and NaN per column are reported. Constant columns and classes with fewer than `min_class_count` samples are dropped and reported.
8. An optional per-class cap `max_per_class` is applied to very large datasets; `__time` is preserved.
9. Label conflicts (identical feature vectors with different labels) are only reported here; they are resolved after splitting, on the training partition (Section 3).

Dataset-specific decisions (`configs/datasets.yaml`):

- **NF-v3**: `SRC_TO_DST_SECOND_BYTES` and `DST_TO_SRC_SECOND_BYTES` are dropped. They are bytes divided by duration, which gives inf or NaN for zero-duration flows, and the information is already contained in `IN_BYTES`, `OUT_BYTES` and `FLOW_DURATION`.
- **NF-CSE-CIC-IDS2018-v3**: `FTP-BruteForce` and `DoS_attacks-SlowHTTPTest` are dropped. These two attacks were erroneously collected and their traffic contains only SYN and RST packets (Liu and Patras, 2022), in line with the errors documented for CSE-CIC-IDS2018 by Liu et al. (2022). The classes are removed before the audit (`classes_dropped_config` in `audit.json`: 386,720 and 105,550 flows).

Each dataset receives an `audit.json` (rows, columns, types, duplicates and label conflicts per class, inf/NaN per column, class distribution, availability of time and host). Table 2 of the article is built from these files.

### Fitted steps (training partition only)

- `encoder.py`, invertible encoder $`T`$:
  - numerical features: $`+\infty \to`$ training maximum, $`-\infty \to`$ training minimum, NaN $`\to`$ training median, then signed log and a quantile-normal transform (custom implementation with a differentiable PyTorch counterpart; integer features are dequantised during fitting so that guidance gradients do not vanish);
  - categorical features: 32 most frequent values plus `__other__`, one-hot coded as $`\pm 1`$;
  - flag bits: $`\pm 1`$.
- `constraints.py`: constraint thresholds (lower and upper bounds of bytes and packets), retained templates and mined invariants.

## 3. Splitting protocols (`validaug/splits.py`)

| Protocol | Description | Role |
|---|---|---|
| `random_naive` | stratified random 60/20/20, no conflict resolution, no deduplication | common practice in the literature |
| `random` | stratified random 60/20/20 after deduplication | reference for measuring inflation |
| `temporal_stratified` | within each class, flows sorted by time and cut | main leakage-aware protocol |
| `group` | within each class, no source host in two partitions; classes with fewer than three source hosts fall back to `temporal_stratified` (listed as `group_fallback_classes` in `runs/<phase>/splits/*.json`) | host-disjoint protocol |
| `temporal_global` | one global temporal cut | reports the classes missing from a partition (not used in the article) |
| cross-dataset | train and validate on A, test on all of B (shared features), binary or grouped labels | generalisation |

After splitting (all protocols except `random_naive`):

1. Label conflicts in the training partition are resolved by majority vote within each group of identical feature vectors (ties: the group is dropped). Validation and test are left unchanged, so the label ambiguity met in deployment is preserved.
2. Validation and test rows identical (features and label) to a training row are removed, and the training partition is deduplicated.

The difference `random_naive` → `random` measures duplicate leakage. The difference `random` → `temporal_stratified` or `group` combines the removal of temporal or host leakage with the genuine shift between periods or hosts; Section 8.3 of the article measures this shift.

Counts are written to `runs/<phase>/splits/*.json`. A protocol that needs a missing column (no timestamp or no host) is skipped with a warning.

**Scenarios.**
- `rare`: keep 1% or 5% of the rare classes in the training partition.
- `holdout`: remove one attack type from training and validation and measure its detection on test. Class names are matched case-insensitively; lists and `auto` are accepted.
- Any scenario may add `task: binary` (Benign vs. Attack on the same split; labels are merged and duplicate removal is re-run, because flows that differed only by attack type become exact duplicates). Scenarios may also restrict `methods:` and `classifiers:`.

**Tasks.** The main task is multi-class (Benign plus attack types). Binary metrics (`binary_tpr`, `binary_fpr`, `binary_roc_auc`, `tpr_at_fpr1`) are derived from every multi-class run (attack = any non-Benign class). Runs trained on binary labels come from `task: binary` scenarios (phases B and D) and from `labels: binary` cross-dataset pairs (phase F).

## 4. VALID-Aug (Section 4 of the article and Section S1 of the supplementary material)

| Component | File | Notes |
|---|---|---|
| Class-conditional tabular diffusion (classifier-free guidance, EMA, class-balanced sampling) | `diffusion.py` | categorical and bit features are diffused as Gaussians on the $`\pm 1`$ codes |
| Constraint energy $`V(x)`$ and projection $`\Pi_{\mathcal{C}}`$ | `constraints.py` | templates plus thresholds calibrated on training data; a template is kept only if at most 0.5% of the real training flows violate it |
| Boundary reward $`B(x,y)`$ | `guidance.py` | ensemble of three MLPs (bootstrap samples, noise-augmented training) |
| Three-criterion filter | `filtering.py` | realistic (rank-based C2ST against real flows of the same class), not memorised (distance to closest record), useful (margin) |
| Per-class budget | `budget.py` | ratio $`r_c`$ searched on validation data (Optuna TPE) |

**Energy and reward.** With $`\rho_k`$ the violation of constraint $`k`$ in the original feature space and $`m_y^e(x)`$ the margin of class $`y`$ for ensemble member $`e`$:

```math
V(x)=\sum_{k} w_k\,\rho_k\bigl(T^{-1}(x)\bigr)^2,
\qquad
B(x,y)=-\frac{1}{E}\sum_{e=1}^{E}\bigl(m_y^e(x)-\delta\bigr)^2,\quad \delta>0,
```

with $`w_k=1`$ for hard constraints and $`w_k=0.5`$ for mined invariants.

**Guidance as implemented.** At step $`t`$ the denoised estimate is

```math
\hat\epsilon=(1+w)\,\epsilon_\theta(x_t,t,y)-w\,\epsilon_\theta(x_t,t,\varnothing),
\qquad
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}} .
```

It is moved along normalised gradients ($`K=3`$ inner steps), and the noise estimate is recomputed before the ancestral update:

```math
\hat x_0' = \hat x_0-\lambda_c(t)\,\frac{\nabla V(\hat x_0)}{\lVert\nabla V(\hat x_0)\rVert}
              +\lambda_b(t)\,\frac{\nabla B(\hat x_0,y)}{\lVert\nabla B(\hat x_0,y)\rVert},
\qquad
\tilde\epsilon=\frac{x_t-\sqrt{\bar\alpha_t}\,\hat x_0'}{\sqrt{1-\bar\alpha_t}} .
```

The schedules are

```math
\lambda_c(t)=\lambda_c\,(1-t/T)^{\gamma},
\qquad
\lambda_b(t)=\lambda_b\,\mathbf{1}\!\left[t_1\le t/T\le t_2\right].
```

Implementation notes:
- Gradients are taken with respect to $`\hat x_0`$; the Jacobian of $`\epsilon_\theta`$ is not propagated.
- Because the gradients are normalised, $`\lambda_c`$ and $`\lambda_b`$ are step lengths in the encoded space. The option `max_grad_norm` sets the step norm; it is not a clipping threshold.
- The mechanism is guidance in $`\hat x_0`$-space in the spirit of DPS and Universal Guidance. What is specific to this work is the NetFlow energy $`V`$, the boundary reward $`B`$ and their coupling with projection, filter and budget, not the guidance mechanism itself.

**Budget search.** For each minority class the ratio $`r_c`$ is chosen from the grid $`\{0,\tfrac14,\tfrac12,1,2,5,\beta\}`$. The number of added samples is

```math
n_c=\min\bigl(r_c N_c,\ \lvert\mathrm{pool}_c\rvert\bigr)\ \le\ \min\bigl(\beta N_c,\ \texttt{pool\_cap}\bigr).
```

The objective is the mean validation macro-F1 of two proxy classifiers (XGBoost and a small MLP). A configuration is adopted only if it beats "no augmentation" by at least `budget_min_gain`; otherwise no sample is added.

## 5. Methods (`validaug/augmenters.py`)

- **Baselines**: `none`, `ros`, `smote`, `borderline_smote`, `adasyn`, `smote_enn`, `ctgan`, `tvae` (`pip install ctgan`; trained on a class-capped subsample of at most 50k rows that keeps the rare classes), `tabddpm` and `tabsyn` (official code, see below), `diffusion` (our class-conditional diffusion without guidance, i.e. the backbone of VALID-Aug) and `diffusion_proj` (post-hoc projection only).
- **Probe**: `validaug`.
- **Ablations**: `va_no_gc`, `va_no_gb`, `va_no_proj`, `va_no_filter`, `va_default_budget`, `va_no_delta`, `va_guidance_only`.

**Shared budget.** All baselines add, for every class $`c`$ other than the largest,

```math
n_c=\min\bigl(\max(N_c,\ \alpha N_{\max}),\ \beta N_c\bigr)-N_c,\qquad \alpha=0.5,\ \beta=10 .
```

VALID-Aug adds at most $`\min(\beta N_c,\ \texttt{pool\_cap})`$ samples per class. No class weights are used. The validation partition always contains real flows only. VALID-Aug uses it for guidance tuning (phase C), the budget search and early stopping; the baselines use it for early stopping only. The test partition is never used for any decision. If the shared budget gives zero samples to every class of a cell (for example, a binary task in which the attack class is larger than $`\alpha`$ times the benign class), the runner runs `none` only and logs a warning.

**Comparable violation rates.** The raw output of every generator goes through the same decoding as the diffusion samples (`encoder.decode_like_generator`: integer columns rounded, values clipped to the training range, unknown categories mapped to `__other__`) before `validity_raw` is computed. Results written by an older code version can be recomputed with

```bash
python scripts/fix_validity.py --exp configs/phaseD_main.yaml --results runs/phaseD_main/results.jsonl
```

**Training-set size of the deep generators.** CTGAN, TVAE, TabSyn and TabDDPM are trained on the class-capped subsample of at most 50k rows, our diffusion model on up to `max_train_rows` (300k). Phase G contains a control in which the diffusion model is also trained on 50k rows (`configs/phaseG_diff50k.yaml`).

### Official TabSyn and TabDDPM

Both run from the official repository `amazon-science/tabsyn` (ICLR 2024; it also ships the TabDDPM baseline used in the TabSyn paper), pinned to a tested commit, in a separate conda environment, and are driven by `validaug/external.py`.

Clone the official repository, pin it to the tested commit and install it in its own environment, then set in `aug.external` of the phase configuration:

```yaml
aug:
  external:
    python: <path to that environment>/bin/python
    tabsyn_root: <path to the cloned tabsyn repository>
```

- TabDDPM runs class-conditionally (`is_y_cond = true`, `external.tabddpm_y_cond`), its native mode. TabSyn is unconditional (the label is a column) and per-class quotas are met by rejection sampling (`max_rounds`); any shortfall is recorded in `aug_info.shortfall`.
- In our runs the official code received compatibility patches only: epochs set through environment variables with the official defaults kept, `verbose=` removed from `ReduceLROnPlateau`, `map_location` for checkpoints, and the device passed to the sampler instead of a hard-coded `cuda:0`. Models and objectives are unchanged.
- Training data: the same class-capped subsample as CTGAN and TVAE, in the original feature space, with the label as a categorical column.
- Checkpoints stay in the TabSyn repository. Generated pools are cached in `runs/cache/.../external/` (fingerprinted), so all classifiers and phases reuse the same synthetic data. Each external command is logged to `runs/cache/.../external/<name>.log`.

**Classifiers**: `xgboost`, `rf`, `mlp`, `cnn1d`, `lstm`, `ft_transformer`.

## 6. Evaluation on four axes (`validaug/metrics.py`)

| Axis | Metrics |
|---|---|
| Utility | macro-F1, weighted-F1, MCC, macro PR-AUC, per-class recall and F1, rare-F1, macro-F1 over attack classes and over common attack classes, binary TPR/FPR, TPR at 1% FPR, hold-out detection |
| Trade-off | per-class F1 change relative to `none`: number of classes gaining or losing more than 0.01, mean gain and loss, worst drop and most-hurt class (`tradeoff_*.csv`) |
| Fidelity | mean Wasserstein distance, correlation difference, C2ST AUC |
| Validity | `validity_raw` (violation rate of the generator output before projection), `validity_final` |
| Privacy | distance to closest record (DCR), DCR ratio, distance-based membership-inference AUC |
| Cost | augmentation, training and inference time; GPU energy (NVML) |

`real_test_violation` is the violation rate of the real test flows; it shows that the constraints hold on real data.

## 7. Running the phases

| Phase | Configuration | Content | Article |
|---|---|---|---|
| A | `datasets.yaml` | preprocessing and audit | Table 2 |
| B | `phaseB_protocol.yaml` | protocol-induced inflation, oversamplers under each protocol | Sections 6.1–6.2 |
| C | written by `tune_guidance.py` | guidance hyper-parameters tuned on validation | Table S4 |
| D | `phaseD_main.yaml` | main comparison, scarce regimes, binary task, budget reliability | Sections 6.3, 6.4, 6.6 |
| E | `phaseE_ablation.yaml` | ablation | Section 6.5 |
| F | `phaseF_generalization.yaml` | hold-out attacks and cross-dataset transfer | Section 6.7 |
| G | `phaseG_classweights.yaml`, `phaseG_diff50k.yaml`, `phaseG_nocap.yaml` | class weights, generator trained on 50k rows, uncapped data | Sections 7.1–7.3 |
| H | `phaseH_tabsota.yaml` | official TabDDPM and TabSyn | Section 7.4 |

```bash
./run_all.sh A   # data preparation and audit
./run_all.sh B   # protocol-induced inflation
./run_all.sh C   # tune lambda_c, lambda_b, delta, t1, t2 on validation per dataset -> runs/tune_<ds>.yaml
                 # scripts/collect_tuning.py then writes them to aug_per_dataset in the D/E/F configurations
./run_all.sh D   # main comparison (heaviest)
./run_all.sh E   # ablation (see Known issues)
./run_all.sh F   # hold-out and cross-dataset
# phases G and H
python scripts/run_grid.py --exp configs/phaseG_classweights.yaml --out runs/phaseG_classweights
python scripts/run_grid.py --exp configs/phaseG_diff50k.yaml      --out runs/phaseG_diff50k
python scripts/run_grid.py --exp configs/phaseG_nocap.yaml        --out runs/phaseG_nocap
python scripts/run_grid.py --exp configs/phaseH_tabsota.yaml      --out runs/phaseH_tabsota
```

Options:
- **Two GPUs**: `CUDA_VISIBLE_DEVICES=1 python scripts/run_grid.py --exp configs/phaseD_main.yaml --only-datasets nf_unsw_v3 nf_ton_v3 --out runs/phaseD_main_gpu1`, and the remaining datasets on the other GPU.
- **Overrides**: `--set aug.validaug.lambda_c=0.5 --out runs/x`. Values in `aug_per_dataset` take precedence over `--set` (see Known issues).
- **Per-dataset parameters**: `aug_per_dataset: {nf_ton_v3: {validaug: {lambda_c: ..., ...}}}` is deep-merged into `aug` for that dataset; a cross-dataset cell `A->B` uses the block of A. Phase C writes only the five guidance keys there; `pool_cap` and `oversample` of phase C are not carried over to phase D.
- **Resume**: re-run the same command; cells already present in `results.jsonl` are skipped. Rows are keyed by (dataset, split, scenario, seed, method, classifier) only. After a code change that affects a classifier or an augmenter, delete the affected lines (or the whole file) before resuming, because `analyze.py` cannot distinguish old from new results.
- **Scenario keys**: `methods:` (whitelist), `exclude_methods:` (blacklist), `classifiers:`, `splits:`, `datasets:`, `task: binary`.
- **Diffusion cache** (`cache_root: runs/cache`): one model per cell, stored as `runs/cache/<ds>/<split>/<scenario>/seed*/diffusion_<fingerprint>.pt`. It is shared by every diffusion-based method (`diffusion`, `diffusion_proj`, `validaug` and the ablations) and by phases C–G. The generator is therefore trained once per cell, and all variants use the same generator, so ablation differences come only from guidance, filter and budget. The fingerprint covers the training data, classes, seed and diffusion hyper-parameters; changing any of them triggers retraining.

## 8. Outputs

Per run directory `runs/<phase>/`:

| File | Content |
|---|---|
| `results.jsonl` | one line per (dataset, split, scenario, seed, method, classifier) with all metrics, times, filter and budget information |
| `run.log` | full log |
| `config_used.json` | effective configuration, including `--set` overrides |
| `environment_*.json` | Python and package versions, PyTorch/CUDA/cuDNN, GPUs, driver, git commit, `pip freeze` |
| `splits/*.npz`, `splits/*.json` | exact train/validation/test indices and duplicate/conflict removal counts |
| `constraints/*.json` | fitted constraint set per cell (thresholds, rejected templates with their training violation rates, real-test violation, encoder categories) |
| `synthetic/<cell>/<method>.parquet` | up to `save_synthetic` synthetic flows in the original space, and `<method>__raw.parquet` before projection |
| `analysis/` | tables (CSV, LaTeX), critical-difference diagrams, Wilcoxon–Holm tests, inflation and four-axis tables |

Reproducibility archive:

```bash
python scripts/make_artifact.py --out artifact                    # code, configs, audits, results, splits, constraints, analysis
python scripts/make_artifact.py --out artifact --with-synthetic   # also the stored synthetic flows
```

It writes `MANIFEST.json` (SHA-256 of every file) and a `.tar.gz`. Datasets are never included.

## 9. Analysis and the tables and figures of the article

| Script | Output | Article |
|---|---|---|
| `scripts/analyze.py` | per-phase tables, critical-difference diagrams, Wilcoxon–Holm tests, inflation, trade-off and four-axis tables | Section 6 |
| `scripts/analyze_controls.py` | class weights, generator size, uncapped data, TabDDPM/TabSyn, two-sample diagnosis | Section 7 |
| `scripts/mechanism_analysis.py` | label-conflict ceiling, class overlap, train–test shift (`runs/mechanism/`) | Section 8 |
| `scripts/make_feature_table.py` | feature lists | Tables S1–S2 |
| `scripts/make_figures.py` | Figures 2–8 and S1 | all data figures |
| `scripts/robustness_ci.py` | bootstrap intervals over cells, blocks and whole datasets; seed standard deviation (`runs/robustness_ci.txt`) | Section 6.1, Table 7, Section 9.4 |
| `scripts/verify_tables.py` | recomputes the result tables from `results.jsonl` and compares them cell by cell with the LaTeX source (`runs/verify_report.md`) | Tables 5, 6, 8–10, 12, 13 and S5–S11 |

```bash
python scripts/verify_tables.py --runs runs --tex manuscript.tex supplementary.tex --out runs/verify_report.md
python scripts/make_figures.py  --runs runs --mechanism runs/mechanism --out figures
```

`make_figures.py` needs only pandas, numpy, scipy and matplotlib; it trains no model and reads no dataset.

## 10. Known issues

- `run_all.sh E` contains a sensitivity loop that passes `--set aug.validaug.lambda_c=...`. The per-dataset block `aug_per_dataset` overrides these values, so all runs in `runs/phaseE_sens/` are identical. They are not used in the article. Remove the loop, or let `--set` take precedence, before reusing it.
- The first-party `diffusion` baseline is not the official TabDDPM. The official TabDDPM and TabSyn are run through `validaug/external.py` with one seed (Section 7.4 of the article).

## 11. Layout

```
validaug/   schema.py io.py splits.py encoder.py constraints.py diffusion.py guidance.py filtering.py
            budget.py augmenters.py external.py classifiers.py metrics.py runner.py stats.py utils.py
scripts/    make_toy_data, prepare_data, run_grid, tune_guidance, collect_tuning, collect_sensitivity,
            analyze, analyze_controls, mechanism_analysis, make_feature_table, make_figures,
            robustness_ci, verify_tables, fix_validity, make_artifact, Figure_1.tex
configs/    datasets.yaml, base_gpu.yaml, exp_smoke.yaml, phaseB/D/E/F_*.yaml, phaseG_*.yaml, phaseH_tabsota.yaml
figures/    Figure_2.pdf ... Figure_8.pdf, Figure_S1.pdf
run_all.sh
```

## References

Liu, H., Patras, P., 2022. NetSentry: A deep learning approach to detecting incipient large-scale network attacks. Computer Communications 191, 119–132. https://doi.org/10.1016/j.comcom.2022.04.020

Liu, L., Engelen, G., Lynar, T., Essam, D., Joosen, W., 2022. Error prevalence in NIDS datasets: A case study on CIC-IDS-2017 and CSE-CIC-IDS-2018. In: IEEE Conference on Communications and Network Security (CNS), pp. 254–262. https://doi.org/10.1109/CNS56114.2022.9947235
