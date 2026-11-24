# Data

The datasets are not redistributed with this repository. Download them from their providers, place the CSV files as shown below, and run the preparation step. The paths are declared in `configs/datasets.yaml`.

## Expected layout

```
data/
  raw/
    NF-UNSW-NB15-v3/            *.csv
    NF-ToN-IoT-v3/              *.csv
    NF-BoT-IoT-v3/              *.csv
    NF-CSE-CIC-IDS2018-v3/      *.csv
    CSE-CIC-IDS-2017-improved/  *.csv
  processed/                    written by scripts/prepare_data.py
```

Every CSV file in a folder is read (`*.csv`), so the original file names can be kept. Only the CSV release of the corrected CIC-IDS2017 is supported: the loader needs its `Timestamp` and `Src IP` columns for the temporal and host-disjoint splits.

## Sources

| Folder | Dataset | Provider and download | Raw flows (Table 2 of the article) |
|---|---|---|---|
| `NF-UNSW-NB15-v3` | NF-UNSW-NB15-v3 | University of Queensland, https://rdm.uq.edu.au/files/abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f | 2,365,424 |
| `NF-ToN-IoT-v3` | NF-ToN-IoT-v3 | University of Queensland, https://rdm.uq.edu.au/files/343e2e8c-6e6e-4a0c-813d-a46acea1b7f4 | 27,520,260 |
| `NF-BoT-IoT-v3` | NF-BoT-IoT-v3 | University of Queensland, https://rdm.uq.edu.au/files/db0187cc-1f78-4215-8e34-303364498866 | 16,933,808 |
| `NF-CSE-CIC-IDS2018-v3` | NF-CSE-CIC-IDS2018-v3 | University of Queensland, https://rdm.uq.edu.au/files/4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22 | 20,115,529 |
| `CSE-CIC-IDS-2017-improved` | CIC-IDS2017, corrected release of Engelen et al. (WTMC 2021) | DistriNet, KU Leuven, https://intrusion-detection.distrinet-research.be/WTMC2021/Dataset/dataset.zip (CSV files, regenerated on 24/11/2021) | 2,100,814 |

Overview pages: the NetFlow datasets are listed at https://staff.itee.uq.edu.au/marius/NIDS_datasets/; the corrected CIC-IDS2017 is described at https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html, together with the fixed CICFlowMeter used to regenerate it (https://github.com/GintsEngelen/CICFlowMeter). The same site also offers a later release (`CICIDS2017_improved.zip`, https://intrusion-detection.distrinet-research.be/CNS2022/) with revised labels and additional features; it was **not** used in this study and is not interchangeable with the WTMC 2021 files.

The raw-flow counts can be used to check a download; they are the row counts of our copies before identifier removal, per-class capping and deduplication. For CIC-IDS2017 the composition table on the provider page sums to 2,100,021 flows, 793 fewer than our copy of the CSV files (2,100,814). The difference is confined to three labels (Benign +624, PortScan +128, "Attempted" +41); all other class counts are identical. The files were regenerated several times in 2021, and the table on the page may predate the last regeneration.

## Preparation

```bash
python scripts/prepare_data.py --datasets nf_unsw_v3 nf_ton_v3 nf_bot_v3 nf_cic2018_v3 cic2017_fixed
```

This writes `data/processed/<dataset>/data.parquet` and `audit.json` (duplicates, label conflicts, class distribution). Dataset-specific decisions are set in `configs/datasets.yaml` and described in Section 2 of the main README:

- NF-v3: `SRC_TO_DST_SECOND_BYTES` and `DST_TO_SRC_SECOND_BYTES` are dropped.
- NF-CSE-CIC-IDS2018-v3: the classes `FTP-BruteForce` and `DoS_attacks-SlowHTTPTest` are dropped.
- Corrected CIC-IDS2017: flows labelled "Attempted" are dropped (`attempted: drop`).
- Classes with more than 200,000 flows are subsampled to 200,000 (seeded); classes with fewer than 50 flows are dropped. The uncapped variants `nf_unsw_v3_nocap` and `cic2017_nocap` keep all flows.

For a smoke test without the real data, `python scripts/make_toy_data.py` writes synthetic toy files to `data/raw/toy_nf/` and `data/raw/toy_cic/`.

## Citation

If you use these datasets, cite their providers:

- NetFlow v3: M. Luay, S. Layeghy, N. Noorbin, M. Sarhan, G. Kulatilleke, N. Moustafa, M. Portmann, Time matters: Temporal NetFlow features for ML-based network intrusion detection, IEEE Access 14 (2026) 66899–66913, https://doi.org/10.1109/ACCESS.2026.3688204 (preprint: arXiv:2503.04404), and M. Sarhan, S. Layeghy, M. Portmann, Towards a standard feature set for network intrusion detection system datasets, Mobile Networks and Applications 27 (2022) 357–370, https://doi.org/10.1007/s11036-021-01843-0.
- Original datasets behind the NetFlow versions: UNSW-NB15 (Moustafa and Slay, 2015), ToN_IoT (Moustafa, 2021), Bot-IoT (Koroniotis et al., 2019), CSE-CIC-IDS2018 (Sharafaldin et al., 2018 methodology).
- Corrected CIC-IDS2017 (requested by the providers): G. Engelen, V. Rimmer, W. Joosen, Troubleshooting an intrusion detection dataset: the CICIDS2017 case study, in: IEEE Security and Privacy Workshops (SPW), Workshop on Traffic Measurements for Cybersecurity (WTMC), 2021, pp. 7–12, https://doi.org/10.1109/SPW53761.2021.00009. The original CIC-IDS2017: Sharafaldin et al. (2018).

Full references are in the article.
