#!/usr/bin/env python3
"""Recompute the tables of the paper from the raw per-run results and compare them
with the numbers printed in the LaTeX source.

Usage (from the root of the code repository):
    python scripts/verify_tables.py --runs runs --tex manuscript.tex supplementary.tex \
        --out runs/verify_report.md

Every table cell is recomputed from runs/<phase>/results.jsonl, rounded as in the paper
and compared with the printed value. A cell is reported as MISMATCH when the printed value
differs from the recomputed one by more than one unit in the last printed digit.
"""
import argparse, json, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")

DS = ["cic2017_fixed", "nf_bot_v3", "nf_cic2018_v3", "nf_ton_v3", "nf_unsw_v3"]
DSNAME = {"cic2017_fixed": "CIC-IDS2017", "nf_bot_v3": "NF-BoT-IoT-v3", "nf_cic2018_v3": "NF-CSE-CIC-IDS2018-v3",
          "nf_ton_v3": "NF-ToN-IoT-v3", "nf_unsw_v3": "NF-UNSW-NB15-v3"}
MNAME = {"None": "none", "ROS": "ros", "SMOTE": "smote", "Borderline-SMOTE": "borderline_smote", "B-SMOTE": "borderline_smote",
         "ADASYN": "adasyn", "SMOTE-ENN": "smote_enn", "CTGAN": "ctgan", "TVAE": "tvae", "Diffusion": "diffusion",
         "Diffusion+Proj": "diffusion_proj", "VALID-Aug": "validaug", "\\method{}": "validaug", "method": "validaug"}
CLFNAME = {"XGBoost": "xgboost", "Random forest": "rf", "MLP": "mlp", "1D-CNN": "cnn1d", "LSTM": "lstm", "FT-Transformer": "ft_transformer"}

def load(runs, phase):
    p = Path(runs) / phase / "results.jsonl"
    if not p.exists():
        return None
    return pd.DataFrame([json.loads(l) for l in open(p) if l.strip()])

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1, run)
    return adj

def blocks(df, metric):
    return df.groupby(["method", "dataset", "classifier"])[metric].mean().unstack("method")

# ---------------------------------------------------------------- LaTeX parsing
def tex_tables(paths):
    out = {}
    for p in paths:
        s = open(p, encoding="utf-8").read()
        for m in re.finditer(r"\\label\{(tab:[^}]+)\}", s):
            i = s.find("\\begin{tabular}", m.end()); j = s.find("\\end{tabular}", i)
            if i < 0 or (s.find("\\end{table}", m.end()) < i):
                continue
            body = s[i:j]
            body = body[body.find("\n") + 1:]
            rows = []
            for r in re.split(r"\\\\", body):
                r = re.sub(r"\\(toprule|midrule|bottomrule)|\\cmidrule\{[^}]*\}", "", r).strip()
                if not r or "&" not in r:
                    continue
                cells = [c.strip() for c in r.split("&")]
                rows.append(cells)
            out[m.group(1)] = rows
    return out

def clean_label(c):
    c = re.sub(r"\\best\{(.*?)\}", r"\1", c)
    c = c.replace("\\quad", "").replace("\\textit", "").replace("$", "").replace("\\", "").strip()
    c = re.sub(r"[{}]", "", c)
    return c.strip()

def parse_cell(c):
    """Return list of (value, flag) with flag '<' for '<0.001'; None for '--'."""
    c = re.sub(r"\\best\{(.*?)\}", r"\1", c)
    c = c.replace("{,}", "").replace("\\%", "%").replace("$", "").replace("^{*}", "").replace("\\pm", "±")
    if c.strip() in ("--", "-", "–", ""):
        return [None]
    vals = []
    for m in re.finditer(r"(<\s*)?([+\-−±]?)(\d+(?:\.\d+)?)", c):
        lt, sign, num = m.group(1), m.group(2), m.group(3)
        v = float(num) * (-1 if sign in ("-", "−") else 1)
        vals.append((v, "<" if lt else "", len(num.split(".")[1]) if "." in num else 0))
    return vals or [None]

# ---------------------------------------------------------------- comparison
REPORT = []
def compare(label, printed_rows, expected, key_of_row):
    """expected: dict row_key -> list of floats (or None) in printed order (flattened)."""
    if printed_rows is None:
        REPORT.append((label, "SKIP", "table not found in the LaTeX source")); return
    n_ok = n_bad = 0; bad = []
    for cells in printed_rows:
        k = key_of_row(cells)
        if k is None or k not in expected:
            continue
        flat = [x for c in cells[1:] for x in parse_cell(c)]
        exp = expected[k]
        if len(flat) != len(exp):
            bad.append(f"{k}: {len(flat)} printed values vs {len(exp)} expected"); n_bad += 1; continue
        for pos, (pv, ev) in enumerate(zip(flat, exp)):
            if ev is None or pv is None:
                if (ev is None) != (pv is None) and not (ev is not None and np.isnan(ev)):
                    bad.append(f"{k} col {pos+1}: printed {pv} expected {ev}"); n_bad += 1
                continue
            v, flag, dec = pv
            tol = 1.0001 * 10 ** (-dec) if dec else 0.5001
            if flag == "<":
                good = ev < v + 1e-12
            else:
                good = abs(v - ev) <= tol
            if good:
                n_ok += 1
            else:
                n_bad += 1; bad.append(f"{k} col {pos+1}: printed {flag}{v} recomputed {ev:.4f}")
    status = "OK" if n_bad == 0 and n_ok > 0 else ("MISMATCH" if n_bad else "SKIP")
    REPORT.append((label, status, f"{n_ok} cells match" + ("" if not bad else "; " + " | ".join(bad[:12]) + (" ..." if len(bad) > 12 else ""))))

def by_method(cells):
    return MNAME.get(clean_label(cells[0]).replace("textsc", ""), None) if cells else None

# ---------------------------------------------------------------- recomputation
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--tex", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    T = tex_tables(a.tex)
    B = load(a.runs, "phaseB_protocol"); D = load(a.runs, "phaseD_main"); E = load(a.runs, "phaseE_ablation")
    F = load(a.runs, "phaseF_generalization"); CW = load(a.runs, "phaseG_classweights")
    D50 = load(a.runs, "phaseG_diff50k"); NC = load(a.runs, "phaseG_nocap")

    # ---- Table: protocol inflation (phase B, none, multi-class)
    if B is not None:
        d = B[(B.method == "none") & (B.scenario == "full")]
        g = d.groupby(["dataset", "split"]).macro_f1.mean().unstack()
        exp = {DSNAME[k]: [g.loc[k, s] for s in ["random_naive", "random", "temporal_stratified", "group"]] for k in DS}
        compare("tab:inflation", T.get("tab:inflation"), exp, lambda c: clean_label(c[0]) if clean_label(c[0]) in exp else None)
        # ---- gain per protocol
        rows = []
        exp = {}
        piv = B[B.scenario == "full"].set_index(["dataset", "classifier", "seed", "split", "method"])
        ms = ["ros", "smote", "borderline_smote", "adasyn", "smote_enn"]
        PN = {"Naive random": "random_naive", "Random": "random", "Temporal": "temporal_stratified", "Group": "group"}
        MT = {"Macro-F1": "macro_f1", "Rare-F1": "rare_f1", "MCC": "mcc"}
        for mt, col in MT.items():
            for pn, sp in PN.items():
                sub = B[(B.scenario == "full") & (B.split == sp)].pivot_table(index=["dataset", "classifier", "seed"], columns="method", values=col)
                exp[(mt, pn)] = [(sub[m] - sub["none"]).mean() for m in ms]
        state = {"mt": None}
        def key_gain(c):
            l0 = clean_label(c[0]); l0 = re.sub(r"multirow4\*", "", l0)
            if l0: state["mt"] = l0
            k = (state["mt"], clean_label(c[1])); return ("ROW", k) if k in exp else None
        rows = [[c[1]] + c[2:] for c in (T.get("tab:gain-protocol") or [])]
        # keep metric context by pre-processing
        ctx = []; cur = None
        for c in T.get("tab:gain-protocol") or []:
            l0 = re.sub(r"multirow\{4\}\{\*\}", "", c[0]); l0 = clean_label(l0)
            if l0: cur = l0
            ctx.append([f"{cur}|{clean_label(c[1])}"] + c[2:])
        exp2 = {f"{k[0]}|{k[1]}": v for k, v in exp.items()}
        compare("tab:gain-protocol", ctx, exp2, lambda c: c[0] if c[0] in exp2 else None)

    if D is not None:
        tf = D[(D.split == "temporal_stratified") & (D.scenario == "full") & ~D.method.isin(["tabddpm", "tabsyn"])]
        # ---- main macro-F1 per dataset
        g = tf.groupby(["method", "dataset"]).macro_f1.mean().unstack()
        exp = {m: [g.loc[m, k] for k in DS] + [g.loc[m].mean()] for m in g.index}
        compare("tab:main-f1", T.get("tab:main-f1"), exp, by_method)
        # ---- further metrics + paired tests vs VALID-Aug
        b = blocks(tf, "macro_f1"); ms = [m for m in b.columns if m != "validaug"]
        ps = [wilcoxon(b["validaug"], b[m]).pvalue for m in ms]; ad = dict(zip(ms, holm(ps)))
        exp = {}
        for m in b.columns:
            r = tf[tf.method == m]
            row = [r.rare_f1.mean(), r.macro_f1_common.mean(), r.mcc.mean()]
            if m != "validaug":
                dlt = b["validaug"] - b[m]
                row += [dlt.mean(), int((dlt > 0).sum()), int((dlt < 0).sum()), ad[m]]
            else:
                row += [None, None, None]
            exp[m] = row
        compare("tab:main-mcc", T.get("tab:main-mcc"), exp, by_method)
        # ---- per classifier
        g = tf.groupby(["classifier", "method"]).macro_f1.mean().unstack()
        cols = ["none", "ros", "smote", "ctgan", "diffusion", "validaug"]
        exp = {k: [g.loc[v, m] for m in cols] for k, v in CLFNAME.items()}
        compare("tab:perclf", T.get("tab:perclf"), exp, lambda c: clean_label(c[0]) if clean_label(c[0]) in exp else None)
        # ---- per-class trade-off
        rec = []
        base = tf[tf.method == "none"].set_index(["dataset", "classifier", "seed"])
        for _, r in tf[tf.method != "none"].iterrows():
            k = (r.dataset, r.classifier, r.seed)
            if k not in base.index: continue
            f0 = base.loc[k, "f1_per_class"]; f1 = r.f1_per_class
            dl = [f1[c] - f0[c] for c in f1 if c in f0]
            rec.append((r.method, sum(x > 0.01 for x in dl), sum(x < -0.01 for x in dl), min(dl)))
        t = pd.DataFrame(rec, columns=["m", "imp", "deg", "worst"]).groupby("m").mean()
        exp = {m: list(t.loc[m]) for m in t.index}
        compare("tab:tradeoff", T.get("tab:tradeoff"), exp, by_method)
        # ---- scarce
        exp = {}
        for m in tf.method.unique():
            row = []
            for sc in ["rare5", "rare1"]:
                r = D[(D.split == "temporal_stratified") & (D.scenario == sc) & (D.method == m)]
                full = len(r) >= 45
                row += [r.macro_f1.mean(), r.rare_f1.mean(), r.mcc.mean()] if full else [None, None, None]
            exp[m] = row
        compare("tab:scarce", T.get("tab:scarce"), exp, by_method)
        # ---- validity / fidelity / privacy / cost (all cells of the main comparison)
        # one value per augmentation cell (the classifiers of a cell share the same synthetic data); binary cells excluded
        main = D[~D.method.isin(["none", "tabddpm", "tabsyn"]) & (D.scenario != "binary")].drop_duplicates(["dataset", "split", "scenario", "seed", "method"])
        exp = {}
        for m, r in main.groupby("method"):
            def mm(c, scale=1.0):
                v = r[c].dropna() if c in r else pd.Series(dtype=float)
                return v.mean() * scale if len(v) else None
            exp[m] = [mm("validity_raw", 100), mm("validity_final", 100), mm("c2st_auc"), mm("wasserstein_mean"),
                      mm("dcr_frac_exact_copy", 100), mm("mia_auc"), mm("aug_wall_s")]
        compare("tab:axes", T.get("tab:axes"), exp, by_method)
        # ---- budget: validation gain vs test gain
        def vg(df):
            g = [r["budget"]["gain"] for r in df.aug_info if isinstance(r, dict) and isinstance(r.get("budget"), dict) and "gain" in r["budget"]]
            return float(np.mean(g)) if g else None
        exp = {}
        for lab, sp, sc in [("Temporal, full data", "temporal_stratified", "full"), ("Temporal, rare classes at 5%", "temporal_stratified", "rare5"),
                            ("Temporal, rare classes at 1%", "temporal_stratified", "rare1"), ("Group, full data", "group", "full"), ("Random, full data", "random", "full")]:
            d = D[(D.split == sp) & (D.scenario == sc)]
            p = d.pivot_table(index=["dataset", "classifier", "seed"], columns="method", values="macro_f1")
            exp[lab] = [vg(d[d.method == "validaug"].drop_duplicates(["dataset", "seed"])), (p["validaug"] - p["none"]).mean()]
        compare("tab:budget", T.get("tab:budget"), exp, lambda c: clean_label(c[0]).replace("%", "%") if clean_label(c[0]) in exp else None)

    if E is not None:
        exp = {}
        EN = {"validaug": "VALID-Aug", "va_no_gc": "w/o constraint guidance", "va_no_gb": "w/o boundary guidance", "va_no_proj": "w/o projection",
              "va_no_filter": "w/o filter", "va_default_budget": "default budget", "va_no_delta": "delta=0", "va_guidance_only": "guidance only",
              "diffusion": "Diffusion (no component)"}
        for m, name in EN.items():
            f = E[(E.scenario == "full") & (E.method == m)]; r5 = E[(E.scenario == "rare5") & (E.method == m)]
            exp[name] = [f.macro_f1.mean(), f.mcc.mean(), f.validity_raw.mean() * 100, f.validity_final.mean() * 100, r5.macro_f1.mean(), r5.mcc.mean()]
        def key_abl(c):
            l = clean_label(c[0]).replace("textsc", "").replace("method", "VALID-Aug").replace("delta", "delta").replace(" ", " ")
            l = l.replace("delta=0", "delta=0").replace("delta =0", "delta=0")
            l = re.sub(r"\s+", " ", l).replace("delta=0", "delta=0")
            if l.startswith("delta"): l = "delta=0"
            return l if l in exp else None
        compare("tab:ablation", T.get("tab:ablation"), exp, key_abl)

    if F is not None:
        exp = {}
        ms = ["none", "smote", "ctgan", "diffusion", "validaug"]
        def row(df):
            g = df.groupby("method").macro_f1.mean(); return [g.get(m, None) for m in ms]
        h = F[F.split == "temporal_stratified"]
        exp["UNSW, hold-out Backdoor"] = row(h[(h.dataset == "nf_unsw_v3") & (h.scenario == "holdout_backdoor")])
        exp["UNSW, hold-out rarest class"] = row(h[(h.dataset == "nf_unsw_v3") & (h.scenario == "holdout_rarest")])
        exp["ToN, hold-out Backdoor"] = row(h[(h.dataset == "nf_ton_v3") & (h.scenario == "holdout_backdoor")])
        exp["ToN, hold-out rarest class"] = row(h[(h.dataset == "nf_ton_v3") & (h.scenario == "holdout_rarest")])
        c = F[(F.split == "cross") & (F.scenario == "binary")]
        for lab, ds in [("BoT to ToN", "nf_bot_v3->nf_ton_v3"), ("CIC18 to UNSW", "nf_cic2018_v3->nf_unsw_v3"),
                        ("ToN to BoT", "nf_ton_v3->nf_bot_v3"), ("UNSW to CIC18", "nf_unsw_v3->nf_cic2018_v3")]:
            exp[lab] = row(c[c.dataset == ds])
        def key_gen(c):
            l = clean_label(c[0]).replace("to", " to ").replace("rightarrow", " to ")
            l = re.sub(r"\s+", " ", l).replace("hold-out Backdoor", "hold-out Backdoor").strip()
            l = l.replace("hold-out Backdoor", "hold-out Backdoor")
            for k in exp:
                if re.sub(r"\s", "", k).lower() == re.sub(r"\s", "", l).lower(): return k
            return None
        compare("tab:general", T.get("tab:general"), exp, key_gen)

    if CW is not None and D is not None:
        both = pd.concat([D[D.split == "temporal_stratified"], CW[CW.split == "temporal_stratified"]])
        both = both[both.classifier != "ft_transformer"]
        exp = {}
        names = {"none": "None", "none_cw": "None + balanced weights", "none_cw_sqrt": "None + square-root weights", "ros": "ROS", "smote": "SMOTE",
                 "smote_enn": "SMOTE-ENN", "diffusion": "Diffusion", "validaug": "VALID-Aug"}
        for m, n in names.items():
            row = []
            for sc in ["full", "rare5", "rare1"]:
                cl = set(CW[(CW.scenario == sc)].classifier)
                r = both[(both.scenario == sc) & (both.method == m) & both.classifier.isin(cl)]
                row += [r.macro_f1.mean(), r.mcc.mean()]
            exp[n] = row
        def key_cw(c):
            l = clean_label(c[0]).replace("textsc", "").replace("method", "VALID-Aug")
            return l if l in exp else None
        compare("tab:cw", T.get("tab:cw"), exp, key_cw)

    if D50 is not None and D is not None:
        cl = set(D50.classifier)
        base = D[(D.split == "temporal_stratified") & (D.scenario == "full") & D.classifier.isin(cl)]
        allr = pd.concat([base, D50])
        g = allr.groupby(["method", "dataset"]).macro_f1.mean().unstack()
        names = {"None": "none", "CTGAN (50k)": "ctgan", "TVAE (50k)": "tvae", "Diffusion (50k)": "diffusion_50k", "Diffusion (300k)": "diffusion", "VALID-Aug": "validaug"}
        exp = {n: [g.loc[m, k] for k in DS] + [g.loc[m].mean()] for n, m in names.items()}
        compare("tab:d50", T.get("tab:d50"), exp, lambda c: (lambda l: l if l in exp else None)(clean_label(c[0]).replace("textsc", "").replace("method", "VALID-Aug")))

    if NC is not None:
        names = {"None": "none", "None + balanced weights": "none_cw", "None + square-root weights": "none_cw_sqrt", "ROS": "ros", "SMOTE": "smote",
                 "Diffusion": "diffusion", "VALID-Aug": "validaug"}
        g = NC.groupby(["method", "dataset", "classifier"]).macro_f1.mean()
        exp = {}
        for n, m in names.items():
            if m not in NC.method.unique(): continue
            r = NC[NC.method == m]
            exp[n] = [g[(m, "cic2017_nocap", "xgboost")], g[(m, "cic2017_nocap", "mlp")], g[(m, "nf_unsw_v3_nocap", "xgboost")],
                      g[(m, "nf_unsw_v3_nocap", "mlp")], r.macro_f1.mean(), r.mcc.mean()]
        compare("tab:nocap", T.get("tab:nocap"), exp, lambda c: (lambda l: l if l in exp else None)(clean_label(c[0]).replace("textsc", "").replace("method", "VALID-Aug")))

    # ---------------------------------------------------------------- report
    lines = ["# Table verification report", "", "| Table | Status | Details |", "|---|---|---|"]
    for lab, st, det in REPORT:
        lines.append(f"| {lab} | {st} | {det} |")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
    sys.exit(1 if any(st == "MISMATCH" for _, st, _ in REPORT) else 0)

if __name__ == "__main__":
    main()
