"""Generate synthetic data with NF-v3-like and CIC-IDS2017-like structure for smoke tests.

NOT for paper results. Only checks that the pipeline runs end-to-end, including real-world
"traps": duplicated flows, inf/NaN, constant columns, "Attempted" labels,
column names with leading spaces (CIC), attacks concentrated on few hosts.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NF_CLASSES = {  # name: (share, main protocol, destination port, number of source hosts)
    "Benign": (0.80, 6, 443, 200), "DDoS": (0.08, 6, 80, 30), "scanning": (0.06, 6, 22, 6),
    "Brute Force": (0.03, 6, 21, 4), "Backdoor": (0.02, 17, 31337, 3), "injection": (0.01, 6, 80, 3),
}


def make_nf(n, rng):
    rows = []
    t0 = 1_700_000_000_000
    for cls, (p, proto, port, nh) in NF_CLASSES.items():
        k = int(n * p)
        prot = np.where(rng.random(k) < 0.85, proto, rng.choice([6, 17, 1], k))
        inp = rng.geometric(0.05 if cls == "Benign" else 0.3, k)
        outp = np.maximum(0, inp + rng.integers(-2, 3, k))
        hdr = np.where(prot == 6, 40, np.where(prot == 17, 28, 28))
        payload = {"Benign": 600, "DDoS": 20, "scanning": 2, "Brute Force": 80, "Backdoor": 300, "injection": 900}[cls]
        inb = inp * (hdr + rng.integers(0, payload + 1, k))
        outb = outp * (hdr + rng.integers(0, payload + 1, k))
        dur = (rng.exponential(2000 if cls == "Benign" else 200, k) * (inp > 1)).astype(int)
        start = t0 + np.sort(rng.integers(0, 3_600_000 * 24, k))
        if cls != "Benign":  # attacks concentrated in time
            start = t0 + rng.integers(3_600_000 * (2 + list(NF_CLASSES).index(cls) * 3),
                                      3_600_000 * (5 + list(NF_CLASSES).index(cls) * 3), k)
        flags = np.where(prot == 6, rng.choice([2, 18, 24, 17, 4, 27], k), 0)
        mn = np.minimum(inb // np.maximum(inp, 1), 1500)
        d = pd.DataFrame({
            "FLOW_START_MILLISECONDS": start, "FLOW_END_MILLISECONDS": start + dur,
            "IPV4_SRC_ADDR": [f"10.0.{list(NF_CLASSES).index(cls)}.{h}" for h in rng.integers(0, nh, k)],
            "L4_SRC_PORT": rng.integers(1024, 65535, k),
            "IPV4_DST_ADDR": "192.168.1.1", "L4_DST_PORT": np.where(rng.random(k) < 0.8, port, rng.integers(1, 65535, k)),
            "PROTOCOL": prot, "L7_PROTO": np.round(rng.choice([7.0, 91.178, 5.0, 0.0], k), 3),
            "IN_BYTES": inb, "IN_PKTS": inp, "OUT_BYTES": outb, "OUT_PKTS": outp,
            "TCP_FLAGS": flags, "CLIENT_TCP_FLAGS": flags, "SERVER_TCP_FLAGS": np.where(prot == 6, flags & 18, 0),
            "FLOW_DURATION_MILLISECONDS": dur, "DURATION_IN": (dur * rng.random(k)).astype(int), "DURATION_OUT": 0,
            "MIN_TTL": 64, "MAX_TTL": rng.choice([64, 128], k),
            "LONGEST_FLOW_PKT": mn + rng.integers(0, 50, k), "SHORTEST_FLOW_PKT": np.maximum(mn - rng.integers(0, 20, k), 0),
            "MIN_IP_PKT_LEN": np.maximum(mn - 10, 0), "MAX_IP_PKT_LEN": mn + 30,
            "SRC_TO_DST_SECOND_BYTES": inb / np.maximum(dur / 1000, 1e-3),
            "RETRANSMITTED_IN_BYTES": np.where(prot == 6, (inb * rng.random(k) * 0.05).astype(int), 0),
            "RETRANSMITTED_IN_PKTS": 0, "TCP_WIN_MAX_IN": np.where(prot == 6, rng.choice([1024, 65535, 8192], k), 0),
            "ICMP_TYPE": np.where(prot == 1, rng.choice([0, 2048], k), 0), "DNS_QUERY_ID": rng.integers(0, 65535, k),
            "DNS_QUERY_TYPE": 0, "SRC_TO_DST_IAT_MIN": 0,
            "SRC_TO_DST_IAT_AVG": (dur / np.maximum(inp, 1)).astype(int), "SRC_TO_DST_IAT_MAX": dur,
            "Label": int(cls != "Benign"), "Attack": cls,
        })
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df = pd.concat([df, df.sample(frac=0.03, random_state=0)], ignore_index=True)  # duplicates
    df.loc[df.sample(frac=0.001, random_state=1).index, "SRC_TO_DST_SECOND_BYTES"] = np.inf
    return df.sample(frac=1.0, random_state=2).reset_index(drop=True)


def make_cic(n, rng):
    labels = rng.choice(["BENIGN", "DoS Hulk", "PortScan", "FTP-Patator", "Bot", "DoS Hulk - Attempted"],
                        n, p=[0.8, 0.08, 0.06, 0.03, 0.02, 0.01])
    fp = rng.geometric(0.2, n); bp = rng.geometric(0.25, n) - 1
    fl = rng.integers(0, 1500, (n, 3)); fl.sort(1)
    bl = rng.integers(0, 1500, (n, 3)); bl.sort(1)
    iat = rng.exponential(1e5, (n, 3)); iat.sort(1)
    dur = iat[:, 2] * fp
    return pd.DataFrame({
        "Flow ID": "x", " Source IP": [f"172.16.0.{i}" for i in rng.integers(0, 50, n)], " Source Port": rng.integers(1, 65535, n),
        " Destination IP": "192.168.10.50", " Destination Port": rng.choice([80, 443, 21, 22, 8080], n),
        " Protocol": rng.choice([6, 17], n, p=[0.9, 0.1]),
        " Timestamp": pd.date_range("2017-07-03 08:00", periods=n, freq="1s").strftime("%d/%m/%Y %H:%M:%S"),
        " Flow Duration": dur.astype(int), " Total Fwd Packets": fp, " Total Backward Packets": bp,
        "Total Length of Fwd Packets": fp * fl[:, 1], " Total Length of Bwd Packets": bp * bl[:, 1],
        " Fwd Packet Length Max": fl[:, 2], " Fwd Packet Length Min": fl[:, 0], " Fwd Packet Length Mean": fl[:, 1],
        "Bwd Packet Length Max": bl[:, 2], " Bwd Packet Length Min": bl[:, 0], " Bwd Packet Length Mean": bl[:, 1],
        " Flow IAT Mean": iat[:, 1], " Flow IAT Max": iat[:, 2], " Flow IAT Min": iat[:, 0],
        "Flow Bytes/s": np.where(rng.random(n) < 0.01, np.inf, fp * fl[:, 1] / np.maximum(dur, 1) * 1e6),
        " SYN Flag Count": rng.integers(0, 2, n), " ACK Flag Count": rng.integers(0, 2, n),
        " Fwd Header Length": fp * 20, " Fwd Header Length.1": fp * 20, " Bwd PSH Flags": 0,
        " Label": labels,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(a.out)
    (out / "toy_nf").mkdir(parents=True, exist_ok=True)
    (out / "toy_cic").mkdir(parents=True, exist_ok=True)
    make_nf(a.n, rng).to_csv(out / "toy_nf" / "toy_nf_v3.csv", index=False)
    make_nf(a.n // 2, np.random.default_rng(7)).to_csv(out / "toy_nf" / "toy_nf_v3_b.csv", index=False)
    make_cic(a.n // 2, rng).to_csv(out / "toy_cic" / "toy_cic.csv", index=False)
    print("Toy data written to", out)
