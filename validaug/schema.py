"""Schemas for the NIDS dataset families.

Two completely different feature families:
  * ``netflow``      : NF-*-v2 / NF-*-v3 (nProbe, 43/53 features, UPPER-CASE columns).
  * ``cicflowmeter`` : original CIC-IDS2017/2018 (MachineLearningCVE) and the label-corrected
                        release by Engelen et al. (slightly different column names).

Column names are normalized to lower snake_case and aliases are mapped to canonical names
so that constraint templates can be shared.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TCP_FLAG_BITS = ["fin", "syn", "rst", "psh", "ack", "urg", "ece", "cwr"]


def norm_col(c: str) -> str:
    c = c.strip().lower()
    c = re.sub(r"[^0-9a-z]+", "_", c)
    return c.strip("_")


# --------------------------------------------------------------------------
# Alias -> canonical for CICFlowMeter (original 2017 and Engelen 2021 releases)
# --------------------------------------------------------------------------
CIC_ALIASES = {
    "destination_port": "dst_port", "dst_port": "dst_port",
    "source_port": "src_port", "src_port": "src_port",
    "source_ip": "src_ip", "src_ip": "src_ip",
    "destination_ip": "dst_ip", "dst_ip": "dst_ip",
    "total_fwd_packets": "fwd_pkts", "total_fwd_packet": "fwd_pkts",
    "total_backward_packets": "bwd_pkts", "total_bwd_packets": "bwd_pkts",
    "total_length_of_fwd_packets": "fwd_bytes", "total_length_of_fwd_packet": "fwd_bytes",
    "total_length_of_bwd_packets": "bwd_bytes", "total_length_of_bwd_packet": "bwd_bytes",
    "fwd_packets_length_total": "fwd_bytes", "bwd_packets_length_total": "bwd_bytes",  # parquet release
    "fwd_packet_length_max": "fwd_pkt_len_max", "fwd_packet_length_min": "fwd_pkt_len_min",
    "fwd_packet_length_mean": "fwd_pkt_len_mean", "fwd_packet_length_std": "fwd_pkt_len_std",
    "bwd_packet_length_max": "bwd_pkt_len_max", "bwd_packet_length_min": "bwd_pkt_len_min",
    "bwd_packet_length_mean": "bwd_pkt_len_mean", "bwd_packet_length_std": "bwd_pkt_len_std",
    "min_packet_length": "pkt_len_min", "packet_length_min": "pkt_len_min",
    "max_packet_length": "pkt_len_max", "packet_length_max": "pkt_len_max",
    "packet_length_mean": "pkt_len_mean", "packet_length_std": "pkt_len_std",
    "flow_iat_min": "flow_iat_min", "flow_iat_mean": "flow_iat_mean", "flow_iat_max": "flow_iat_max",
    "fwd_iat_min": "fwd_iat_min", "fwd_iat_mean": "fwd_iat_mean", "fwd_iat_max": "fwd_iat_max",
    "bwd_iat_min": "bwd_iat_min", "bwd_iat_mean": "bwd_iat_mean", "bwd_iat_max": "bwd_iat_max",
    "fwd_iat_total": "fwd_iat_total", "bwd_iat_total": "bwd_iat_total",
    "flow_duration": "flow_duration",
    "fin_flag_count": "fin_flag_count", "syn_flag_count": "syn_flag_count",
    "rst_flag_count": "rst_flag_count", "psh_flag_count": "psh_flag_count",
    "ack_flag_count": "ack_flag_count", "urg_flag_count": "urg_flag_count",
    "cwe_flag_count": "cwr_flag_count", "cwr_flag_count": "cwr_flag_count",
    "ece_flag_count": "ece_flag_count",
}


@dataclass
class DatasetSchema:
    name: str
    family: str  # "netflow" | "cicflowmeter"
    label_col: str  # multi-class label (normalized name)
    benign_values: list = field(default_factory=lambda: ["benign", "normal"])
    binary_label_col: str | None = None
    time_col: str | None = None
    host_col: str | None = None
    id_cols: list = field(default_factory=list)  # always removed from the feature set
    categorical_cols: list = field(default_factory=list)
    bitmask_cols: list = field(default_factory=list)  # split into 8 bits
    port_cols: dict = field(default_factory=dict)  # {"dst_port": "categorical"|"drop"}
    protocol_col: str | None = None
    tcp_value: int = 6
    icmp_value: int = 1
    extra_drop: list = field(default_factory=list)


def netflow_schema(name: str) -> DatasetSchema:
    return DatasetSchema(
        name=name,
        family="netflow",
        label_col="attack",
        binary_label_col="label",
        time_col="flow_start_milliseconds",
        host_col="ipv4_src_addr",
        id_cols=[
            "ipv4_src_addr", "ipv4_dst_addr", "l4_src_port",
            "flow_start_milliseconds", "flow_end_milliseconds",
            "dns_query_id",  # random ID -> leakage/noise
        ],
        categorical_cols=["protocol", "l7_proto", "icmp_type", "icmp_ipv4_type",
                          "dns_query_type", "ftp_command_ret_code"],
        bitmask_cols=["tcp_flags", "client_tcp_flags", "server_tcp_flags"],
        port_cols={"l4_dst_port": "categorical"},
        protocol_col="protocol",
    )


def cic_schema(name: str) -> DatasetSchema:
    return DatasetSchema(
        name=name,
        family="cicflowmeter",
        label_col="label",
        time_col="timestamp",
        host_col="src_ip",
        id_cols=["flow_id", "src_ip", "dst_ip", "src_port", "timestamp", "id"],
        categorical_cols=["protocol"],
        bitmask_cols=[],
        port_cols={"dst_port": "categorical"},
        protocol_col="protocol",
        extra_drop=["attempted_category"],
    )


def make_schema(cfg: dict) -> DatasetSchema:
    fam = cfg["family"]
    sch = netflow_schema(cfg["name"]) if fam == "netflow" else cic_schema(cfg["name"])
    for k, v in cfg.get("overrides", {}).items():
        setattr(sch, k, v)
    return sch
