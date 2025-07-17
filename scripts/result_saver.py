# scripts/result_saver.py

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt


def save_results(
    results_df: pd.DataFrame, out_dir: str, dataset: str, seq: str
) -> None:
    """
    Save results DataFrame to CSV and generate error plots.

    - Tables under out_dir/tables/
    - Graphs under out_dir/graphs/

    Args:
        results_df: DataFrame with columns ['frame', 'abs_trans_err', 'abs_rot_err_deg', 'rel_trans_err', 'rel_rot_err_deg', ...]
        out_dir: Root output directory (e.g. outputs/online/stereo_results)
        dataset: 'stereo', 'vo' or 'cb'
        seq: sequence identifier (split or sequence number)
    """
    tables_dir = os.path.join(out_dir, "tables")
    graphs_dir = os.path.join(out_dir, "graphs")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # load method from config
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        feat_type = cfg["feature"]["type"].lower()  # orb / sift / superpoint
        match_type = cfg["matcher"]["type"].lower()  # bf / superglue

    # create prefix: orb, sift, sp
    if feat_type == "superpoint":
        prefix = "sp"
    elif feat_type == "sift":
        prefix = "sift"
    else:
        prefix = "orb"

    # create full tag and output paths
    tag = f"{prefix}_seq_{seq}"
    table_path = os.path.join(tables_dir, f"{tag}.csv")
    graph_path = os.path.join(graphs_dir, f"{tag}.png")

    # Save CSV
    results_df.to_csv(table_path, index=False)

    # Plot and save error graph
    fig, ax = plt.subplots()
    ax.plot(
        results_df["frame"], results_df["abs_trans_err"], label="Abs Translation Error"
    )
    ax.plot(
        results_df["frame"],
        results_df["abs_rot_err_deg"],
        label="Abs Rotation Error (deg)",
    )
    ax.plot(
        results_df["frame"], results_df["rel_trans_err"], label="Rel Translation Error"
    )
    ax.plot(
        results_df["frame"],
        results_df["rel_rot_err_deg"],
        label="Rel Rotation Error (deg)",
    )

    ax.set_xlabel("Frame")
    ax.set_ylabel("Error")
    ax.set_title(f"{dataset.upper()} Error Metrics Sequence {seq}")
    ax.legend()
    fig.tight_layout()

    fig.savefig(graph_path)
    plt.close(fig)
