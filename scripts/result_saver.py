import os
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
        dataset: 'stereo' or 'vo'
        seq: sequence identifier (split or sequence number)
    """
    tables_dir = os.path.join(out_dir, "tables")
    graphs_dir = os.path.join(out_dir, "graphs")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    table_path = os.path.join(tables_dir, f"{dataset}_seq_{seq}.csv")
    results_df.to_csv(table_path, index=False)

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

    graph_path = os.path.join(graphs_dir, f"{dataset}_seq_{seq}.png")
    fig.savefig(graph_path)
    plt.close(fig)
