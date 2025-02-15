# ------------------------------------------------------------------------------
# Reference: https://github.com/grip-unina/ClipBased-SyntheticImageDetection/blob/main/compute_metrics.py
# Modified by Aayan Yadav (https://github.com/ydvaayan)
# ------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from utils.processing import prepare_data


dict_metrics = {
    "auc": lambda label, score: metrics.roc_auc_score(label, score),
    "acc": lambda label, score: metrics.balanced_accuracy_score(label, score > 0),
}


def compute_metrics(data_dir, output_csv, metrics_fun):
    table = pd.read_csv(output_csv)
    list_algs = [
        _ for _ in table.columns if _ != "path"
    ]  # list of all algorithms or models used
    table = prepare_data(data_dir).merge(
        table,
        on=[
            "path",
        ],
    )
    assert "type" in table
    assert "source" in table
    list_sources = sorted(
        [
            source
            for source in set(table["source"])
            if table[table["source"] == source]["type"].iloc[0] != "real"
        ]
    )
    table["label"] = table["type"] != "real"  # 1 for not real and 0 for real

    tab_metrics = pd.DataFrame(index=list_algs, columns=list_sources)
    tab_metrics.loc[:, :] = np.nan
    for source in list_sources:
        tab_source = table[
            (table["type"] == "real") | (table["source"] == source)
        ]  # sub-table containing only of the ai sources and real images
        for alg in list_algs:
            score = tab_source[alg].values - 0.2  # .values convert data to numpy array
            label = tab_source["label"].values
            if np.all(np.isfinite(score)) == False:  # makes sure all scores are finite
                continue

            tab_metrics.loc[alg, source] = metrics_fun(label, score)
    tab_metrics.loc[:, "AVG"] = tab_metrics.mean(1)

    return tab_metrics


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "-i",
        type=str,
        help="The path of the directory that contains data",
        default="./data",
    )
    parser.add_argument(
        "--out_csv",
        "-o",
        type=str,
        help="The path of the output csv file",
        default="./results.csv",
    )
    parser.add_argument(
        "--metrics",
        "-w",
        type=str,
        help="type of metrics ('auc' or 'acc')",
        default="auc",
    )
    parser.add_argument(
        "--save_tab",
        "-t",
        type=str,
        help="The path of the metrics csv file",
        default=None,
    )
    args = vars(parser.parse_args())

    tab_metrics = compute_metrics(
        args["data_dir"], args["out_csv"], dict_metrics[args["metrics"]]
    )
    tab_metrics.index.name = args["metrics"]
    print(tab_metrics.to_string(float_format=lambda x: "%5.3f" % x))

    if args["save_tab"] is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args["save_tab"])), exist_ok=True)
        tab_metrics.to_csv(args["save_tab"])
