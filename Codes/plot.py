# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import functools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.metrics import auc, roc_curve

import sys

sys.path.append("Utils")
from utils import update_args_from_config

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=False, type=bool)
parser.add_argument("--model_dir", default="exp/cifar100", type=str)
parser.add_argument("--keep", default="", type=str)
parser.add_argument("--scores", default="", type=str)
parser.add_argument("--name", default="full", type=str)

temp_args, _ = parser.parse_known_args()
if temp_args.config:
    args = parser.parse_args([])
    update_args_from_config(args, config="config.json")
    args = parser.parse_args(namespace=args)
else:
    args = parser.parse_args()


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    return fpr, tpr, auc(fpr, tpr), acc


def get_best_f1_score(x, score):
    """
    给定真实标签 x 和模型得分 score，计算所有可能阈值下的 F1-score，
    并返回最大 F1-score、对应的最佳阈值和 Precision。
    """
    fpr, tpr, thresholds = roc_curve(x, -score)

    best_f1 = 0
    best_threshold = None
    best_precision = 0

    for threshold in thresholds:
        y_pred = (score <= -threshold).astype(
            int
        )  # predict positive if score <= -threshold

        tp = np.sum((y_pred == 1) & (x == 1))
        fp = np.sum((y_pred == 1) & (x == 0))
        fn = np.sum((y_pred == 0) & (x == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision

    return best_f1, best_threshold, best_precision


def load_data():
    """
    Load our saved scores and then put them into a big matrix.
    """
    global scores, keep, target_scores, target_keep
    scores = []
    keep = []
    target_scores = []
    target_keep = []

    for path in os.listdir(args.model_dir):
        scores.append(np.load(os.path.join(args.model_dir, path, "scores.npy")))
        keep.append(np.load(os.path.join(args.model_dir, path, "keep.npy")))
    target_scores.append(np.load(args.scores))
    target_keep.append(np.load(args.keep))

    scores = np.array(scores)
    keep = np.array(keep)
    target_scores = np.array(target_scores)
    target_keep = np.array(target_keep)

    return scores, keep


def generate_ours(
    keep,
    scores,
    check_keep,
    check_scores,
    in_size=100000,
    out_size=100000,
    fix_variance=False,
):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """

    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    in_size = min(min(map(len, dat_in)), in_size)
    out_size = min(min(map(len, dat_out)), out_size)

    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
        score = pr_in - pr_out

        prediction.extend(score.mean(1))
        answers.extend(ans)

    return prediction, answers


def generate_ours_offline(
    keep,
    scores,
    check_keep,
    check_scores,
    in_size=100000,
    out_size=100000,
    fix_variance=False,
):
    """
    Fit a single predictive model using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    out_size = min(min(map(len, dat_out)), out_size)

    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)

        prediction.extend(score.mean(1))
        answers.extend(ans)
    return prediction, answers


def generate_global(keep, scores, check_keep, check_scores):
    """
    Use a simple global threshold sweep to predict if the examples in
    check_scores were training data or not, using the ground truth answer from
    check_keep.
    """
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        prediction.extend(-sc.mean(1))
        answers.extend(ans)

    return prediction, answers


def do_plot(
    fn, keep, scores, ntest, legend="", metric="auc", sweep_fn=sweep, **plot_kwargs
):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    prediction, answers = fn(keep, scores, target_keep, target_scores)

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr < 0.001)[0][-1]]

    f1_score, threshold, prec = get_best_f1_score(
        np.array(answers, dtype=bool), np.array(prediction)
    )
    with open("f1_scores.txt", "a") as f1:
        f1.write(
            "Name %s Attack %s   F1 Score %.4f, Threshold %.4f, Precision %.4f\n"
            % (args.name, legend, f1_score, threshold, prec)
        )

    with open("outdata.txt", "a") as f:
        f.write(
            "Name %s Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\n"
            % (args.name, legend, auc, acc, low)
        )
    metric_text = ""
    if metric == "auc":
        metric_text = "auc=%.3f" % auc
    elif metric == "acc":
        metric_text = "acc=%.3f" % acc

    plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)
    return (acc, auc)


def fig_fpr_tpr():
    plt.figure(figsize=(4, 3))

    do_plot(generate_ours, keep, scores, 1, "Ours (online)\n", metric="auc")

    do_plot(
        functools.partial(generate_ours, fix_variance=True),
        keep,
        scores,
        1,
        "Ours (online, fixed variance)\n",
        metric="auc",
    )

    do_plot(
        functools.partial(generate_ours_offline),
        keep,
        scores,
        1,
        "Ours (offline)\n",
        metric="auc",
    )

    do_plot(
        functools.partial(generate_ours_offline, fix_variance=True),
        keep,
        scores,
        1,
        "Ours (offline, fixed variance)\n",
        metric="auc",
    )

    do_plot(generate_global, keep, scores, 1, "Global threshold\n", metric="auc")

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.subplots_adjust(bottom=0.18, left=0.18, top=0.96, right=0.96)
    plt.legend(fontsize=8)
    plt.savefig("fprtpr.png")
    plt.show()


def do_analyze():
    do_plot(generate_ours, keep, scores, 1, "Online\n", metric="auc")

    do_plot(
        functools.partial(generate_ours, fix_variance=True),
        keep,
        scores,
        1,
        "Online fixed\n",
        metric="auc",
    )

    do_plot(
        functools.partial(generate_ours_offline),
        keep,
        scores,
        1,
        "Offline\n",
        metric="auc",
    )

    do_plot(
        functools.partial(generate_ours_offline, fix_variance=True),
        keep,
        scores,
        1,
        "Offline fixed\n",
        metric="auc",
    )

    do_plot(generate_global, keep, scores, 1, "Global threshold\n", metric="auc")


if __name__ == "__main__":
    load_data()
    do_analyze()
