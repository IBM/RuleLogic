"""Vizualize data related to an experiment"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import ticker
from matplotlib.dates import num2date


def _nanround(x, decimals=3):
    if x is None:
        return x
    elif np.isnan(x):
        return x
    else:
        return "%.5f" % x


def print_evals(eval_dict, exp_id):
    print(f"Experiment {exp_id}:")
    for key, val in eval_dict.items():
        print(f"{key}: {_nanround(val, 4)}", end="\t")
    print()
    print()


def plot_rule_triggers(tstamps, gold, pred, taskID, ruleID, outfile):
    """plot comparing triggers of a rule"""

    assert len(gold) == len(tstamps), "Flags must have same length as tstamps."
    assert len(pred) == len(tstamps), "Flags must have same length as tstamps."

    fig, ax = plt.subplots()

    ax.step(tstamps, np.asarray(gold), label="ground truth")
    ax.step(tstamps, np.asarray(pred), label="prediction", linestyle="--")
    # label only 4 points on the x axis, formatting labels over two lines
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_formatter(fmt_stamp)
    # #hourly minor ticks
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1/24))

    ax.yaxis.set_major_locator(ticker.FixedLocator([0, 1]))
    ax.set_ylabel("Rule Triggered")

    title = f"task: {taskID} \n rule: {ruleID}"
    ax.set_title(title)

    ax.legend()
    fig.set_size_inches(w=7, h=5)

    fig.savefig(outfile)
    plt.close(fig)


def fmt_stamp(val, pos):
    """formatter function for plot

    Splits the iso timestamp into two lines
    """

    dt = num2date(val)
    xx = dt.isoformat()
    label = xx.replace("T", "\n")

    return label
