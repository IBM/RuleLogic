"""
Command-Line Interface for Rule Logic Evaluation
=================================================

This module provides a CLI for common operations in rule-based anomaly detection:
- Viewing rule clauses and variables
- Accessing ground truth data
- Generating prompts for LLMs
- Matching sensors to rule variables
- Plotting sensor data with ground truth and predictions

Usage
-----
Run commands using typer:
    python -m rule_logic_eval.cli [command] [options]

Available commands:
    clauses         - Display logical clauses for a rule
    variables       - List variables for a rule
    truth           - Show ground truth for an instance
    prompt_llama    - Generate LLM prompt for an instance
    sensors         - Match and display sensors for an instance
    plot_gt         - Plot sensor data with ground truth overlay
    plot_all_instances - Batch plot multiple instances
    embeddings      - Show variable-sensor matching scores
"""

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib import ticker
from rule_logic_eval.utils.parse import get_sensor_data_wide, set_flags
from rule_logic_eval.utils.plot import fmt_stamp
from rule_logic_eval.utils.prompts import llama70b
from rule_logic_eval.utils.varmatch import (match_variables, ot_match_vars,
                                            ssee_match_variables)

# Initialize Typer app
app = typer.Typer(help="Rule Logic Evaluation CLI")


@app.command()
def clauses(ruleid: str, rule_file: str = "rules.json") -> list:
    """
    Display logical clauses for a specific rule.
    
    Parameters
    ----------
    ruleid : str
        Rule identifier (e.g., 'AH00035')
    rule_file : str
        Path to rules JSON file (default: 'rules.json')
        
    Returns
    -------
    list
        List of logic clause strings
        
    Examples
    --------
    $ python -m rule_logic_eval.cli clauses AH00035
    """
    with open(rule_file, "rt") as f:
        rf = json.load(f)

    rule_info = rf[ruleid]
    stmt = rule_info["current logic"]
    print(stmt)
    return stmt


@app.command()
def variables(ruleid: str, rule_file: str = "rules.json") -> list:
    """
    Get the variables pertaining to a rule.
    
    Parameters
    ----------
    ruleid : str
        Rule identifier
    rule_file : str
        Path to rules JSON file (default: 'rules.json')
        
    Returns
    -------
    list
        List of variable names used in the rule
        
    Examples
    --------
    $ python -m rule_logic_eval.cli variables AH00035
    """
    with open(rule_file, "rt") as f:
        rf = json.load(f)

    rule_info = rf[ruleid]
    vars_list = rule_info["variables"]

    return vars_list


@app.command()
def truth(anomaly_id: str, truthfile: str = "ground_truth.json") -> dict:
    """
    Print the ground truth for an anomaly instance.
    
    Parameters
    ----------
    anomaly_id : str
        Unique identifier for the anomaly instance
    truthfile : str
        Path to ground truth JSON file (default: 'ground_truth.json')
        
    Returns
    -------
    dict
        Ground truth data including start/end timestamps
        
    Examples
    --------
    $ python -m rule_logic_eval.cli truth 5caf3b2b-7e81-4ff0-9dd7-6eb500b310a0
    """
    with open(truthfile, "rt") as tf:
        gt = json.load(tf)

    task_truth = next(filter(lambda x: x["id"] == anomaly_id, gt))
    print(task_truth)
    return task_truth


@app.command()
def prompt_llama(instance_file: str):
    """
    Build and print the LLM prompt for an instance.
    
    Generates a prompt suitable for Llama models based on rule clauses
    and available sensors.
    
    Parameters
    ----------
    instance_file : str
        Path to instance JSON file
        
    Examples
    --------
    $ python -m rule_logic_eval.cli prompt_llama instances/AH00035_abc123.json
    """
    with open(instance_file, "rt") as f:
        inst = json.load(f)
    
    ruleID = inst["rule"]
    sensors = list(inst["available_sensors"].keys())
    rule_clauses = clauses(ruleID)
    prompt = llama70b(rule_clauses, sensors)

    for line in prompt:
        print(line)


@app.command()
def sensors(
    instance_file: str, 
    matching_method: str = 'cossim', 
    print_df: bool = True, 
    pen_rule: float = 1.0
) -> tuple:
    """
    Print sensors of interest for an instance.
    
    Matches rule variables to available sensors using the specified method
    and displays the resulting sensor data.
    
    Parameters
    ----------
    instance_file : str
        Path to instance JSON file
    matching_method : str
        Method for matching variables to sensors:
        - 'cossim': Cosine similarity (default)
        - 'ot': Optimal transport
    print_df : bool
        Whether to print the DataFrame (default: True)
    pen_rule : float
        Penalty parameter for optimal transport (default: 1.0)
        
    Returns
    -------
    tuple
        (rule_variables, sensor_dataframe, matching_results)
        
    Examples
    --------
    $ python -m rule_logic_eval.cli sensors instances/AH00035_abc123.json
    $ python -m rule_logic_eval.cli sensors instances/AH00035_abc123.json --matching-method ot
    """
    with open(instance_file, "rt") as f:
        inst = json.load(f)
    
    ruleID = inst["rule"]
    sensors = list(inst["available_sensors"].keys())

    rulevars = variables(ruleID)
    
    # Match variables to sensors using specified method
    if matching_method == 'cossim':
        x = match_variables(rulevars, sensors)
    else:
        x = ot_match_vars(rulevars, sensors, pen_rule=pen_rule, pen_sensor=0.9)

    # Select columns of interest
    coi = ["timestamp_str"] + x['best_sensor'].to_list()

    instid = inst['id']
    df = get_sensor_data_wide(f"sensor_data/{instid}.json")

    if print_df:
        N, _ = df.shape
        pd.set_option("display.max_rows", N)
        print(df[coi])

    return rulevars, df[coi], x


def dict_unpack_helper(val):
    """
    Helper function to unpack dictionary values.
    
    If the value is a dictionary, returns the first value.
    Otherwise, returns the value as-is.
    
    Parameters
    ----------
    val : any
        Value to unpack
        
    Returns
    -------
    any
        Unpacked value
    """
    if type(val) == dict:
        return next(iter(val.values()))
    else:
        return val


@app.command()
def plot_gt(
    instance_file: str,
    truthfile: str = "ground_truth.json",
    preddir: str = None,
    figpath: str = None,
    matching_method: str = 'cossim',
    pen_rule: float = 1.0
):
    """
    Plot sensors of interest with ground truth overlay.
    
    Creates a multi-panel plot showing sensor data over time with
    ground truth anomaly windows highlighted. Optionally includes
    model predictions if available.
    
    Parameters
    ----------
    instance_file : str
        Path to instance JSON file
    truthfile : str
        Path to ground truth JSON file (default: 'ground_truth.json')
    preddir : str, optional
        Directory containing prediction results
    figpath : str, optional
        Path to save figure (if None, displays interactively)
    matching_method : str
        Method for matching variables to sensors (default: 'cossim')
    pen_rule : float
        Penalty parameter for optimal transport (default: 1.0)
        
    Examples
    --------
    $ python -m rule_logic_eval.cli plot_gt instances/AH00035_abc123.json
    $ python -m rule_logic_eval.cli plot_gt instances/AH00035_abc123.json --preddir experiments/model-arch/exp1
    """
    # Get sensor data and matching
    rulevars, df, matching = sensors(
        instance_file, 
        matching_method=matching_method, 
        pen_rule=pen_rule, 
        print_df=False
    )
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df['timestamp_str'] = pd.to_datetime(df['timestamp_str'])

    # Get rule clauses
    ruleid = instance_file.split('/')[-1].split('_')[0]
    rule_logic = clauses(ruleid)

    # Get ground truth data
    anomaly_id = instance_file.split('/')[-1].split('_')[-1][:-5]
    gt = truth(anomaly_id, truthfile)
    truth_begin = datetime.fromisoformat(gt["start_datetime_utc"])
    truth_end = datetime.fromisoformat(gt["end_datetime_utc"])
    truth_flags = set_flags(df['timestamp_str'], truth_begin, truth_end)

    # Get predicted data if available
    if preddir:
        expdir = ruleid + '_' + anomaly_id
        json_files = list((Path(preddir) / expdir).rglob("*.json"))
        if json_files:
            res = json_files[-1]
            with open(res, "r") as rp:
                pred = json.load(rp)
            timestamps = list(pred.keys())
            predict_flags = np.zeros_like(df['timestamp_str'], dtype=bool)
            if timestamps:
                for ts_string in timestamps:
                    tss = ts_string[1:-1].split(", ")
                    res_begin = datetime.fromisoformat(tss[0])
                    res_end = datetime.fromisoformat(tss[1])
                    predict_flags = np.logical_or(
                        set_flags(df['timestamp_str'], res_begin, res_end), 
                        predict_flags
                    )
        else:
            preddir = None

    # Create multi-panel plot
    nrows, ncols = (len(rulevars)) // 2 + (len(rulevars)) % 2, 2
    plt.rcParams.update({'axes.labelsize': 'small'})
    fig, axs = plt.subplots(nrows, ncols, dpi=100, squeeze=False)
    
    for i, col in enumerate(matching['best_sensor']):
        # Handle special cases
        if col == '${weatherRef} Condition':  # Handle None in weather conditions
            df[col] = df[col].ffill().bfill()
        elif 'Schedule' in col:
            df[col] = df[col].apply(lambda x: dict_unpack_helper(x))

        # Plot sensor data
        axs[i//2][i%2].scatter(x=df['timestamp_str'], y=df[col], s=2, c='k')

        # Overlay ground truth and predictions on secondary axis
        rax = axs[i//2][i%2].twinx()
        rax.step(
            df['timestamp_str'], 
            np.asarray(truth_flags), 
            label="ground truth", 
            c='red', 
            alpha=0.5, 
            zorder=-2
        )
        if preddir:
            rax.step(
                df['timestamp_str'], 
                np.asarray(predict_flags), 
                label="prediction", 
                c='blue', 
                alpha=0.5, 
                zorder=-2
            )
        rax.set_axis_off()

        # Format x-axis
        axs[i//2][i%2].xaxis.set_major_locator(ticker.MaxNLocator(4))
        axs[i//2][i%2].xaxis.set_major_formatter(fmt_stamp)

        # Set title showing variable-sensor mapping
        axs[i//2][i%2].set_title(f'{rulevars[i]} = {col}')
        axs[i//2][i%2].set_xlabel('')

    # Add overall title with rule information
    fig.suptitle(f'RULE {ruleid} @ {anomaly_id}:\n{chr(10).join(rule_logic)}')

    # Adjust figure size based on number of panels
    fig_width, fig_height = fig.get_size_inches()
    fig.set_size_inches((fig_width * 2, fig_height * (1.2 + 0.2 * (i//2))))
    fig.tight_layout()

    # Save or display
    if figpath:
        # error if directory doesn't exist
        if (os.path.exists(figpath) or os.path.isdir(figpath)) == False:
            fpth = Path.cwd().joinpath(figpath)
            raise FileNotFoundError(f"Directory {str(fpth)} not found; please create it.")

        # build the figure path and write the file
        pth = Path.cwd().joinpath(figpath, f"{ruleid}_{anomaly_id}.png")
        fig.savefig(pth)
        plt.close()
    else:
        plt.show()


@app.command()
def plot_all_instances(
    instance_dir: str,
    truthfile: str = "ground_truth.json",
    figpath: str = None,
    expdir: str = None,
    matching_method: str = 'cossim',
    pen_rule: float = 1.0
):
    """
    Batch plot multiple instances.
    
    Generates plots for all instances in a directory or listed in a text file.
    
    Parameters
    ----------
    instance_dir : str
        Path to directory containing instance files, or path to text file
        with instance names (one per line)
    truthfile : str
        Path to ground truth JSON file (default: 'ground_truth.json')
    figpath : str, optional
        Directory to save figures
    expdir : str, optional
        Directory containing prediction results
    matching_method : str
        Method for matching variables to sensors (default: 'cossim')
    pen_rule : float
        Penalty parameter for optimal transport (default: 1.0)
        
    Examples
    --------
    $ python -m rule_logic_eval.cli plot_all_instances instances/
    $ python -m rule_logic_eval.cli plot_all_instances good-instances.txt --figpath figures/
    """
    # Load instance list
    if instance_dir.endswith('.txt'):  # Instance dir is text file
        with open(instance_dir, 'r') as f:
            instance_list = f.read().split('\n')
    else:  # Instance dir is folder
        dirlist = os.listdir(instance_dir)
        instance_list = [str(Path.cwd().joinpath(instance_dir, ent )) for ent in dirlist]
    # Plot each instance
    for instance in instance_list:
        if instance.startswith('.'):
            continue
        plot_gt(
            instance,
            truthfile=truthfile,
            matching_method=matching_method,
            pen_rule=pen_rule,
            preddir=expdir,
            figpath=figpath
        )


@app.command()
def embeddings(
    instance_file: str,
    matching_method: str = 'cossim',
    pen_rule: float = 1.0,
    pen_sensor: float = 0.9,
    plot: bool = False
):
    """
    Display variable-sensor matching scores.
    
    Shows the embedding-based matching between rule variables and
    available sensors using the specified method.
    
    Parameters
    ----------
    instance_file : str
        Path to instance JSON file
    matching_method : str
        Method for matching:
        - 'cossim': Cosine similarity (default)
        - 'ssee': Semantic similarity
        - 'ot': Optimal transport
    pen_rule : float
        Penalty parameter for optimal transport (default: 1.0)
    pen_sensor : float
        Sensor penalty for optimal transport (default: 0.9)
    plot : bool
        Whether to generate visualization (default: False)
        
    Examples
    --------
    $ python -m rule_logic_eval.cli embeddings instances/AH00035_abc123.json
    $ python -m rule_logic_eval.cli embeddings instances/AH00035_abc123.json --matching-method ot --plot
    """
    with open(instance_file, "rt") as f:
        inst = json.load(f)
    
    ruleID = inst["rule"]
    sensors = list(inst["available_sensors"].keys())

    rulevars = variables(ruleID)
    
    # Compute matching using specified method
    if matching_method == 'cossim':
        x = match_variables(rulevars, sensors, plot=plot)
    elif matching_method == 'ssee':
        x = ssee_match_variables(rulevars, sensors, threshold=0.0)
    else:
        x = ot_match_vars(
            rulevars, 
            sensors, 
            pen_rule=pen_rule, 
            pen_sensor=pen_sensor, 
            plot=plot
        )
    
    print(x)


if __name__ == "__main__":
    app()
