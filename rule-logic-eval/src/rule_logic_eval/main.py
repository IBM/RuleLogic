"""
Main Evaluation Pipeline
=========================

This module provides the main evaluation pipeline for rule-based anomaly detection.
It processes experimental results, computes evaluation metrics, and generates
visualizations comparing predictions against ground truth.

The pipeline:
1. Loads experiment configuration
2. Processes each experiment directory
3. Computes evaluation metrics (precision, recall, F1)
4. Generates trigger plots
5. Saves results to CSV

Usage
-----
Run the evaluation pipeline:
    python -m rule_logic_eval.main <experiment_root> <config_file>

Example:
    python -m rule_logic_eval.main experiments/model-arch/exp1 dataset_config.json

Configuration
-------------
The config file should be a JSON file containing:
- eval_results_root: Directory to save evaluation results
- rule_file: Path to rules.json
- ground_truth_file: Path to ground_truth.json
- sensor_data_path: Path to sensor data directory
"""

import json
import os
import warnings
from pathlib import Path

import pandas as pd
import typer
from rule_logic_eval.utils import eval, parse, plot


def main(
    experiment_root: str,
    config_file: str,
) -> None:
    """
    Run the evaluation pipeline on experimental results.
    
    This function processes all experiments in the specified root directory,
    evaluates predictions against ground truth, and generates reports.
    
    Parameters
    ----------
    experiment_root : str
        Path to root directory containing experiment subdirectories.
        Each subdirectory should contain prediction results in JSON format.
    config_file : str
        Path to JSON configuration file containing evaluation settings.
        
    Output
    ------
    Creates evaluation results in the configured output directory:
    - res.csv: CSV file with evaluation metrics for all experiments
    - *_triggers.png: Trigger plots for each experiment
    
    Examples
    --------
    >>> main("experiments/granite-8b-zeroshot/exp1", "dataset_config.json")
    
    Notes
    -----
    The function suppresses warnings during execution to avoid cluttering
    the output with non-critical messages.
    """
    # Load configuration
    with open(config_file, "r") as cf:
        _config = json.load(cf)
        config = parse.EvalConfig(**_config)

    # Setup output directory
    _exp_root = Path(experiment_root)
    eval_root = Path(config.eval_results_root) / _exp_root.name
    eval_root.mkdir(parents=True, exist_ok=True)

    # Suppress warnings during evaluation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        df = []

        # Process each experiment
        for exp in os.listdir(_exp_root):
            # Skip hidden files
            if exp.startswith("."):
                continue
                
            # Process experiment and get flags
            true_flags, pred_flags, tstamps, rule_id = parse.process(
                exp, experiment_root, config
            )
            
            # Only evaluate if predictions exist
            if len(pred_flags) > 0:
                # Compute evaluation metrics
                eval_dict = eval.eval(pred_flags, true_flags)
                plot.print_evals(eval_dict, exp)
                
                # Generate trigger plot
                plot.plot_rule_triggers(
                    tstamps,
                    true_flags,
                    pred_flags,
                    exp,
                    rule_id,
                    eval_root / f"{exp}_triggers.png",
                )

                # Add metadata to results
                eval_dict["exp_id"] = exp
                
                # Load rule complexity
                with open(config.rule_file, "r") as rf:
                    data = json.load(rf)
                    eval_dict["complexity"] = data[rule_id]["complexity"]
                    
                df.append(eval_dict)

    # Save consolidated results
    df = pd.DataFrame(df)
    df.to_csv(eval_root / "res.csv")
    
    print(f"\nEvaluation complete. Results saved to: {eval_root / 'res.csv'}")
    print(f"Total experiments processed: {len(df)}")


if __name__ == "__main__":
    typer.run(main)
