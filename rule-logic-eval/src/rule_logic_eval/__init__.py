"""
Rule Logic Evaluation Package
==============================

A package for evaluating rule-based anomaly detection systems on sensor data.

This package provides tools for:
- Loading and parsing rule definitions
- Matching rule variables to sensor data
- Evaluating model predictions against ground truth
- Visualizing results and generating reports

Main Components
---------------
- cli: Command-line interface for common operations
- main: Main evaluation pipeline
- utils: Utility modules for parsing, evaluation, and plotting

Usage
-----
Command-line interface:
    python -m rule_logic_eval.cli [command] [options]

Programmatic usage:
    from rule_logic_eval.utils import eval, parse, plot
"""

__version__ = "0.1.0"
__author__ = "Rule Logic Eval Team"

# Package metadata
__all__ = ["cli", "main", "utils"]
