# Rule Logic Evaluation Package

A Python package for evaluating rule-based anomaly detection systems on sensor data. This package provides tools for matching rule variables to sensors, evaluating model predictions against ground truth, and visualizing results.

## Features

- **Rule Management**: Load and parse rule definitions with logical clauses
- **Variable Matching**: Multiple methods for matching rule variables to sensor data
  - Cosine similarity (embedding-based)
  - Optimal transport
  - Semantic similarity (SSEE)
- **Evaluation Metrics**: Compute precision, recall, F1 scores
- **Visualization**: Generate plots comparing predictions to ground truth
- **CLI Tools**: Command-line interface for common operations
- **Batch Processing**: Evaluate multiple experiments efficiently

## Installation

### From Source

```bash
cd rule-logic-eval
pip install -e .
```

### Dependencies

The package requires:

- Python 3.8+
- pandas
- numpy
- matplotlib
- typer
- scikit-learn (for metrics)

## Quick Start

### Command-Line Interface

The package provides a CLI for common operations:

```bash
# View rule clauses
python -m rule_logic_eval.cli clauses AH00035

# List rule variables
python -m rule_logic_eval.cli variables AH00035

# Show ground truth for an instance
python -m rule_logic_eval.cli truth 5caf3b2b-7e81-4ff0-9dd7-6eb500b310a0

# Match sensors to rule variables
python -m rule_logic_eval.cli sensors instances/AH00035_abc123.json

# Plot sensor data with ground truth
python -m rule_logic_eval.cli plot_gt instances/AH00035_abc123.json

# Batch plot all instances
python -m rule_logic_eval.cli plot_all_instances instances/ --figpath figures/
```

### Evaluation Pipeline

Run the full evaluation pipeline on experimental results:

```bash
python -m rule_logic_eval.main experiments/model-arch/exp1 dataset_config.json
```

### Programmatic Usage

```python
from rule_logic_eval.utils import eval, parse, plot
from rule_logic_eval.utils.varmatch import match_variables

# Load rule data
with open("rules.json") as f:
    rules = json.load(f)

# Match variables to sensors
rulevars = ["temperature", "pressure", "flow_rate"]
sensors = ["sensor_temp_01", "sensor_press_02", "sensor_flow_03"]
matches = match_variables(rulevars, sensors)

# Evaluate predictions
from sklearn.metrics import f1_score
f1 = f1_score(y_true=true_flags, y_pred=pred_flags)
```

## Project Structure

```
rule-logic-eval/
├── src/
│   └── rule_logic_eval/
│       ├── __init__.py          # Package initialization
│       ├── cli.py               # Command-line interface
│       ├── main.py              # Main evaluation pipeline
│       └── utils/               # Utility modules
│           ├── eval.py          # Evaluation metrics
│           ├── parse.py         # Data parsing utilities
│           ├── plot.py          # Visualization functions
│           ├── prompts.py       # LLM prompt generation
│           └── varmatch.py      # Variable matching algorithms
├── pyproject.toml               # Package configuration
└── README.md                    # This file
```

## Configuration

The evaluation pipeline requires a configuration file (JSON format):

```json
{
  "eval_results_root": "eval_results/",
  "rule_file": "rules.json",
  "ground_truth_file": "ground_truth.json",
  "sensor_data_path": "sensor_data/"
}
```

### Rules File Format

The `rules.json` file should contain rule definitions:

```json
{
  "AH00035": {
    "rule name": "Air Handler Temperature Control",
    "variables": ["status", "cooling_valve", "heating_valve", "supply_temp", "mixed_temp"],
    "current logic": [
      "IF status == 'On'",
      "AND cooling_valve > 0",
      "THEN supply_temp < mixed_temp"
    ],
    "complexity": 15.5
  }
}
```

### Ground Truth Format

The `ground_truth.json` file should contain anomaly windows:

```json
[
  {
    "id": "5caf3b2b-7e81-4ff0-9dd7-6eb500b310a0",
    "start_datetime_utc": "2023-01-15T10:00:00Z",
    "end_datetime_utc": "2023-01-15T12:00:00Z"
  }
]
```

## CLI Commands Reference

### `clauses`

Display logical clauses for a specific rule.

```bash
python -m rule_logic_eval.cli clauses <ruleid> [--rule-file rules.json]
```

### `variables`

List variables used in a rule.

```bash
python -m rule_logic_eval.cli variables <ruleid> [--rule-file rules.json]
```

### `truth`

Show ground truth for an anomaly instance.

```bash
python -m rule_logic_eval.cli truth <anomaly_id> [--truthfile ground_truth.json]
```

### `prompt_llama`

Generate LLM prompt for an instance.

```bash
python -m rule_logic_eval.cli prompt_llama <instance_file>
```

### `sensors`

Match and display sensors for an instance.

```bash
python -m rule_logic_eval.cli sensors <instance_file> \
  [--matching-method cossim|ot] \
  [--pen-rule 1.0]
```

### `plot_gt`

Plot sensor data with ground truth overlay.

```bash
python -m rule_logic_eval.cli plot_gt <instance_file> \
  [--truthfile ground_truth.json] \
  [--preddir experiments/model/exp1] \
  [--figpath figures/] \
  [--matching-method cossim]
```

### `plot_all_instances`

Batch plot multiple instances.

```bash
python -m rule_logic_eval.cli plot_all_instances <instance_dir> \
  [--truthfile ground_truth.json] \
  [--figpath figures/] \
  [--expdir experiments/model/exp1]
```

### `embeddings`

Display variable-sensor matching scores.

```bash
python -m rule_logic_eval.cli embeddings <instance_file> \
  [--matching-method cossim|ssee|ot] \
  [--plot]
```

## Variable Matching Methods

### Cosine Similarity (`cossim`)

Uses embedding-based cosine similarity to match variables to sensors. Fast and works well for semantic similarity.

### Optimal Transport (`ot`)

Uses optimal transport distance for matching. More robust but computationally intensive.

Parameters:

- `pen_rule`: Penalty for rule variables (default: 1.0)
- `pen_sensor`: Penalty for sensors (default: 0.9)

### Semantic Similarity (`ssee`)

Uses semantic similarity embeddings with a threshold-based approach.

## Evaluation Metrics

The package computes standard classification metrics:

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)

Where predictions and ground truth are compared at the timestamp level.

## Output Files

### Evaluation Results (`res.csv`)

CSV file containing metrics for all experiments:

```csv
exp_id,precision,recall,f1,accuracy,complexity
AH00035_abc123,0.95,0.87,0.91,0.92,15.5
```

### Trigger Plots (`*_triggers.png`)

Visualization showing:

- Sensor data over time
- Ground truth anomaly windows (red)
- Model predictions (blue)
- Variable-sensor mappings

## Examples

### Example 1: Evaluate a Single Instance

```bash
# View rule information
python -m rule_logic_eval.cli clauses AH00035
python -m rule_logic_eval.cli variables AH00035

# Plot with ground truth
python -m rule_logic_eval.cli plot_gt instances/AH00035_abc123.json
```

### Example 2: Batch Evaluation

```bash
# Run evaluation pipeline
python -m rule_logic_eval.main experiments/granite-8b-zeroshot/exp1 config.json

# Plot all instances with predictions
python -m rule_logic_eval.cli plot_all_instances instances/ \
  --expdir experiments/granite-8b-zeroshot/exp1 \
  --figpath figures/
```

### Example 3: Compare Matching Methods

```bash
# Cosine similarity
python -m rule_logic_eval.cli embeddings instances/AH00035_abc123.json \
  --matching-method cossim --plot

# Optimal transport
python -m rule_logic_eval.cli embeddings instances/AH00035_abc123.json \
  --matching-method ot --pen-rule 1.0 --plot
```


## Version History

| Version | Date          | Description                                            | 
| --------|---------------|--------------------------------------------------------|
| 1.0.1   |  3 April 2026 | semantic versioning; fix plot-all-instances (issue #3) |
| 0.1.0   |  2 April 2026 | initial version                                        | 