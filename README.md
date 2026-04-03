# RuleLogic: A dataset for evaluating generated code for rule-based equipment monitoring

This repository contains the _RuleLogic_ dataset, as decribed in the paper _RuleLogic: A dataset for evaluating generated code for rule-based equipment monitoring_. The data has been cleaned and organized to facilitate streamlined testing and validation of AI agents on industrial asset monitoring tasks.

## Overview

The top-level directory provides a collection of instances with their corresponding sensor data and ground truth annotations. This dataset is ideal for:

- Quick evaluation of monitoring rule implementations
- Baseline policy testing
- Algorithm validation
- Performance benchmarking

## Directory Structure

```
./
├── ground_truth.json          # Anomaly window annotations for all instances
├── rules.json                 # Monitoring rule definitions
├── instances/                 # Instance definitions
│   └── {rule_id}_{anomaly_id}.json
└── sensor_data/              # Time-series sensor measurements
    └── {anomaly_id}.json
```

## Data Components

### Ground Truth (`ground_truth.json`)

Contains annotated anomaly windows for all instances in the dataset. Each entry defines a time period during which anomalous behavior was observed in the industrial system.

**Format:**

```json
[
  {
    "id": "5caf3b2b-7e81-4ff0-9dd7-6eb500b310a0",
    "start_datetime_utc": "2022-03-02T12:00:00.000Z",
    "end_datetime_utc": "2022-03-02T15:15:00.000Z"
  }
]
```

**Fields:**

- `id`: Unique identifier for the anomaly instance (UUID format)
- `start_datetime_utc`: Beginning of the anomaly window (ISO 8601 UTC)
- `end_datetime_utc`: End of the anomaly window (ISO 8601 UTC)

### Rules (`rules.json`)

Contains the monitoring rule definitions that are applied to sensor data to detect anomalous behavior. Each rule specifies the conditions that must be met for an anomaly to be flagged.

**Format:**

```json
{
  "AH00035": {
    "rule name": "AHU - Static Pressure in Duct When Unit Commanded Off",
    "current logic": [
      "Supply Fan Status = 0",
      "Subtype NOT MULTI or VAV",
      "Duct Static Pressure > 0.2 in H2O",
      "Met for 3 Hours"
    ],
    "utterance": "The behaviour of non MULTI or non VAV type asset is anomalous if the supply fan is off and the static pressure in the duct is greater than .2 H2O for more than 3 hours straight.",
    "variables": [
      "Supply Fan Status",
      "Duct Static Pressure"
    ],
    "complexity": 8.0156097709
  }
}
```

**Fields:**

- `rule name`: Human-readable name describing the monitoring rule
- `current logic`: Array of logical conditions that define the rule
  - Conditions specify thresholds, comparisons, and temporal requirements
  - Often includes duration requirements (e.g., "Met for 3 Hours")
- `utterance`: Natural language description of the rule's intent and behavior
- `variables`: List of sensor variables referenced in the rule logic
- `complexity`: Numeric score indicating rule complexity (see Complexity Levels section)

### Instances Directory (`instances/`)

Contains test instance files that define the monitoring scenarios. Each file specifies:

- The monitoring rule to apply
- The anomaly ID for corresponding sensor data
- Available sensors and their data types
- System metadata (type, subtype, status)

**Filename Convention:** `{rule_id}_{anomaly_id}.json`

**Example:** `AH00035_5caf3b2b-7e81-4ff0-9dd7-6eb500b310a0.json`

**File Structure:**

```json
{
  "id": "5caf3b2b-7e81-4ff0-9dd7-6eb500b310a0",
  "rule": "AH00035",
  "available_sensors": {
    "0fiv0Spl Heat Transfer Energy (Calc)": "Number",
    "hHt98i2w Heating Valve %": "Number",
    "IFgpnW3M Supply Fan Status": "Bool",
    "Status": "Bool",
    "type": "Str",
    "subType": "Str"
  }
}
```

**Sensor Data Types:**

- `Number`: Numeric measurements (temperatures, pressures, percentages, etc.)
- `Bool`: Boolean states (on/off, true/false)
- `Str`: String values (equipment types, identifiers)

### Sensor Data Directory (`sensor_data/`)

Contains time-series sensor measurements for each anomaly instance. Data is stored in JSON format with timestamps as keys.

**Filename Convention:** `{anomaly_id}.json`

**Example:** `5caf3b2b-7e81-4ff0-9dd7-6eb500b310a0.json`

**File Structure:**

```json
{
  "2022-03-02T06:00:00.000Z": {
    "0fiv0Spl Heat Transfer Energy (Calc)": 0.0,
    "hHt98i2w Heating Valve %": 0,
    "IYdjRsyy Cooling Valve %": 0,
    "xTb764tw Mixed Air Temperature": 72.9052963257,
    "IFgpnW3M Supply Fan Status": null,
    "Status": false,
    "type": "AHU",
    "subType": ""
  },
  "2022-03-02T06:15:00.000Z": {
    ...
  }
}
```

**Characteristics:**

- **Sampling Interval:** Typically 15 minutes
- **Timestamp Format:** ISO 8601 UTC
- **Missing Values:** Represented as `null`

## Usage Examples

### Loading Rules

```python
import json

# Load rule definitions
with open('rules.json', 'r') as f:
    rules = json.load(f)

# Access specific rule
rule_id = "AH00035"
rule = rules[rule_id]
print(f"Rule: {rule['rule name']}")
print(f"Variables: {rule['variables']}")
print(f"Complexity: {rule['complexity']}")
print(f"Logic: {rule['current logic']}")
```

### Loading Ground Truth

```python
import json

# Load ground truth annotations
with open('ground_truth.json', 'r') as f:
    ground_truth = json.load(f)

# Find specific anomaly window
anomaly_id = "5caf3b2b-7e81-4ff0-9dd7-6eb500b310a0"
anomaly = next(a for a in ground_truth if a['id'] == anomaly_id)
print(f"Anomaly window: {anomaly['start_datetime_utc']} to {anomaly['end_datetime_utc']}")
```

### Loading Instance Data

```python
import json
# Load instance definition
instance_file = 'instances/AH00035_5caf3b2b-7e81-4ff0-9dd7-6eb500b310a0.json'
with open(instance_file, 'r') as f:
    instance = json.load(f)

rule_id = instance['rule']
available_sensors = instance['available_sensors']
print(f"Rule: {rule_id}, Sensors: {len(available_sensors)}")
```

### Loading Sensor Data

```python
import pandas as pd
import json

# Load sensor time-series data
sensor_file = 'sensor_data/5caf3b2b-7e81-4ff0-9dd7-6eb500b310a0.json'
with open(sensor_file, 'r') as f:
    sensor_data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame.from_dict(sensor_data, orient='index')
df.index = pd.to_datetime(df.index)
df = df.sort_index()

print(f"Data shape: {df.shape}")
print(f"Time range: {df.index.min()} to {df.index.max()}")
```

### Using with rule-logic-eval Package

Install the package using pip:
```bash
pip install -e rule-logic-eval
```

```bash
# View rule information
rulogic clauses AH00035
rulogic variables AH00035

# View instance details
rulogic sensors instances/AH00035_5caf3b2b-7e81-4ff0-9dd7-6eb500b310a0.json

# Plot sensor data with ground truth overlay
rulogic plot-gt instances/AH00035_5caf3b2b-7e81-4ff0-9dd7-6eb500b310a0.json

# Batch plot all instances
rulogic plot-all-instances instances/ --figpath figures/
```

Or, alternatively with uv:
```bash
uv init --bare --python 3.13    # assuming no .venv yet
uv add rule-logic-eval          # adds the package
uv run rulogic clauses AH00035  # run a sample command
```

## Data Quality

### Completeness

- All instances have corresponding sensor data files
- All sensor data files have matching ground truth annotations
- All instances reference rules defined in `rules.json`

### Data Integrity

- Timestamps are consistently formatted in UTC
- Missing values are explicitly marked as `null`
- Sensor names are anonymised versions of original system identifiers

## Complexity Levels

Rules are categorized by complexity score, which reflects the number of variables, logic clauses, and conditional statements:

- **Low (0-15)**: Simple conditions, few variables (e.g., single threshold checks)
- **Medium (15-35)**: Multiple conditions, moderate logic (e.g., combined sensor checks)
- **High (35-250)**: Complex nested logic, many variables (e.g., multi-stage conditions)

**Complexity Formula:** `sqrt((n/k)² + (n*k)²)`

- `n` = number of unique variables
- `k` = logic length (with penalties for OR/IF statements)
