import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from pandas import DataFrame

@dataclass
class EvalConfig:
    instances_folder: str
    data_folder: str
    rule_file: str
    truth_file: str
    eval_results_root: str


def set_flags(tstamps, begin, end):
    b = np.argmin([abs(x - begin) for x in tstamps])
    e = np.argmin([abs(x - end) for x in tstamps])
    flags = [False] * len(tstamps)
    flags[b:e] = [True] * (e - b)
    return flags


def process(exp_id, exp_root, config: EvalConfig):
    with open(f"{config.instances_folder}/{exp_id}.json", "r") as tp:
        task = json.load(tp)
        anomaly_id = task["id"]
        rule_id = task["rule"]

    with open(f"{config.data_folder}/{anomaly_id}.json", "r") as dp:
        data = json.load(dp)
        stamps = sorted(map(lambda x: datetime.fromisoformat(x), data.keys()))

    with open(config.truth_file, "r") as gp:
        truth = json.load(gp)
        task_truth = next(filter(lambda x: x["id"] == anomaly_id, truth))
        truth_begin = datetime.fromisoformat(task_truth["start_datetime_utc"])
        truth_end = datetime.fromisoformat(task_truth["end_datetime_utc"])
        truth_flags = set_flags(stamps, truth_begin, truth_end)

    erp = Path(exp_root)
    results = list((erp / exp_id).rglob("*.json"))
    if results:
        res = results[-1]
        with open(res, "r") as rp:
            truth = json.load(rp)
            timestamps = list(truth.keys())
            if timestamps:
                predict_flags = np.zeros_like(stamps, dtype=bool)
                # ts_string = timestamps[0]
                for ts_string in timestamps:
                    tss = ts_string[1:-1].split(", ")
                    res_begin = datetime.fromisoformat(tss[0])
                    res_end = datetime.fromisoformat(tss[1])
                    predict_flags = np.logical_or(
                        set_flags(stamps, res_begin, res_end), predict_flags
                    )
            else:
                return truth_flags, np.zeros_like(stamps, dtype=bool), stamps, rule_id
    else:
        return truth_flags, [], stamps, rule_id
    return truth_flags, predict_flags, stamps, rule_id


def get_sensor_data_wide(datafile) -> DataFrame:
    """gets sensor df in wide format"""

    with open(datafile, "rt") as f:
        dat = json.load(f)
    
    # time stamps are keys
    tstamps_str = list(dat.keys())

    dflist = []
    for ts in tstamps_str:

        tsdt = datetime.fromisoformat(ts)

        dfrow = dat[ts]

	# add timestamps to the dict
        toadd = {"timestamp_str": ts,
                 "timestamp" : tsdt}

        dfrow.update(toadd)

        dflist.append(dfrow)


    
    df = DataFrame(dflist)

    return df
