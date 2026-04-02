from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem

def _ot_preprocess(ls: list):
    ls = np.array(ls, dtype=np.float32)
    return ls / np.sum(ls)

def ot_score(source: list, target: list, epsilon=None):
    source = _ot_preprocess(source)
    target = _ot_preprocess(target)

    support = np.linspace(0, 1, source.shape[0]).reshape(-1, 1) # time is regularly spaced

    if np.sum(source) > 0.0:
        geom = pointcloud.PointCloud(support, epsilon=epsilon)
        problem = linear_problem.LinearProblem(geom, source, target)
        output = sinkhorn.Sinkhorn()(problem)
        return float(output.reg_ot_cost)
    return None

def eval(pred, truth):
    return {
        "acc": accuracy_score(y_pred=pred, y_true=truth),
        "prec": precision_score(y_pred=pred, y_true=truth),
        "rec": recall_score(y_pred=pred, y_true=truth),
        "f1": f1_score(y_pred=pred, y_true=truth),
        "ot": ot_score(source=pred, target=truth),
        
    }