from pandas import DataFrame
from sentence_transformers import SentenceTransformer, util

import numpy as np
import re


from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem

embedding_model="all-mpnet-base-v2"
model = SentenceTransformer(embedding_model)

def calculate_embeddings(rule_variables: list, asset_sensors: list):
    sensor_embeddings = model.encode(asset_sensors)
    rulevar_embeddings = model.encode(rule_variables)

    # Compute cosine similarities
    cosine_scores = util.cos_sim(rulevar_embeddings, sensor_embeddings)

    return sensor_embeddings, rulevar_embeddings, cosine_scores

def match_variables(
    rule_variables: list, asset_sensors: list, plot:bool=False
) -> DataFrame:
    """Matches variables mentioned in a rule to sensors on the asset.
    Returns best match for each rule variable using cosine similarity.
    """

    _, _, cosine_scores = calculate_embeddings(rule_variables, asset_sensors)
    print(plot)

    Nvar = len(rule_variables)
    best_matches = []
    for i in range(Nvar):

        rv = rule_variables[i]
        # index of max score
        j = cosine_scores[i, :].argmax()

        # best_sensor
        bs = asset_sensors[j]
        # similarity score
        cossim = float(cosine_scores[i, j])

        res = {"rule_variable": rv, "best_sensor": bs, "cosine_similarity": cossim}
        best_matches.append(res)

    if plot:
        plt.imshow(cosine_scores)
        plt.yticks(ticks=range(Nvar), labels=rule_variables)
        plt.xticks(ticks=range(len(asset_sensors)), labels=asset_sensors, rotation=60, ha='right')
        plt.title('Cosine Similarities between Var Pairs')
        plt.tight_layout()
        plt.show()

    mdf = DataFrame(best_matches)
    return mdf

import matplotlib.pyplot as plt

def ot_match_vars(rule_variables: list, asset_sensors: list, pen_rule=0.999, pen_sensor=0.999, plot=False) -> DataFrame:
    """Matches variables mentioned in a rule to sensors on the asset.
    Returns best match for each rule variable using unbalanced OT.
    """

    sensor_embeddings, rulevar_embeddings, _ = calculate_embeddings(rule_variables, asset_sensors)

    geom = pointcloud.PointCloud(rulevar_embeddings, sensor_embeddings)
    problem = linear_problem.LinearProblem(geom, tau_a=pen_rule, tau_b=pen_sensor)
    output = sinkhorn.Sinkhorn()(problem)


    if plot:
        plt.imshow(output.matrix)
        plt.yticks(ticks=range(len(rule_variables)), labels=rule_variables)
        plt.xticks(ticks=range(len(asset_sensors)), labels=asset_sensors, rotation=60, ha='right')
        plt.title('Transport Plan between Var Sets')
        plt.tight_layout()
        plt.show()

    matches = []
    for i, var in enumerate(rule_variables):
        j = np.argmax(output.matrix[i])
        matches.append({'rule_variable': var, 'best_sensor': asset_sensors[j], 'confidence': output.matrix[i,j] / np.sum(output.matrix[i,:])})

    mdf = DataFrame(matches)
    return mdf

def ssee_match_variables(
    rule_variables: list, asset_sensors: list, threshold: float=0.0
) -> DataFrame:
    """Matches variables mentioned in a rule to sensors on the asset.
    Returns best match for each rule variable using SSEE (based on ranked cosine similarity).
    """

    # rule vars = candidate, asset sensors = gold
    _, _, cosine_scores = calculate_embeddings(rule_variables, asset_sensors)

    gold_matches = {e: [] for e in rule_variables} # list of matches for each candidate entry
    for i, e_c in enumerate(rule_variables):
        for j, e_g in enumerate(asset_sensors):
            if cosine_scores[i, j] >= threshold:
                gold_matches[e_c].append((e_g, float(cosine_scores[i, j])))

    for e_c in gold_matches.keys(): # sort gold matches
        gold_matches[e_c] = sorted(
            gold_matches[e_c],
            key=lambda x: (
                x[1]
            ),
            reverse=True
        )
    
    sorted_candidates = sorted( # sort candidates by quality of the best match
        rule_variables,
        key=lambda e_c: (
            gold_matches[e_c][0][1]
            if gold_matches[e_c] else 0.0
        ),
        reverse=True
    )

    used_gold = set()
    matches = []
    for e_c in sorted_candidates:
        for (e_g, cossim) in gold_matches[e_c]:
            if e_g not in used_gold:
                used_gold.add(e_g)
                matches.append({'rule_variable': e_c, 'best_sensor': e_g, 'cossim': cossim})
                break

    mdf = DataFrame(matches)
    return mdf.set_index('rule_variable').loc[rule_variables].reset_index() # sort into matching order

def greplist(pattern, strings:list, REflags=re.IGNORECASE) -> bool:
    """See if a pattern exists anywhere in a list of strings"""
    has_pattern = False

    for s in strings:
        mat = re.search(pattern, s, REflags)
        if mat:
            has_pattern = True
            break

    return has_pattern

