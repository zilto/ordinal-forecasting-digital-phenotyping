from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error
from imblearn.metrics import macro_averaged_mean_absolute_error

from hamilton.function_modifiers import config, parameterize_sources


@config.when(task="binary_classification")
def scorer_name__binary() -> str:
    return "balanced_accuracy"  # can compute balanced accuracy from mamae


@config.when(task="ordinal_regression")
def scorer_name__ordinal() -> str:
    return "mamae"


def scorer(scorer_name: str) -> Callable:
    if scorer_name == "accuracy":
        return accuracy_score
    elif scorer_name == "balanced_accuracy":
        return balanced_accuracy_score
    elif scorer_name == "mae":
        return mean_absolute_error
    elif scorer_name == "mamae":
        return macro_averaged_mean_absolute_error
    else:
        raise ValueError(f"Invalid `scorer_name`: {scorer_name}")
    

def higher_is_better(scorer_name: str) -> bool:
    if scorer_name in ["accuracy", "balanced_accuracy"]:
        return True
    elif scorer_name in ["mae", "mamae"]:
        return False
    else:
        raise ValueError(f"Invalid `scorer_name`: {scorer_name}")
    

@parameterize_sources(
    train_score=dict(y_pred="y_pred_train", y_true="y_train", scorer="scorer"),
    eval_score=dict(y_pred="y_pred_eval", y_true="y_eval", scorer="scorer"),
)
def best_score(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    scorer: Callable,
) -> float:
    return scorer(y_true=y_true, y_pred=y_pred)


def distribution_train(y_train: np.ndarray) -> tuple:
    unique_val, count_val = np.unique(y_train, return_counts=True)
    probabilities = count_val/count_val.sum()
    return unique_val, probabilities


def bootstrap_scores(
    y_eval: np.ndarray,
    distribution_train: tuple,
    scorer: Callable,
    n_random_samples: int = 1000,
) -> np.ndarray:
    eval_set_size = y_eval.shape[0]
    unique_val, probabilities = distribution_train

    scores = []
    for _ in range(n_random_samples):
        random_preds = np.random.choice(
            unique_val,
            size=eval_set_size,
            p=probabilities
        )
        score = scorer(y_true=y_eval, y_pred=random_preds)
        scores.append(score)

    return scores


def test_alpha__base(alpha: float) -> float:
    return alpha


@config.when(correction="bonferroni")
def test_alpha__bonferroni(alpha: float, n_comparisons: int) -> float:
    return (alpha / n_comparisons)


def bootstrap_confidence_criterion(
    bootstrap_scores: np.ndarray,
    higher_is_better: bool,
    test_alpha: float,
) -> float:
    if higher_is_better:
        confidence_level = 1 - test_alpha
    else:
        confidence_level = test_alpha

    return np.quantile(bootstrap_scores, confidence_level).astype(float)


def is_eval_score_significant(
    eval_score: float,
    bootstrap_confidence_criterion: float,
    higher_is_better: bool,
) -> bool:
    if higher_is_better:
        return eval_score > bootstrap_confidence_criterion
    
    return eval_score < bootstrap_confidence_criterion


def performance_metrics(
    scorer_name: str,
    eval_score: float,
    bootstrap_confidence_criterion: float,
    is_eval_score_significant: bool,
    higher_is_better: bool,
) -> dict:
    return dict(
        name=scorer_name,
        result=eval_score,
        confidence_criterion=bootstrap_confidence_criterion,
        is_significant=bool(is_eval_score_significant),
        higher_is_better=higher_is_better,
    )