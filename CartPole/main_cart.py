from dqn_ner_cart import ner_cart
from dqn_cer_cart import cer_cart
from dqn_her_cart import her_cart
from dqn_per_cart import per_cart
import numpy as np

if __name__ == "__main__":
    runs = 5
    num_episodes = 3000
    # Hyperparameters of the Experience Replay
    batch_size = 32
    buf_size = 50_000

    # NER
    scores_ner = []
    for run in range(runs):
        _, score_rate_ner = (
                ner_cart(
                n_episodes_=num_episodes
                ,batch_size_=batch_size
                ,buf_size_=buf_size
                )
            )
        scores_ner.append(score_rate_ner)

    # CER
    scores_cer = []
    for run in range(runs):
        _, score_rate_cer = (
                cer_cart(
                n_episodes_=num_episodes
                ,batch_size_=batch_size
                ,buf_size_=buf_size
                )
            )
        scores_cer.append(score_rate_cer)

    # HER
    scores_her = []
    for run in range(runs):
        _, score_rate_her = (
                her_cart(
                n_episodes_=num_episodes
                ,batch_size_=batch_size
                ,buf_size_=buf_size
                )
            )
        scores_her.append(score_rate_her)

    # PER
    """
    scores_per = []
    for run in range(runs):
        _, score_rate_per = (
                per_cart(
                n_episodes_=num_episodes
                ,batch_size_=batch_size
                ,buf_size_=buf_size
                )
            )
        scores_per.append(score_rate_per)
    """
    # TODO: Fix PER
    # TODO: Compute Standard error of the mean for each interval
    # TODO: Plot means for all the runs with the confidence interval
    # TODO: Use 3 different hyperparameters for buf_size and batch_size

