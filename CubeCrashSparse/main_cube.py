from dqn_ner_cube import ner_cube
from dqn_cer_cube import cer_cube
from dqn_her_cube import her_cube
import numpy as np

if __name__ == "__main__":
    runs = 5
    num_episodes = 3000
    # Hyperparameters of the Experience Replay
    batch_size = 32
    buf_size = 50_000

    # NER
    succes_rates_ner = []
    for run in range(runs):
        succes_rates_ner.append(
            ner_cube(
                n_episodes_=num_episodes
                ,batch_size_=batch_size
                ,buf_size_=buf_size
                )
            )

    # CER
    succes_rates_cer = []
    for run in range(runs):
        succes_rates_cer.append(
            cer_cube(
                n_episodes_=num_episodes
                ,batch_size_=batch_size
                ,buf_size_=buf_size
                )
            )
        
    # HER
    succes_rates_her = []
    for run in range(runs):
        succes_rates_her.append(
            her_cube(
                n_episodes_=num_episodes
                ,batch_size_=batch_size
                ,buf_size_=buf_size
                )
            )
    # TODO: Add PER
    # TODO: Compute Standard error of the mean for each interval
    # TODO: Plot means for all the runs with the confidence interval
    # TODO: Use 3 different hyperparameters for buf_size and batch_size

