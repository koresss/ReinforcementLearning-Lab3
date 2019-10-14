from dqn_ner_cube import ner_cube
from dqn_cer_cube import cer_cube
from dqn_her_cube import her_cube


if __name__ == '__main__':
    n = 5
    ner_success_rate = []
    cer_success_rate = []
    her_success_rate = []
    # NER
    for i in range(n):
        ner_success_rate.append(ner_cube(3000, 20000, 32))
    # CER
    for i in range(n):
        cer_success_rate.append(cer_cube(3000, 20000, 32))
    # HER
    for i in range(n):
        her_success_rate.append(her_cube(3000, 20000, 32))
