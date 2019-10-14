from dqn_cer_bitflip import cer_bitflip
from dqn_her_bitflip import her_bitflip
from dqn_ner_bitflip import ner_bitflip
import numpy as np

if __name__ == "__main__":
	runs = 5
	num_episodes = 3000
	batch_size = 128
	buf_size = 50.000

	# NER
	succes_rates_ner = []
	for run in range(runs):
		succes_rates_ner.append(
			ner_bitflip(
				n_episodes_=num_episodes
				,batch_size_=batch_size
				,buf_size_=buf_size
				)
			)
	# CER
	succes_rates_cer = []
	for run in range(runs):
		succes_rates_cer.append(
			cer_bitflip(
				n_episodes_=num_episodes
				,batch_size_=batch_size
				,buf_size_=buf_size
				)
			)
	# HER
	succes_rates_her = []
	for run in range(runs):
		succes_rates_her.append(
			her_bitflip(
				n_episodes_=num_episodes
				,batch_size_=batch_size
				,buf_size_=buf_size
				)
			)