import numpy as np

import pkgMethod
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge

def single_point(individuum, probability, low, high):
	#print(individuum)
	chromosomes = [list(x) for x in individuum]
	#print("chormo:")
	#for c in chromosomes:
	#	print(c)
	#new_chrs = []
	new_ints = []
	'''
	for c in chromosomes:
		if np.random.random_sample() < probability:
			index = np.random.randint(1, len(c))
			c[index] = "1" if c[index] == "0" else "0"
		new_chrs.append(int("".join(c), 2))
	print("new_chrs:")
	print(new_chrs)
	print(low)
	print(high)
	'''
	for i in chromosomes:
		ii = int("".join(i), 2)
		if np.random.random_sample() < probability:
			ii = np.random.choice([low,high-1])
		new_ints.append(ii)
	new_ints = np.array(new_ints)
	#print("new_ints:")
	#print(new_ints)
	#input()
	return new_ints