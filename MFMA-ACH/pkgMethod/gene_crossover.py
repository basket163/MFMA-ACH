import numpy as np

import pkgMethod
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge

def exchange_parts(a, b, n):
	return a[:n] + b[n:], b[:n] + a[n:]


def multi_point(left, right, probability, m=2):
	left = left.chromosomes()
	right = right.chromosomes()
	cross_points = sorted(np.random.choice(range(1, len(left[0])), m))
	if np.random.random_sample() > probability:
		return left, right
	off_left = []
	off_right = []
	for a, b in zip(left, right):
		for cp in cross_points:
			a, b = exchange_parts(a, b, cp)
		off_left.append(a)
		off_right.append(b)
	return off_left, off_right


def one_point(left, right, probability):
	return multi_point(left, right, probability, m=1)


def uniform(left, right, probability):
	left = left.chromosomes()
	right = right.chromosomes()
	if np.random.random_sample() > probability:
		return left, right
	off_left = []
	off_right = []
	for a, b in zip(left, right):
		chrs = np.array([np.random.choice(genes, 2, replace=False) for genes in zip(a, b)])
		off_left.append("".join(chrs[:, 0]))
		off_right.append("".join(chrs[:, 1]))
	return off_left, off_right


def shuffler(left, right, probability):
	left = [list(x) for x in left.chromosomes()]
	right = [list(x) for x in right.chromosomes()]
	if np.random.random_sample() > probability:
		return left, right
	off_left = []
	off_right = []
	sample = list(range(1, len(left[0])))
	cp = np.random.choice(range(1, len(left[0])))
	for a, b in zip(left, right):
		for i_l, i_r in zip(np.random.choice(sample, len(sample), False), np.random.choice(sample, len(sample), False)):
			a[i_l], b[i_r] = b[i_r], a[i_l]
		a, b = exchange_parts("".join(a), "".join(b), cp)
		off_left.append(a)
		off_right.append(b)
	return off_left, off_right


def reduced_surrogate(left, right, probability):
	#print("corss----")
	#print(left)
	#print("--")
	#print(right)
	left = left.chromosomes()  #['+00000000000010', '+00000000000001', ...]
	#print("chorm--")
	#print(left)
	right = right.chromosomes()  #'+00000000000001'
	#print("--")
	#print(right)
	if np.random.random_sample() > probability:
		return left, right
	off_left = []
	off_right = []
	#print("choice from:")
	choiceFrom = range(1, len(left))
	#print(choiceFrom)
	cp = np.random.choice( choiceFrom )
	#print("choice cp is {}".format(cp))
	off_left, off_right = exchange_parts(left, right, cp)
	#print("after corss")
	#print(off_left)
	#print("--")
	#print(off_right)
	#input()
	return off_left, off_right
	'''
	for a, b in zip(left, right):
		cp = np.random.choice(
			[i for i, (aa, bb) in enumerate(zip(a, b)) if aa != bb] or range(1, len(a))
			)
		print("a {}, b {} cp {}".format(a,b,cp))
		input()
		a, b = exchange_parts(a, b, cp)
		off_left.append(a)
		off_right.append(b)
	'''
	#print("off---")
	#print(off_left)
	#print("--")
	#print(off_right)
	#input()
	