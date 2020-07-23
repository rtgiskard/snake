#!/usr/bin/env python
# coding: utf8

import random

def random_seq(pool):
	for _i_ in range(0, len(pool)):
		idx = random.randint(0, len(pool)-1)
		yield pool.pop(idx)

# vi: set ts=4 noexpandtab foldmethod=indent :
