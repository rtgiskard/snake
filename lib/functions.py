#!/usr/bin/env python
# coding: utf8

import random

def random_seq(pool):
	for _i_ in range(0, len(pool)):
		idx = random.randint(0, len(pool)-1)
		yield pool.pop(idx)

def color_pool(n):
	for i in range(0, n):
		if   i/n < 1/6: yield '#00ff{:0>2}'.format(hex(int((6*i/n-0) * 0xff))[2:])
		elif i/n < 2/6: yield '#00{:0>2}ff'.format(hex(int((2-6*i/n) * 0xff))[2:])
		elif i/n < 3/6: yield '#{:0>2}00ff'.format(hex(int((6*i/n-2) * 0xff))[2:])
		elif i/n < 4/6: yield '#ff00{:0>2}'.format(hex(int((4-6*i/n) * 0xff))[2:])
		elif i/n < 5/6: yield '#ff{:0>2}00'.format(hex(int((6*i/n-4) * 0xff))[2:])
		else:			yield '#{:0>2}ff00'.format(hex(int((6-6*i/n) * 0xff))[2:])

# vi: set ts=4 noexpandtab foldmethod=indent :
