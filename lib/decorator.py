#!/usr/bin/env python
# coding: utf8

import time
from functools import wraps

def echo_func(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		print(f'calling {func.__name__} ..')
		return func(*args, **kwargs)

	return wrapper

def echo_func_count(func):
	counter = [0]

	@wraps(func)
	def wrapper(*args, **kwargs):
		print(f'calling {func.__name__} x{counter[0]} ..')
		counter[0] += 1
		return func(*args, **kwargs)

	return wrapper

def count_func_time(func):
	ts_avg = [0, 0]

	@wraps(func)
	def wrapper(*args, **kwargs):
		ts=time.process_time_ns()

		orig_return = func(*args, **kwargs)

		ts_this = time.process_time_ns() - ts
		ts_avg[0] += 1
		ts_avg[1] = (ts_avg[1] * (ts_avg[0]-1) + ts_this)/ts_avg[0]
		print(f'T({func.__name__}) us: '
				f'{ts_this/1000:.2f}  {ts_avg[1]/1000:.2f}:{ts_avg[0]}')

		return orig_return

	return wrapper

# vi: set ts=4 noexpandtab foldmethod=indent :
