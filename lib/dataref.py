#!/usr/bin/env python
# coding: utf8

from .datatypes import Vector

class VECTORS:
	UP = Vector(0,-1)
	DOWN = -UP
	LEFT = UP.T
	RIGHT = -LEFT

	def __iter__(self):
		return iter((self.UP, self.RIGHT, self.DOWN, self.LEFT))

class TransMatrix:
	""" matrix: [col1, col2] """
	ROTATE_LEFT = [[0, 1], [-1, 0]]
	ROTATE_RIGHT = [[0, -1], [1, 0]]

class DEBUG:
	pause = False

# vi: set ts=4 noexpandtab foldmethod=indent :
