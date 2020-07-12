#!/usr/bin/env python
# coding: utf8

import random
import numpy as np

from .datatypes import Vector
from .decorator import *
from .dataref import *
from .functions import *

"""
todo:
. 封闭区间最大长度(重扫描时间点长度二分)
. 如到达食物后，围成空间不足长度，则游荡会再查看
"""

class Snake:
	def __init__(self, width=40, height=40):
		self.area_w = width
		self.area_h = height

		# food, body, aim, seed
		self.snake_reset()

	@property
	def length(self):
		return len(self.body)

	@property
	def area_size(self):
		return self.area_w * self.area_h

	@property
	def vec_head2food(self):
		return self.food - self.head

	@property
	def head(self):
		return self.body[0]
	@head.setter
	def head(self, point):
		self.head.x = point.x
		self.head.y = point.y

	def is_died(self):
		return self.head in self.body[1:] or not self.is_inside(self.head)

	def is_inside(self, point):
		return ( point.x >= 0 and point.x < self.area_w
				and point.y >= 0 and point.y < self.area_h )

	def is_full(self):
		return self.length == self.area_size

	def is_aim_valid(self, aim):
		head_next = self.head + aim
		return self.is_inside(head_next) and head_next not in self.body

	def is_aim_right(self, aim):
		# todo: more check
		if self.is_aim_valid(aim):
			if self.graph is not None:
				return self.graph_aim_deadend(self.graph, self.head, aim)
			else:
				return True

		return False

	def snake_reset(self):
		self.body = [ Vector(int(self.area_w/2), int(self.area_h/2)) ]
		self.aim = VECTORS.DOWN

		self.food = self.new_food()
		self.graph = None
		self.graph_path = None

		# re-seed on reset
		random.seed()

	def area_resize(self, width, height, reset=False):
		if not reset:
			for pos in [ self.food, *self.body ]:
				# the original pos is inside
				if pos.x >= width or pos.y >= height:
					return False

		self.area_w = width
		self.area_h = height

		return True

	def move(self, aim=None):
		""" move after died may revive the snake """
		if aim: # set new direction
			self.aim = aim
		else:	# keep moving
			aim = self.aim

		# insert the new head
		self.body.insert(0, self.head + aim)

		# if got food, generate new, then reset graph
		if self.head == self.food:
			self.food = self.new_food()
			self.graph = None
			self.graph_path = None
		else:
			self.body.pop()

		# return after pop even died
		if self.is_died():
			return False

		return True

	def new_food(self):
		if self.is_full():
			return None

		space_id = random.randint(0, self.area_size - self.length - 1)

		for seq in range(0, self.area_size):
			vec_seq = Vector(seq % self.area_w, seq // self.area_w)

			if vec_seq in self.body:
				continue

			if space_id == 0:
				return vec_seq
			else:
				space_id -= 1

	#@count_func_time
	def graph_scan(self, md_fast=True):
		# in np, it's (row, col), it's saved/read in transposed style
		graph = np.zeros((self.area_w, self.area_h), dtype=np.int32)
		pipe = [self.head]

		while len(pipe) > 0:
			elem = pipe.pop(0)

			"""with md_fast False, it's possible to reach food not currently connected"""
			if md_fast and elem == self.food:
				break

			# distance from head to the elem
			dist_elem = graph[elem.x, elem.y]

			# neighbors of head
			neighbors = [ elem + aim for aim in VECTORS() ]

			if dist_elem == 0:
				# the head, can not go backward directly
				neighbors.remove(self.head - self.aim)
				body_check = self.body[1:]
			else:
				# the tail have moved forward
				body_check = self.body[:-dist_elem]

			for nb in random_seq(neighbors):
				if self.is_inside(nb) and nb not in body_check:
					if graph[nb.x, nb.y] == 0:
						"""
						with condition `raph[nb.x, nb.y] > dist_elem+1`,
						this will reverse update the filled path which will
						make the graph accurate to reflect the distance,
						but useless for path search
						"""
						graph[nb.x, nb.y] = dist_elem + 1
						# add to pipe
						pipe.append(nb)

		self.graph = graph

	#@count_func_time
	def graph_path_scan(self, dest, md_rev=True):
		if self.graph[dest.x, dest.y] == 0:
			"""food not reacheable, begin survive mode"""
			path = self.graph_path_survive(self.graph, self.head, dest)
		else:
			"""
			attention for start:
				if it's snake's head, the original graph[head] is likely to be
				the length+1, which is decided by the graph scan logic, this
				may be useful for survive mode, but do no help to reach food,
				and cause trouble for path search, just override here
			"""
			self.graph[self.head.x, self.head.y] = 0

			if md_rev:
				path = self.graph_path_gen_rev(self.graph, self.head, dest)
			else:
				path = self.graph_path_dfs(self.graph, self.head, dest)

		# keep only current path
		for i in range(len(path)):
			path[i] = path[i][0]

		self.graph_path = path

	def graph_path_dfs(self, graph, start, end):
		"""正向深度搜索，无优化
		大范围 DFS 时间复杂度与路径深度呈指数关系
		"""
		# the stack
		path = [[start]]

		while len(path) > 0:
			elem = path[-1][0]

			if elem == end:
				break

			nbs = []
			for nb in random_seq([ elem + aim for aim in VECTORS() ]):
				# if can move forward to the neighbor, add to nbs
				if self.is_inside(nb) and graph[nb.x, nb.y] == graph[elem.x, elem.y]+1:
					nbs.append(nb)

			if len(nbs) > 0 and graph[elem.x, elem.y] < graph[end.x, end.y]:
				path.append(nbs)
			else:			# the path is died or beyond end
				# revert to last branch
				while len(path) > 0 and len(path[-1]) == 1:
					path.pop()

				# remove current choice from the branch
				if len(path) > 0:
					path[-1].pop(0)

		return path

	def graph_path_gen_rev(self, graph, start, end):
		"""反向搜索，利用已有信息，提高效率"""
		# the stack
		path = [[end]]

		while len(path) > 0:
			elem = path[0][0]

			# there should be only one path from start to end with the generated graph
			if graph[elem.x, elem.y] == graph[start.x, start.y] + 1:
				path.insert(0, [start])
				break

			# try to keep path straight
			if len(path) > 1:
				aim = path[0][0] - path[1][0]
				choices = [ elem + aim for aim in (aim, aim.T, -aim.T) ]
			else:
				choices = random_seq([ elem + aim for aim in VECTORS() ])

			for nb in choices:
				# if can move forward to the neighbor, add to nbs
				if self.is_inside(nb) and graph[nb.x, nb.y] == graph[elem.x, elem.y]-1:
					path.insert(0, [nb])
					break
			else: # unexpected case
				raise Warning('UNEXPECTED CASE')

		return path

	def graph_aim_deadend(self, graph, start, aim):
		pos = start + aim
		return True

	def graph_path_survive(self, graph, start, end):
		path = []
		return path


	def get_aim_graph(self):
		""" just follow current graph path

			return None if path not valid or head not in path
		"""
		try:
			# find current head in path
			next_id = self.graph_path.index(self.head) + 1
		except:
			return None

		return self.graph_path[next_id] - self.head

	def get_aim_greedy(self, md_diag=True):
		pd_cross = self.aim.pd_cross(self.vec_head2food)
		pd_dot = self.aim.pd_dot(self.vec_head2food)

		if md_diag:
			""" 斜线 """
			if pd_cross > 0:
				matrix = TransMatrix.ROTATE_LEFT
			elif pd_cross < 0:
				matrix = TransMatrix.ROTATE_RIGHT
			else:				# pd_cross == 0: 同向，反向，到达
				if pd_dot >= 0:		# 同向或到达，保持
					return self.aim
				else:				# 反向，随机转向
					matrix = random.choice([TransMatrix.ROTATE_LEFT, TransMatrix.ROTATE_RIGHT])
		else:
			""" 少转弯 """
			if pd_dot > 0:		# 前方，保持
				return self.aim
			else:				# 垂直或后方
				if pd_cross > 0:	# 左后
					matrix = TransMatrix.ROTATE_LEFT
				elif pd_cross < 0:	# 右后
					matrix = TransMatrix.ROTATE_RIGHT
				else:				# 反向，随机转向
					matrix = random.choice([TransMatrix.ROTATE_LEFT, TransMatrix.ROTATE_RIGHT])

		return self.aim.trans_linear(matrix, inplace=False)

	def get_auto_aim(self, mode, md_sub=True):
		"""
		return None if no valid aim, which means died and just keep aim

		take care of the init operation for conrresponding auto mode
		"""

		if mode == AutoMode.GRAPH:
			aim_next = self.get_aim_graph()

		elif mode == AutoMode.GREEDY:
			aim_next = self.get_aim_greedy(md_sub)

		else:				# random
			aim_next = None

		aim_choices = [ self.aim, self.aim.T, -self.aim.T ]
		aim_choices = [ fb for fb in random_seq(aim_choices) ]

		if aim_next:
			aim_choices.remove(aim_next)
			aim_choices.insert(0, aim_next)

		# in case, fallback
		for aim in aim_choices:
			if self.is_aim_right(aim):
				return aim

		return None

# vi: set ts=4 noexpandtab foldmethod=indent :
