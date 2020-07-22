#!/usr/bin/env python
# coding: utf8

import random
import numbers
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
		return self.is_inside(head_next) and head_next not in self.body[1:]

	def snake_reset(self):
		self.body = [ Vector(int(self.area_w/2), int(self.area_h/2)) ]
		self.aim = VECTORS.DOWN

		self.food = self.new_food()
		self.graph = None
		self.path = None

		# re-seed on reset
		random.seed()

	def snake_load(self, data):
		bacup_area = (self.area_w, self.area_h)

		try:
			area_w, area_h = data['area']
			body = [ Vector(*pos) for pos in data['body'] ]
			aim = Vector(*data['aim'])
			food = Vector(*data['food'])

			# verify
			# check integral
			for pos in body + [ aim, food, Vector(area_w, area_h) ]:
				for num in pos:
					if not isinstance(num, numbers.Integral):
						raise Exception()

			# check area limit, aim, food position
			if area_w < 10 or area_h < 10 or \
					aim not in VECTORS() or \
					food in body:
				raise Exception()

			# check reverse aim
			if len(body) > 1 and aim + body[0] == body[1]:
				raise Exception()

			# assign area limit for is_inside()
			self.area_w, self.area_h = (area_w, area_h)

			# check inside
			for pos in body + [food]:
				if not self.is_inside(pos):
					raise Exception()
		except:
			self.area_w, self.area_h = bacup_area
			return False
		else:
			self.snake_reset()

			self.body = body
			self.aim = aim
			self.food = food

			return True

	def snake_save(self):
		# return body with list() for direct dump
		return {
				'area': [ self.area_w, self.area_h ],
				'body': [ (pos.x, pos.y) for pos in self.body ],
				'aim': ( self.aim.x, self.aim.y ),
				'food': ( self.food.x, self.food.y )
				}

	def area_resize(self, width, height, do_reset=False):
		if do_reset:
			self.area_w = width
			self.area_h = height

			# reset for body rely on new area
			self.snake_reset()
		else:
			if width < self.area_w or height < self.area_h:
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
			self.path = None
		else:
			self.body.pop()

		# return after pop even died
		if self.is_died():
			return False

		return True

	def new_food(self):
		""" 随机生成食物

			策略：
				1. 生成地图
				2. 标记 body 区域
				3. 从未标记区域随机选择
		"""

		if self.is_full():
			return None

		# init to True for later nonzero()
		snake_map = np.ones((self.area_w, self.area_h), dtype='bool')

		for elem in self.body:
			snake_map[elem.x, elem.y] = False

		new_x,new_y = random.choice(np.transpose(snake_map.nonzero()))
		return Vector(new_x, new_y)

	def rect_border_body(self, with_food=True):
		"""获取包围 body 最小矩形，并外扩一周，作为 bfs 边界限制"""
		rect = [ x_min, y_min, x_max, y_max ]

	@count_func_time
	def scan_path_and_graph(self, md_fast=True):
		"""best first search"""

		# in np, it's (row, col), it's saved/read in transposed style
		graph = np.zeros((self.area_w, self.area_h), dtype=np.int32)
		graph_parent = np.zeros((self.area_w, self.area_h), dtype=Vector)

		dist_adj = 2				# 调整间隙
		h_weight = 1				# 预估成本权重

		# 预估距离
		dist = lambda a,b: abs(a.x-b.x) + abs(a.y-b.y)
		# 成本评估
		cost = lambda elem: graph[elem.x, elem.y] + h_weight * dist(elem, self.food)

		# 如通过优先队列，则无需两层循环
		pipe = [self.head]

		# scan_graph
		while len(pipe) > 0:
			cost_ref = cost(pipe[0])
			pipe_next = []

			# make sure expand at least one elem in a cycle, or infinite loop
			for elem in pipe:
				if md_fast and elem == self.food:
					break

				# distance from head to the elem
				dist_elem = graph[elem.x, elem.y]

				if cost(elem) - cost_ref < dist_adj:
					# neighbors of head
					neighbors = [ elem + aim for aim in VECTORS() ]

					if dist_elem == 0:
						# the head, can not go backward directly
						neighbors.remove(self.head - self.aim)
						body_check = self.body[1:]
					else:
						# the tail have moved forward
						body_check = self.body[:-dist_elem]

					for nb in neighbors:
						if self.is_inside(nb) and nb not in body_check:
							if graph[nb.x, nb.y] == 0:
								graph[nb.x, nb.y] = dist_elem + 1
								graph_parent[nb.x, nb.y] = elem
								pipe_next.append(nb)
				else:
					pipe_next.append(elem)

			pipe_next.sort(key=cost)
			pipe = pipe_next

		# graph_to_path
		if graph[self.food.x, self.food.y]:
			path = [self.food]
			while path[0] != self.head:
				parent = graph_parent[path[0].x, path[0].y]
				path.insert(0, parent)
		else:
			path = []

		self.graph = graph
		self.path = path

	def body_after_eat(self):
		"""called right after graph scan and food is reachable"""

		virt_length = self.length + 1
		path_length = len(self.path)

		# construct new body after eat food
		virt_body = []

		# both head and food is in the path
		for i in range(min(virt_length, path_length)):
			virt_body.append(self.path[-i-1])

		# if virt_length <= path_length, will not enter loop
		for i in range(virt_length - path_length):
			virt_body.append(self.body[i+1])

		return virt_body

	@count_func_time
	def can_cycle_of_life(self, body):
		""" check weather head can reach tailer for body

			BFS 终止条件，搜索首度进入原本的 body
		"""

		# if the dtype is uint, -graph[*] is still unsigned ..
		graph = np.zeros((self.area_w, self.area_h), dtype=np.int32)

		pipe = [ body[0] ]

		while len(pipe) > 0:
			elem = pipe.pop(0)

			dist_elem = graph[elem.x, elem.y]

			if dist_elem == 0:
				body_check = body
				body_safe = []
			else:
				body_check = body[:-dist_elem]
				body_safe = body[-dist_elem:]

			for nb in [ elem + vect for vect in VECTORS() ]:
				if self.is_inside(nb) and graph[nb.x, nb.y] == 0:
					if nb in body_safe:
						return True

					if nb not in body_check:
						graph[nb.x, nb.y] = dist_elem + 1
						pipe.append(nb)

		self.graph = graph
		return False

	def path_set_wander(self):
		""" 闲逛

			草履虫模式：遇到障碍反向：反向意味接连转向

			基本策略：
			1. 折叠前进
				总体前进方向：与初始方向垂直
			2. 单侧预留逃生通道，另一侧可填满
				逃生通道侧：初始方向的反向
			3. 闲逛路径长度：1/4 body 长度

			实现：
			1. 保持方向
			2. 保持方向的终点判断
			3. 转向后再次转向
		"""
		# todo

		is_safe = lambda x: self.is_inside(x) and x not in self.body

		path = [self.head]

		aim_cur = self.aim
		dir_v = self.aim.T
		if not is_safe(self.head + dir_v):
			dir_v = -dir_v

		for i in range(self.length//2):
			for aim in (aim_cur, dir_v):
				move_to = path[-1] + aim
				if is_safe(move_to):
					path.append(move_to)
					break
			else:	# no valid path
				break

			if aim == dir_v:
				aim_cur = -aim_cur

		if len(path) > 1:
			self.path = path


	def get_aim_path(self):
		""" just follow current graph path

			the path has at least two element

			keep direction on exit path
			return None if path not valid or head not in path
		"""
		try:
			# find current head in path
			next_id = self.path.index(self.head) + 1
		except:
			return None

		if next_id < len(self.path):
			return self.path[next_id] - self.head
		else:
			# reach the end of path (for path from wander)
			# keep aim for the step to exit path
			return self.path[-1] - self.path[-2]

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
			aim_next = self.get_aim_path()

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
			if self.is_aim_valid(aim):
				return aim

		return None

# vi: set ts=4 noexpandtab foldmethod=indent :
