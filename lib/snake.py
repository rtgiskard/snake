#!/usr/bin/env python
# coding: utf8

import random
import heapq
import numbers
import numpy as np

from .datatypes import *
from .decorator import *
from .dataref import *
from .functions import *

"""
Snake:
	1. after eat last food, no food is able to be generated,
		the snake will keep run forever(circle of life: COL)
	2. the snake is allowed to follow it's tail closely:
		tail followed by head.
		this is decided by is_aim_valid() and move()
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
		return self.is_inside(head_next) and head_next not in self.body[1:-1]

	def snake_reset(self):
		self.body = [ Vector(int(self.area_w/2), int(self.area_h/2)) ]
		self.aim = VECTORS.DOWN

		self.food = self.new_food()
		self.graph = None			# the dist map
		self.path = None			# path to follow, general lead to food
		self.path_col = []			# path for COL, food safety check and COL cache

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
		"""
		parameter:
			aim: the direction to move on,
				need pre-check for valid move
		return: bool
			True: if alive after move
			False: if not

		note:
			1. move after died may revive the snake
		"""

		if aim:	# set new direction
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
		return (not self.is_died())

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

		# ATTENTION: thw new x,y is np.int64, convert to python int
		# or json dump may failed with it
		new_x,new_y = random.choice(np.transpose(snake_map.nonzero()))
		return Vector(int(new_x), int(new_y))


	def body_after_eat(self):
		"""called right after scan and food is reachable"""

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

	def body_rect_with_border(self, body, food=None):
		"""获取包围 body 最小矩形，并外扩一周，作为 bfs 边界限制

			外扩一周后应足以囊括有效搜索区域
		"""

		# 以 head 初始 rect
		rect = Rect(*body[0], *body[0])

		# 最小矩形
		for elem in body[1:]:
			rect.extend(elem)

		# 边界内外扩一周
		if rect.x0 > 0:
			rect.x0 -= 1
		if rect.y0 > 0:
			rect.y0 -= 1
		if rect.x1 < self.area_w-1:
			rect.x1 += 1
		if rect.y1 < self.area_h-1:
			rect.y1 += 1

		# food
		if food:
			rect.extend(food)

		return rect


	def BFS(self, graph, start=None, end=None, restrict=None):
		"""A* best first search

		parameters:
			graph: a list of info to construct the graph
				[ area_w, area_h, body ]
			start: the search start point and related info
				None: default to body[0]
				for dynamic body on move, it should be body[0]
			end: the point where search can stop
				None for circle of life
			restrict: list of restrictions for the search
				[ rect, aim ]

				rect: 优化的搜索边界
				aim: 初始方向

		return:
			(path, graph)

			path: the path to follow, a list even empty
			graph: a map for distance of node, an array even all zero
		"""
		# parse parameters
		array_dim = graph[:2]
		body = graph[-1]
		rect, aim_0 = restrict

		# start should be body[0]
		if start is None:
			start = body[0]

		# in np, it's (row, col), it's saved/read in transposed style
		map_dist = np.zeros(array_dim, dtype=np.int32)
		map_parent = np.zeros(array_dim, dtype=Vector)

		# set cost() and end_set() according to end
		if end is None:
			# 相同优先级: pqueue 退化为 queue, A* 退化成 Dijkstra's
			cost = lambda elem: 1
			end_set = lambda x: body[-x:] if x>0 else []
		else:
			# 成本评估：已确认 + 预估
			cost = lambda elem: map_dist[elem.x, elem.y] \
					+ ( abs(end.x-elem.x) + abs(end.y-elem.y) )
			end_set = lambda x: [ end ]

		# 优先队列
		pqueue = []

		# push_ct 避免比较 vector，同时保证：相同优先级 FIFO
		push_ct = 0
		heapq.heappush(pqueue, (0, push_ct, start))
		# set the start parent to start - aim_0
		map_parent[start.x, start.y] = start - aim_0

		# generate map_dist
		while len(pqueue) > 0:
			_cost, _ct, elem = heapq.heappop(pqueue)

			# distance from head to the elem
			dist_elem = map_dist[elem.x, elem.y]

			# check end loop
			if elem in end_set(dist_elem):
				if end is None:		# for COL BFS, assgin to end
					end = elem
				break

			# neighbors of head
			neighbors = [ elem + aim for aim in VECTORS() ]
			# can not go backward
			neighbors.remove(map_parent[elem.x, elem.y])

			# the tail have moved forward
			void_set = body[:-dist_elem-1]

			for nb in neighbors:
				if rect.is_inside(nb) and map_dist[nb.x, nb.y] == 0:
					if nb not in void_set:
						map_dist[nb.x, nb.y] = dist_elem + 1
						map_parent[nb.x, nb.y] = elem

						push_ct += 1
						heapq.heappush(pqueue, (cost(nb), push_ct, nb))

		# generate path, check end before map_dist
		if end and map_dist[end.x, end.y]:
			path = [end]
			# the `do .. while()` is neccessary for the case head == end,
			# eg. snake with length equals 1 in COL scan
			while True:
				path.insert(0, map_parent[path[0].x, path[0].y])
				if path[0] == start:
					break
		else:
			path = []

		return (path, map_dist)

	@count_func_time
	def scan_path_and_graph(self):
		"""best first search"""

		# bfs 搜索边界
		rect = self.body_rect_with_border(self.body, self.food)

		p_graph = [ self.area_w, self.area_h, self.body ]
		p_start = None
		p_end = self.food
		p_restrict = [ rect, self.aim ]

		return self.BFS(p_graph, p_start, p_end, p_restrict)

	@count_func_time
	def scan_cycle_of_life(self, body=None):
		""" scan path from head to tail for given body

			col path:
				1, start == body[0], end in body
				2. elem between not in body
					as it's the shortest path to form the cycle
				3. self.path_col is only set/reset after eat food safely or follow COL

			parameter: None for current body
			return: (path, graph)
		"""
		if body is None:
			body = self.body

		# bfs 搜索边界
		rect = self.body_rect_with_border(body)

		p_graph = [ self.area_w, self.area_h, body ]
		p_start = None
		p_end = None
		p_restrict = [ rect, self.aim ]

		return self.BFS(p_graph, p_start, p_end, p_restrict)

	def path_set_wander(self, length=None):
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

			todo：如前进时无法回到同侧，则放弃该方向
				如 dir_v 为右方：
					检查：右方，右前方，前方
					如均为空，则可前进，否则转向
				当方向为 -aim 时，检查 右前 ..

			预留通道：aim 方向预留 1~2 个空格
		"""
		# 默认闲逛长度 length/8， 最短为 4
		if length is None:
			length = self.length // 8
		length = min(length, 4)

		is_safe = lambda x: self.is_inside(x) and x not in self.body

		path = [self.head]

		aim_cur = self.aim
		dir_v = self.aim.T
		if not is_safe(path[-1] + dir_v):
			dir_v = -dir_v

		for i in range(length):
			for aim in (aim_cur, dir_v, -dir_v):
				move_to = path[-1] + aim
				if is_safe(move_to):
					path.append(move_to)
					break
			else:	# no valid path
				break

			if aim == dir_v:
				aim_cur = -aim_cur
			elif aim == -dir_v:
				dir_v = -dir_v
				aim_cur = -aim_cur

		if len(path) > 1:
			self.path = path

	@count_func_time
	def path_set_col(self, body=None):
		"""set path to col path, fallback to wander

			1. check existing col path for current state

			2. if valid: set to it
				else: recalc col path

			3. if col path present: set to it
				else: wander
		"""
		if body is None:
			body = self.body

		mark_rescan = True

		# check existing col path, rely on col path property
		if len(self.path_col) > 0:
			if self.path_col[0] == body[0] \
					and self.path_col[-1] in body:
				path_to_check = self.path_col[1:-1]

				if len(path_to_check) == 0:
					""" the extreme COL

						food is reachable, but not surely safe,
						every step forward may require a scan of path and path_col

						extrem COL is only possible with len(body) >= 4

						to avoid scan every step, extend the path forward,
						limit forward step within len(body)-2 to avoid dead
						loop, and set to ~length/8 with minimum to (4-2)
					"""

					forward_step = (len(body) - 2)//8
					forward_step = max(2, forward_step)

					for i in range(forward_step):
						self.path_col.append(body[-2-i])

					mark_rescan = False
				else:
					""" COL with blank between head and tail

						when the col path is end, may or may not be extrem COL
					"""
					# generate body mask map to speed up path_col check
					snake_map = np.zeros((self.area_w, self.area_h), dtype='bool')
					for elem in self.body:
						snake_map[elem.x, elem.y] = True

					for elem in path_to_check:
						if snake_map[elem.x, elem.y]:
							# if any block run into body, rescan
							break
					else:
						mark_rescan = False

		if mark_rescan:
			self.path_col, _graph = self.scan_cycle_of_life()

		# final check for col
		if len(self.path_col) > 0:
			self.path = self.path_col
			# reset COL path after apply COL
			self.path_col = []
		else:
			self.path_set_wander()

	#@count_func_time
	def update_paths_and_graph(self):
		""" op pack for all path rescan and update

			return: bool
				True: go and eat food
				False: follow col or wander

			pseudo code:
				scan_for_path_to_food

				if path exist:
					if food safe:
						go_eat_food
						return

				follow col path, with fallback wander
		"""

		# scan and save the path to food
		self.path, self.graph = self.scan_path_and_graph()

		# save path for COL if exist
		if len(self.path) > 0:
			new_body = self.body_after_eat()
			path, graph = self.scan_cycle_of_life(new_body)
			if len(path) > 0:	# food safe, save col path, go eat food
				self.path_col = path
				return True
			else:				# food not safe, try to follow COL
				# set graph to the state after eat
				self.graph = graph

		self.path_set_col()
		return False


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
