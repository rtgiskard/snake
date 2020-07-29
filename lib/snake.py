#!/usr/bin/env python
# coding: utf8

import random
import heapq
import numbers
import numpy as np

from .datatypes import *
from .decorator import *
from .dataref import *

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
	def head(self):
		return self.body[0]

	def is_full(self):
		return self.length == self.area_size

	def is_died(self):
		return self.head in self.body[1:] or not self.is_inside(self.head)

	def is_inside(self, point):
		return ( point.x >= 0 and point.x < self.area_w
				and point.y >= 0 and point.y < self.area_h )

	def is_aim_valid(self, aim):
		""" within single step forward, the snake is impossible to run into:
			body[0:3]: as the first 3 block is impossible to be around head
			body[-1]: the snake is moving forward
		"""
		head_next = self.head + aim
		return self.is_inside(head_next) and head_next not in self.body[3:-1]

	def is_move_safe(self, pos=None, body=None, step=0):
		""" whether a move is safe

			only check the original body area, the move stratage should make
			sure that the snake will not run into the path again

			pos: the position to move to
			body: what is moving
			step: step that the body has moved before
		"""
		if body is None: body = self.body
		if pos is None: pos = body[0] + self.aim

		return self.is_inside(pos) and pos not in body[:-(i+1)]


	def snake_reseed(self):
		random.seed()

	def snake_reset(self):
		self.body = [ Vector(int(self.area_w/2), int(self.area_h/2)) ]
		self.aim = VECTORS.DOWN

		self.food = self.new_food()
		self.graph = None			# the dist map
		self.path = None			# path to follow, general lead to food
		self.path_col = []			# path for COL, food safety check and COL cache
		self.path_unsafe = None		# cached unsafe path to food

		# re-seed on reset
		random.seed()

	def snake_load(self, data):
		bacup_area = (self.area_w, self.area_h)
		backup_rnd = random.getstate()

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

			# convert and recover random state
			if 'rnd_state' in data.keys():
				data['rnd_state'][1] = tuple(data['rnd_state'][1])
				rnd_state = tuple(data['rnd_state'])
				random.setstate(rnd_state)
		except:
			self.area_w, self.area_h = bacup_area
			random.setstate(backup_rnd)

			return False
		else:
			self.snake_reset()

			self.body = body
			self.aim = aim
			self.food = food

			return True

	def snake_dump(self):
		# return body with list() for direct dump
		return {
				'area': [ self.area_w, self.area_h ],
				'body': [ (pos.x, pos.y) for pos in self.body ],
				'aim': ( self.aim.x, self.aim.y ),
				'food': ( self.food.x, self.food.y ),
				'rnd_state': random.getstate()
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


	def body_after_step_on_path(self, body, path, step):
		""" called right after scan and food is reachable

			for better perspective, draw the path and body on paper
		"""

		# body and path increase in different direction
		assert body[0] == path[0]
		assert len(path) > 1
		assert step < len(path)

		# copy from path to new body
		new_body = path[:step+1]

		# the blocks left only on old body
		body_step = len(body) - step - 1

		# check whether append or pop to body
		if body_step >= 0:
			for i in range(body_step):
				new_body.insert(0, body[i+1] )
		else:
			for i in range(-body_step):
				new_body.pop(0)

		# reverse inplace to get the real body
		new_body.reverse()

		return new_body

	def body_rect_with_border(self, body, food=None):
		""" 获取包围 body 最小矩形，并外扩一周，作为 bfs 边界限制

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


	def BFS(self, graph, start=None, end=None):
		""" A* best first search

		parameters:
			graph: a list of info to construct the graph
				[ body, (area_w, area_h), (aim_0, rect) ]
					(area_w, area_h): area size
					(aim_0, rect): (initial aim, search only in rect)
			start: the search start point and related info
				None: default to body[0]
				for dynamic body on move, it should be body[0]
			end: the point where search can stop
				None for circle of life

		return:
			(path, graph)

			path: the path to follow, a list even empty
			graph: a map for distance of node, an array even all zero
		"""
		# parse parameters
		body = graph[0]
		array_dim = graph[1]
		aim_0, rect = graph[2]

		# currently start must be body[0]
		assert start is None

		if start is None:
			start = body[0]

		# in np, it's (row, col), it's saved/read in transposed style
		map_dist = np.zeros(array_dim, dtype=np.int32)	# dist from start to current
		map_aim = np.zeros(array_dim, dtype=Vector)		# parent + aim = current

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

		# pqueue: (cost, push_ct, elem)
		# push_ct 避免比较 vector，同时保证：相同优先级 FIFO
		push_ct = 0
		heapq.heappush(pqueue, (0, push_ct, start))
		map_aim[start.x, start.y] = aim_0

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

			"""
			to get a straight path, the neighbors' sequence should be fixed.
			"""
			aim = map_aim[elem.x, elem.y]
			# neighbors of head, todo: random turn?
			neighbors = [ elem + aim for aim in (aim, aim.T, -aim.T) ]

			# the tail have moved forward
			void_set = body[:-dist_elem-1]

			for nb in neighbors:
				if rect.is_inside(nb) and map_dist[nb.x, nb.y] == 0:
					if nb not in void_set:
						map_dist[nb.x, nb.y] = dist_elem + 1
						map_aim[nb.x, nb.y] = nb - elem

						push_ct += 1
						heapq.heappush(pqueue, (cost(nb), push_ct, nb))

		# generate path, check end before map_dist
		if end and map_dist[end.x, end.y]:
			path = [end]
			# the `do .. while()` is neccessary for the case head == end,
			# eg. snake with length equals 1 in COL scan
			while True:
				path.insert(0, path[0] - map_aim[path[0].x, path[0].y])
				if path[0] == start:
					break
		else:
			path = []

		return (path, map_dist)

	def scan_wrapper(self, body=None, aim=None, target=None):
		""" a scan wrapper for BFS search

			parameter: None for current body
			return: (path, graph)
		"""

		if body is None:
			body = self.body

		if aim is None:
			if len(body) > 1:
				aim = body[0] - body[1]
			else:
				aim = self.aim

		# bfs 搜索边界
		rect = self.body_rect_with_border(body, target)

		p_graph = [ body, (self.area_w, self.area_h), (aim, rect) ]
		p_start = None
		p_end = target

		return self.BFS(p_graph, p_start, p_end)

	@count_func_time
	def scan_path_and_graph(self, body=None, aim=None):
		""" scan path for head to target for given body """
		return self.scan_wrapper(body, aim, self.food)

	@count_func_time
	def scan_cycle_of_life(self, body=None, aim=None):
		""" scan path for head to tail for given body

			col path:
				1, path_col[0] == body[0], end in body
				2. any elem in between is not in body
					as it's the shortest path to form the cycle

			for COL with head following tail:
				food is reachable, but not safe to eat,
				every step forward may require a scan of path and path_col, while
				path_col scan can be cheap, path scan may be expensive, and draw
				update may not happen if too slow

				extreme COL is only possible with len(body) >= 4

				it's possible that with the COL step, the snake run into infinite loop
				in the path: if only several breaks have safe path and all the step
				breaks do not hit

			for COL with blank between head and tail:
				case 1: after the path, get col path with length 2
				case 2: after the path, get col path still longer than 2

				with any case, the snake will be able to reach food on next scan,
					whether safe or not safe to eat

				if the next path to food is not safe,
					for case 1, it becomes extreme COL
					for case 2, scan happens at the end of path, but the snake may
						end in infinite loop in the path if every break is not safe

			to avoid unneccessary scan and dead loop:
				cache unsafe path, adjust col path after scan
		"""
		return self.scan_wrapper(body, aim, None)

	def scan_path_wander(self, body=None, step=None):
		"""
			aim, trd, -aim
			v     . .   . .
			. .   > .   ^
			. .     .
		"""

		if body is None: body = self.body
		if step is None: step = self.length // 8
		step = max(step, 4)

		aim = self.aim

		# select trend, also make sure (body[0] + trend) is safe
		for trend in (aim.T, -aim.T):
			if self.is_move_safe(body[0] + trend, body, step=0):
				trd = trend
				break
		else:		# no safe trend, no wander path
			return []

		# check set
		chk_0 = lambda pos: [				# aim, if failed check, turn
					[pos + aim, 1],				# on aim
					[pos + aim + trd, 2],		# next turn possibility
					[pos + 2*aim, 2],			# exit path
					[pos + 2*aim + trd, 3] ]	# exit path
		chk_1 = lambda pos: [				# trd, if failed, exit path
					[pos - aim, 1],				# on turn
					[pos + trd, 1],				# exit path
					[pos - aim + trd, 2],		# exit path
					[pos + aim + trd, 2] ]		# exit path
		chk_2 = lambda pos: [				# -aim, if failed, turn + aim
					[pos - aim, 1],				# on -aim
					[pos - aim + trd, 2] ]		# next turn possibility

		# a cycle of wrap: aim, trd, (-aim, trd)
		chk_list = ( chk_0, chk_1, chk_2 )

		checked = set()
		path = [ body[0] ]

		step_ct = 0
		state = 0

		while step_ct < step:
			chk = chk_list[state](path[-1])

			if wander_move_check(body, step_ct, chk, checked):
				path.append(chk[0][0])
			else:
				if state & 1:
					# state: 1, 3: turn failed, exit
					path.append(path[-1] + aim)
					break
				else:
					# state: 0, 2: aim/-aim failed, turn
					path.append(path[-1] + trd)

				state += 1
				state %= len(chk_list)

			step_ct += 1

		if len(path) > 1:
			return path
		else:
			return []

	def wander_move_check(self, body, step, uncheck, checked):
		""" check four block

			return: bool
				True: if all clear
				False: move on aim will break the wander
		"""
		for pos in uncheck:
			if pos[0] not in checked:
				if self.is_move_safe(pos[0], body, step + pos[1]):
					checked.add(pos[0])
				else:
					return False

		return True

	def validate_col_path_on_body(self, path, body):
		""" validate the col path on given body """

		if path[0] == body[0] and path[-1] in body:
			# middle of path not in body
			# path should be shorter, set(path) is memory efficient
			intersec = set(path[1:-1]).intersection(body)
			return (len(intersec) == 0)
		else:
			return False

	@count_func_time
	def path_set_col(self, body=None):
		""" set path to col path, fallback to wander

			1. check existing col path for current state

			2. if valid: set to it
				else: recalc col path

			3. if col path present: set to it
				else: wander
		"""
		if body is None:
			body = self.body

		# if no path_col or path_col invalid
		if len(self.path_col) == 0 or \
				not self.validate_col_path_on_body(self.path_col, self.body):
			self.path_col, _graph = self.scan_cycle_of_life()

		# final check for col, then set path
		if len(self.path_col) > 0:		# apply and reset path_col
			self.path = self.path_col
			self.path_col = []
		else:							# fallback to wander path
			self.path_set_wander()

	@count_func_time
	def path_set_wander(self, step=None):
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
				如 trd 为右方：
					检查：右方，右前方，前方
					如均为空，则可前进，否则转向
				当方向为 -aim 时，检查 右前 ..

			预留通道：aim 方向预留 1~2 个空格
		"""
		# 默认闲逛长度： length/8， 最短为 4
		if step is None:
			step = self.length // 8
		step = max(step, 4)

		is_safe = lambda x, i: self.is_inside(x) and x not in self.body[:-(i+1)]

		path = [self.head]

		aim_cur = self.aim
		trd = self.aim.T

		# 简单预判前进方向
		if not is_safe(path[-1] + trd, 0):
			trd = -trd

		for i in range(step):
			for aim in (aim_cur, trd, -trd):
				move_to = path[-1] + aim
				if is_safe(move_to):
					path.append(move_to)
					break
			else:	# no valid path
				break

			if aim == trd:
				aim_cur = -aim_cur
			elif aim == -trd:
				trd = -trd
				aim_cur = -aim_cur

		if len(path) > 1:
			self.path = path

	def adjust_path_on_unsafe(self, path_unsafe):
		"""
		path_unsafe: one of the shortest paths to food, any other path start
			from the unsafe path should be longer or at least equal

		compare path and unsafe path, find:
			idx_0: where path start in unsafe path
			idx_1: where path leave unsafe path
			idx_2: where unsafe path leave body for the last time

			if path go apart from unsafe path before unsafe path leave body
				(not likely to happen)
				trust path and follow path
			if path go apart from unsafe path after unsafe path leave body
				cut path to queue rescan

			if path end in unsafe path before unsafe path leave body
				follow unsafe path till it leaves body fot the last time
			if unsafe path end before path go apart
				(not likely to happen)
				follow path
		"""
		idx_0 = 0		# where the match begins (first match)
		idx_1 = 0		# where the match ends (first unmatch after match)
		idx_2 = 0		# where the unsafe path leaves body

		cmp_md = 0		# compare mode

		for i in range(len(path_unsafe)):
			if cmp_md == 0:			# find where match begins
				if path_unsafe[i] == self.path[0]:
					idx_0 = i
					cmp_md = 1
			elif cmp_md == 1:		# real check after match
				if i-idx_0 >= len(self.path):
					# path end in unsafe path
					cmp_md = 2
				elif path_unsafe[i] != self.path[i-idx_0]:
					# path go apart from path_unsafe
					idx_1 = i
					break
			elif cmp_md == 2:		# check for where unsafe path last leaves body
				if path_unsafe[i] in self.body:
					idx_2 = i

		if idx_1 > 0:
			# cut path at where it's apart
			new_path = self.path[:idx_1 - idx_0 + 1]
		elif idx_2 > 0:
			# extend path to where unsafe path leaves body(the last one), stop
			# in body (as unsafe path is unsafe, follow it to leave body is
			# dangerous, but not before it leaves body for the last time)
			new_path = self.path_unsafe[idx_0:idx_2 + 1]
		else:
			new_path = self.path

		if len(new_path) >= 2:
			self.path = new_path


	#@count_func_time
	def update_path_and_graph(self):
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

		if self.path_unsafe is None or self.head not in self.path_unsafe:
			# scan and save the path to food, reset unsafe path
			self.path, self.graph = self.scan_path_and_graph()
			self.path_unsafe = None
		else:
			# if still in unsafe path, follow col
			self.path = []

		# check col after path
		if len(self.path) > 0:
			# construct body after eat food
			new_body = self.body_after_step_on_path( self.body, self.path,
					self.graph[self.food.x, self.food.y]-1)
			new_body.insert(0, self.food)

			path, graph = self.scan_cycle_of_life(new_body)

			if len(path) > 0:	# food safe, save col path, go eat food
				self.path_col = path
				return True
			else:				# food not safe, try to follow COL
				# set graph to the state after eat
				self.graph = graph
				self.path_unsafe = self.path

		# set path to col, or fallback to wander
		self.path_set_col()

		if self.path_unsafe:
			self.adjust_path_on_unsafe(self.path_unsafe)

		return False

	def get_aim_path(self):
		""" just follow current path

			path has at least two element

			return None:
				. path not valid
				. head not in path
				. at end of path
		"""
		try:
			# find current head in path
			next_id = self.path.index(self.head) + 1
			assert next_id < len(self.path)
		except:
			return None

		return self.path[next_id] - self.head

	def get_aim_greedy(self, md_diag=True):
		vec_head2food = self.food - self.head
		pd_cross = self.aim.pd_cross(vec_head2food)
		pd_dot = self.aim.pd_dot(vec_head2food)

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
		return None if no valid aim, which means died

		the fallback aim:
			1. try to keep direction
			2. if need to turn, random turn
		"""

		# for fallback, always try aim first, the default
		aim_T = random.choice((self.aim.T, -self.aim.T))
		aim_choices = [ self.aim, aim_T, -aim_T ]

		if mode == AutoMode.GRAPH:
			aim_next = self.get_aim_path()
		elif mode == AutoMode.GREEDY:
			aim_next = self.get_aim_greedy(md_sub)
		else:	# RANDOM
			aim_next = random.choice(aim_choices)

		if aim_next:
			aim_choices.remove(aim_next)
			aim_choices.insert(0, aim_next)

		# in case, fallback
		for aim in aim_choices:
			if self.is_aim_valid(aim):
				return aim

		return None

# vi: set ts=4 noexpandtab foldmethod=indent :
