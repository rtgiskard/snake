#!/usr/bin/env python
# coding: utf8

import sys
import time, random, math
from functools import wraps
import numpy as np

import gi
gi.require_version('Gtk', '3.0')

from gi.repository import Gio, Gtk, Gdk, GLib
from gi.repository.GdkPixbuf import Pixbuf, PixbufRotation, InterpType
import cairo

"""
todo:
. 封闭区间最大长度(重扫描时间点长度二分)
. 如到达食物后，围成空间不足长度，则游荡会再查看
. 路径避免不必要转弯？
"""

class DEBUG:
	pause = False


def echo_func(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		print('calling {} ..'.format(func.__name__))
		return func(*args, **kwargs)

	return wrapper

def echo_func_count(func):
	counter = [0]

	@wraps(func)
	def wrapper(*args, **kwargs):
		print('calling {} x{} ..'.format(func.__name__, counter[0]))
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
		print("T({}) us: {:.2f}  {:.2f}:{}".format(func.__name__,
			ts_this/1000, ts_avg[1]/1000, ts_avg[0]))

		return orig_return

	return wrapper


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

class TransMatrix:
	# matrix: [col1, col2]
	ROTATE_LEFT = [[0, 1], [-1, 0]]
	ROTATE_RIGHT = [[0, -1], [1, 0]]

class Dot:
	def __init__(self, x=0, y=0):
		self.x = x
		self.y = y

	def __repr__(self):
		return '({}, {})'.format(self.x, self.y)

	def __iter__(self):
		return iter((self.x, self.y))

class Vector(Dot):
	def __init__(self, x=0, y=0):
		super().__init__(x, y)

	def __neg__(self):
		return Vector(-self.x, -self.y)

	def __eq__(self, vec):
		return self.x == vec.x and self.y == vec.y

	def __add__(self, vec):
		return Vector(self.x + vec.x, self.y + vec.y)

	def __sub__(self, vec):
		return Vector(self.x - vec.x, self.y - vec.y)

	def __mul__(self, n):
		return Vector(self.x * n, self.y * n)

	def __rmul__(self, n):
		return Vector(self.x * n, self.y * n)

	def __truediv__(self, n):
		return Vector(self.x / n, self.y / n)

	def __iadd__(self, vec):
		self.x += vec.x
		self.y += vec.y
		return self

	def __isub__(self, vec):
		self.x -= vec.x
		self.y -= vec.y
		return self

	def __imul__(self, n):
		self.x *= n
		self.y *= n
		return self

	def __itruediv__(self, n):
		self.x /= n
		self.y /= n
		return self

	@property
	def T(self):
		return Vector(self.y, self.x)

	# the dot product: self \cdot vec
	def pd_dot(self, vec):
		return self.x * vec.x + self.y * vec.y

	# the cross product: self \times vec
	def pd_cross(self, vec):
		return self.x * vec.y - self.y * vec.x

	def trans_linear(self, matrix, inplace=True):
		# matrix: TransMatrix
		new_x = matrix[0][0] * self.x + matrix[1][0] * self.y
		new_y = matrix[0][1] * self.x + matrix[1][1] * self.y
		if inplace:
			self.x, self.y = (new_x, new_y)
			return self
		else:
			return Vector(new_x, new_y)

class Snake:
	def __init__(self, width=40, height=40):
		self.area_w = width
		self.area_h = height

		# food, body, aim
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
		self.aim = Vector(0,1)

		self.food = self.new_food()
		self.graph = None
		self.graph_path = None

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
		vect = Vector(0, 1)

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
			neighbors = [ elem + vect, elem - vect, elem + vect.T, elem - vect.T ]

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
		vect = Vector(0, 1)

		# the stack
		path = [[start]]

		while len(path) > 0:
			elem = path[-1][0]

			if elem == end:
				break

			nbs = []
			for nb in random_seq([ elem + vect, elem - vect, elem + vect.T, elem - vect.T ]):
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
		vect = Vector(0, 1)

		# the stack
		path = [[end]]

		while len(path) > 0:
			elem = path[0][0]

			# there should be only one path from start to end with the generated graph
			if graph[elem.x, elem.y] == graph[start.x, start.y] + 1:
				path.insert(0, [start])
				break

			for nb in random_seq([ elem + vect, elem - vect, elem + vect.T, elem - vect.T ]):
				# if can move forward to the neighbor, add to nbs
				if self.is_inside(nb) and graph[nb.x, nb.y] == graph[elem.x, elem.y]-1:
					path.insert(0, [nb])
					break
			else: # unexpected case
				print("WARNING: unexpected case")

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

	def get_next_aim(self, mode, md_sub=True):
		"""
		return None if no valid aim, which means died and just keep aim

		take care of the init operation for conrresponding auto mode
		"""

		if mode == 0:		# graph
			aim_next = self.get_aim_graph()

		elif mode == 1:		# greedy
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

class Handler:
	@classmethod
	def on_draw(cls, widget, cr, app):
		app.draw_init(cr)
		app.draw_snake(cr)

		# stop event pass on
		return True

	@classmethod
	def on_toggled(cls, widget, app):
		label_text = widget.get_label()
		widget_label = widget.get_child()
		widget_label.set_markup(app.toggle_text(label_text, widget.get_active()))

		active_state = widget.get_active()

		if widget is app.tg_auto:
			app.data['tg_auto'] = active_state
		elif widget is app.tg_run:
			app.data['tg_run'] = active_state

			if active_state:
				# disable combo once snake active
				app.area_combo.set_sensitive(False)

			if active_state:
				app.timeout_id = GLib.timeout_add(1000/app.data['speed'], app.timer_move, None)
			else:
				if app.timeout_id:
					GLib.source_remove(app.timeout_id)

	@classmethod
	def on_combo_changed(cls, widget, app):
		t_iter = widget.get_active_iter()
		if t_iter:
			model = widget.get_model()
			app.sync_block_area_from_text(model[t_iter][0])

			app.req_draw_size_mini()

	@classmethod
	def on_combo_entry_activate(cls, widget, app):
		entry_text = widget.get_text()

		# if text not changed
		if entry_text == app.get_block_area_text():
			return

		if app.sync_block_area_from_text(entry_text):
			# text valid and set

			model = app.area_combo.get_model()
			t_iter = model.get_iter_first()

			while t_iter:
				if model[t_iter][0] == entry_text:
					# if found, set active
					app.area_combo.set_active_iter(t_iter)
					break
				else:
					t_iter = model.iter_next(t_iter)
			else:
				# if not found, append data and set active
				t_iter = model.append([entry_text])
				app.area_combo.set_active_iter(t_iter)

			app.req_draw_size_mini()
		else:
			# invalid text, recovered from app.data
			widget.set_text(app.get_block_area_text())

	@classmethod
	def on_spin_value_changed(cls, widget, app):
		app.data['speed'] = widget.get_value_as_int()

	@classmethod
	def on_color_set(cls, widget, app):
		app.sync_color(widget)

	@classmethod
	def on_keyboard_event(cls, widget, event, app):
		KeyName = Gdk.keyval_name(event.keyval)
		keyname = KeyName.lower()

		KEY_PRESS = (event.type == Gdk.EventType.KEY_PRESS)
		KEY_RELEASE = (event.type == Gdk.EventType.KEY_RELEASE)

		# press 'ESC' to remove focus
		if KEY_PRESS and keyname == 'escape':
			app.window.set_focus(None)
			return True

		# if any widget focused, return False to pass on the event
		# switch focus by pass on the 'tab' event
		if app.window.get_focus() or (KEY_PRESS and keyname == 'tab'):
			return False

		# now handel all keyboard event here, without pass on

		if KEY_PRESS and keyname == 'h':
			app.panel.set_visible(not app.panel.get_visible())

		elif KEY_PRESS and keyname == 'g':
			if KeyName == 'G':
				app.update_snake_graph()
				app.debug['show_graph'] = True
			else:
				app.debug['show_graph'] = not app.debug['show_graph']

			app.draw.queue_draw()

		elif KEY_PRESS and keyname == 's':
			if KeyName == 'S':
				app.debug['sub_mode'] = True
			else:
				app.debug['sub_mode'] = False

		elif KEY_PRESS and keyname == 'm':
			app.debug['auto_mode'] += 1
			app.debug['auto_mode'] %= 3
			print('auto_mode: {}'.format(app.auto_modes[app.debug['auto_mode']]))

		if app.snake.is_died():
			if KEY_PRESS and keyname == 'r':
				app.reset_on_gameover()

			return True

		if keyname in [ 'up', 'down', 'left', 'right' ]:
			if KEY_PRESS:
				pixbuf = app.pix_arrow_key
				# 非反向即可转向
				if app.snake.aim != -app.map_arrow[keyname][1]:
					# save aim to buffer, apply only before the move
					app.snake_aim_buf = app.map_arrow[keyname][1]
			elif KEY_RELEASE:
				pixbuf = app.pix_arrow

			app.arrows[keyname].set_from_pixbuf(pixbuf.rotate_simple(app.map_arrow[keyname][0]))
		elif KEY_PRESS and keyname == 'p':
			state = app.tg_run.get_active()
			app.tg_run.set_active(not state)
		elif KEY_PRESS and keyname == 'a':
			state = app.tg_auto.get_active()
			app.tg_auto.set_active(not state)
		elif KEY_PRESS and keyname == 'bracketleft':
			app.bt_speed.spin(Gtk.SpinType.STEP_BACKWARD, 1)
		elif KEY_PRESS and keyname == 'bracketright':
			app.bt_speed.spin(Gtk.SpinType.STEP_FORWARD, 1)

		return True

class App(Gtk.Application):
	def __init__(self, *args, **kwargs):
		# allow multiple instance
		super().__init__(*args, application_id='rt.game.snake',
				flags=Gio.ApplicationFlags.NON_UNIQUE, **kwargs)

		self.window = None

		self.data = {
				'snake_width': 8,
				'block_size': 16,
				'block_area': {'width':40, 'height':40},
				'block_area_limit': {'min':10, 'max':999},
				'block_area_margin': 10,
				'block_area_scale': 1,
				'block_area_list': ( '{0}x{0}'.format(i*20) for i in range(1, 11) ),
				'bg_color': 'black',
				'fg_color': 'grey',
				'tg_auto': False,
				'tg_run': False,
				'speed': 8,
				'speed_adj': { 'value':1, 'lower':1, 'upper':99,
					'step_increment':1, 'page_increment':10, 'page_size':0 },
				'image_icon': './data/icon/snake.svg',
				'image_arrow': './data/pix/arrow.svg',
				'image_arrow_key': './data/pix/arrow-key.svg',
				'image_snake_food': './data/pix/bonus5.svg',
				}

		self.debug = {
				'auto_mode': 0,
				'sub_mode': True,
				'show_graph': False,
				}

		self.auto_modes = ['graph', 'greedy', 'random']

		# 注意绘图座标系正负与窗口上下左右的关系
		self.map_arrow = {
				'up': (PixbufRotation.NONE, Vector(0, -1)),
				'down': (PixbufRotation.UPSIDEDOWN, Vector(0, 1)),
				'left': (PixbufRotation.COUNTERCLOCKWISE, Vector(-1, 0)),
				'right': (PixbufRotation.CLOCKWISE, Vector(1, 0))
				}

		self.snake = Snake(self.data['block_area']['width'], self.data['block_area']['height'])
		self.snake_aim_buf = None
		self.timeout_id = None

	def reset_on_gameover(self):
		# reset snake and app
		self.snake.snake_reset()
		self.snake_aim_buf = None
		self.timeout_id = None

		# reset widgets
		self.tg_run.set_active(False)
		self.tg_auto.set_active(False)

		# re-activate widgets
		self.tg_run.set_sensitive(True)
		self.area_combo.set_sensitive(True)

		# reset length label
		self.lb_length.set_text('{}'.format(self.snake.length))

		# redraw
		self.draw.queue_draw()

	def get_block_area_text(self):
		area = self.data['block_area']
		return '{}x{}'.format(area['width'], area['height'])

	def sync_block_area_from_text(self, text):
		try:
			width, height = ( int(x) for x in text.split('x') )

			assert width >= self.data['block_area_limit']['min']
			assert width <= self.data['block_area_limit']['max']
			assert height >= self.data['block_area_limit']['min']
			assert height <= self.data['block_area_limit']['max']
		except:
			return None
		else:
			area = self.data['block_area']
			area['width'], area['height'] = ( width, height )

			return area

	def toggle_text(self, text, active):
		try:
			lstr, rstr = text.split('/')
		except:
			return text

		if active:
			return '<small>{}</small>/<big><b>{}</b></big>'.format(lstr, rstr)
		else:
			return '<big><b>{}</b></big>/<small>{}</small>'.format(lstr, rstr)

	def req_draw_size_mini(self):
		blk_sz = self.data['block_size']
		area_w = self.data['block_area']['width']
		area_h = self.data['block_area']['height']
		margin = self.data['block_area_margin']

		# get current monitor resolution
		display = Gdk.Display.get_default()
		monitor = Gdk.Display.get_monitor(display, 0)
		rect = Gdk.Monitor.get_geometry(monitor)

		area_lim = (int(rect.width * 0.9), int(rect.height * 0.9))
		area = [ blk_sz * area_w + 2 * margin, blk_sz * area_h + 2 * margin ]

		scale_x, scale_y = (1, 1)
		if area[0] > area_lim[0]:
			scale_x = area_lim[0]/area[0]

		if area[1] > area_lim[1]:
			scale_y = area_lim[1]/area[1]

		# use the smaller scale
		scale = scale_x if scale_x < scale_y else scale_y
		self.data['block_area_scale'] = scale

		# snake resize && reset
		self.snake.area_resize(area_w, area_h, True)
		self.snake.snake_reset()

		# request for mini size
		self.draw.set_size_request(area[0] * scale, area[1] * scale)

		# make sure redraw queued
		self.draw.queue_draw()

	def sync_color(self, *widgets):
		for widget in widgets:
			if widget is self.color_fg:
				self.data['fg_color'] = widget.get_rgba().to_string()
			elif widget is self.color_bg:
				self.data['bg_color'] = widget.get_rgba().to_string()

	def set_color(self, *widgets):
		rgba = Gdk.RGBA()
		for widget in widgets:
			if widget is self.color_fg:
				rgba.parse(self.data['fg_color'])
			elif widget is self.color_bg:
				rgba.parse(self.data['bg_color'])

			widget.set_rgba(rgba)

	def load_widgets(self):
		self.builder = Gtk.Builder()
		self.builder.add_from_file('snake.ui')

		self.window = self.builder.get_object('Snake')
		self.panel = self.builder.get_object('PANEL')

		self.draw = self.builder.get_object('DRAW')
		self.tg_auto = self.builder.get_object('TG_AUTO')
		self.tg_run = self.builder.get_object('TG_RUN')
		self.lb_length = self.builder.get_object('LB_LENGTH')
		self.bt_speed = self.builder.get_object('BT_SPEED')
		self.color_fg = self.builder.get_object('COLOR_FG')
		self.color_bg = self.builder.get_object('COLOR_BG')
		self.area_combo = self.builder.get_object('AREA_COMBO')
		self.img_logo = self.builder.get_object('IMG_SNAKE')

	def load_image(self):
		sz_food = self.data['block_size'] * 1.2

		self.pix_icon = Pixbuf.new_from_file(self.data['image_icon'])
		self.pix_food = Pixbuf.new_from_file_at_size(self.data['image_snake_food'], sz_food, sz_food)
		self.pix_arrow = Pixbuf.new_from_file_at_size(self.data['image_arrow'], 28, 28)
		self.pix_arrow_key = Pixbuf.new_from_file_at_size(self.data['image_arrow_key'], 28, 28)

		self.img_logo.set_from_pixbuf(self.pix_icon.scale_simple(24, 24, InterpType.BILINEAR))

	def init_ui(self):
		self.load_widgets()		# load widgets
		self.load_image()		# load image resource

		# attach the window to app
		self.window.set_application(self)

		# main window
		self.window.set_title('Snake')
		self.window.set_icon(self.pix_icon)

		# connect keyevent
		self.window.connect('key-press-event', Handler.on_keyboard_event, self)
		self.window.connect('key-release-event', Handler.on_keyboard_event, self)

		# draw area
		self.draw.connect('draw', Handler.on_draw, self)

		# toggle button
		self.tg_auto.connect('toggled', Handler.on_toggled, self)
		self.tg_run.connect('toggled', Handler.on_toggled, self)
		# set toggle status on init
		self.tg_auto.toggled()
		self.tg_run.toggled()

		# spin of speed
		speed_adj = Gtk.Adjustment(**self.data['speed_adj'])
		self.bt_speed.set_adjustment(speed_adj)
		self.bt_speed.connect('value-changed', Handler.on_spin_value_changed, self)
		# set default speed on init, which will emit value-changed
		self.bt_speed.set_value(self.data['speed'])

		# color box
		self.color_fg.set_title('前景色')
		self.color_bg.set_title('背景色')
		self.color_fg.connect('color-set', Handler.on_color_set, self)
		self.color_bg.connect('color-set', Handler.on_color_set, self)
		# set color from data on init
		self.set_color(self.color_fg, self.color_bg)

		# arrow image
		self.arrows = {}
		for x in [ 'up', 'down', 'left', 'right' ]:
			self.arrows[x] = self.builder.get_object('IMG_{}'.format(x.upper()))
			self.arrows[x].set_from_pixbuf(self.pix_arrow.rotate_simple(self.map_arrow[x][0]))

		# area: combo box
		area_size_store = Gtk.ListStore(str)

		for size in self.data['block_area_list']:
			area_size_store.append([size])

		self.area_combo.set_model(area_size_store)
		self.area_combo.set_entry_text_column(0)
		self.area_combo.connect('changed', Handler.on_combo_changed, self)
		combo_entry = self.area_combo.get_child()
		combo_entry.connect('activate', Handler.on_combo_entry_activate, self)
		combo_entry.set_text(self.get_block_area_text())

		# request for draw area size on init
		self.req_draw_size_mini()

		# to avoid highlight in the entry
		#self.bt_speed.grab_focus_without_selecting()
		# or just avoid focus on those with entry
		self.tg_run.grab_focus()

		self.window.show_all()
		# remove focus on init, must after show
		self.window.set_focus(None)

	#@count_func_time
	def update_snake_graph(self):
		self.snake.graph_scan(self.debug['sub_mode'])
		self.snake.graph_path_scan(self.snake.food)

	#@count_func_time
	def check_update_after_move(self):
		"""
		what and when to update:
		1. the graph
			1.1 eat food
			1.2 off-path:
			1.3 force update
		2. the graph path
			2.1 after graph update
		"""
		# if in graph auto mode
		if self.data['tg_auto'] and self.debug['auto_mode'] == 0:
			# if eat food on move, or off-path
			if self.snake.graph is None or self.snake.head not in self.snake.graph_path:
				self.update_snake_graph()

	#@count_func_time
	def timer_move(self, data):
		if self.data['tg_auto'] and not self.snake_aim_buf:
			aim = self.snake.get_next_aim(self.debug['auto_mode'], self.debug['sub_mode'])
		else:
			aim = self.snake_aim_buf
			self.snake_aim_buf = None

		if self.snake.move(aim):
			""" if current function not end in time, the timeout callback will
			be delayed, which can be checked with time.process_time_ns() print
			"""
			self.timeout_id = GLib.timeout_add(1000/self.data['speed'], self.timer_move, None)

			self.lb_length.set_text('{}'.format(self.snake.length))
			self.check_update_after_move()
		else:
			self.timeout_id = None
			self.tg_run.set_sensitive(False)
			print('game over, died')

		self.draw.queue_draw()

	def rect_round(self, cr, x, y, lx, ly, r):
		cr.move_to(x, y+r)
		cr.arc(x + r, y + r, r, math.pi, -math.pi/2)
		cr.rel_line_to(lx - 2*r, 0)
		cr.arc(x + lx - r, y + r, r, -math.pi/2, 0)
		cr.rel_line_to(0, ly - 2*r)
		cr.arc(x + lx - r, y + ly - r, r, 0, math.pi/2)
		cr.rel_line_to(-lx + 2*r, 0)
		cr.arc(x + r, y + ly - r, r, math.pi/2, math.pi)
		cr.close_path()

	def cross_mark(self, cr, x, y, lx, ly, r):
		cr.move_to(x+r, y+r)
		cr.line_to(x+lx-r, y+ly-r)
		cr.move_to(x+lx-r, y+r)
		cr.line_to(x+r, y+ly-r)
		cr.stroke()

	def draw_gameover(self, cr):
		# relative to the snake window, attention to the transform before
		area_w = self.data['block_size'] * self.data['block_area']['width']
		area_h = self.data['block_size'] * self.data['block_area']['height']

		text_go = 'GAME OVER'
		text_reset = 'Press "r" to reset'

		cr.set_source_rgba(1, 0, 1, 0.8)
		cr.select_font_face('Serif', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)

		cr.set_font_size(48)
		extent_go = cr.text_extents(text_go)

		# litte above center
		cr.move_to((area_w - extent_go.width)/2, (area_h - extent_go.height)/2)
		cr.show_text(text_go)

		cr.set_font_size(20)
		extent_reset = cr.text_extents(text_reset)

		cr.move_to((area_w - extent_reset.width)/2, (area_h - extent_reset.height + extent_go.height)/2)
		cr.show_text(text_reset)

	def draw_snake_graph_path(self, cr):
		# graph path exist and not empty
		if self.snake.graph_path is None or len(self.snake.graph_path) == 0:
				return False

		l = self.data['block_size']

		cr.save()
		cr.set_source_rgba(0, 1, 1, 0.5)
		cr.set_line_width(0.2)

		cr.transform(cairo.Matrix(l, 0, 0, l, l/2, l/2))

		cr.move_to(*self.snake.graph_path[0])
		for pos in self.snake.graph_path[1:]:
			cr.line_to(*pos)

		cr.stroke()
		cr.restore()

	def draw_snake_graph(self, cr):
		if self.snake.graph is None:
			return False

		l = self.data['block_size']

		cr.save()
		cr.set_source_rgba(0, 1, 0, 0.8)
		cr.select_font_face('Serif', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
		cr.set_font_size(0.7)

		cr.transform(cairo.Matrix(l, 0, 0, l, l/2, l/2))

		for x,y in np.transpose(self.snake.graph.nonzero()):
			dist = self.snake.graph[x, y]
			extent = cr.text_extents(str(dist))
			cr.move_to(x - extent.width/2, y + extent.height/2)
			cr.show_text(str(dist))

		cr.restore()

	def draw_init(self, cr):
		width = self.draw.get_allocated_width()
		height = self.draw.get_allocated_height()

		area_w = self.data['block_size'] * self.data['block_area']['width']
		area_h = self.data['block_size'] * self.data['block_area']['height']

		context = self.draw.get_style_context()
		Gtk.render_background(context, cr, 0, 0, width, height)

		# draw background
		rgba = Gdk.RGBA()
		rgba.parse(self.data['bg_color'])
		cr.set_source_rgba(*rgba)
		cr.rectangle(0,0, width, height)
		cr.fill()

		# setup transformation
		scale = self.data['block_area_scale']
		# make sure center is center
		translate = (width/2 - scale * area_w/2, height/2 - scale * area_h/2)

		cr.transform(cairo.Matrix(scale, 0, 0, scale, *translate))

		# draw the edge
		cr.move_to(0, 0)
		cr.rel_line_to(0, area_h)
		cr.rel_line_to(area_w, 0)
		cr.rel_line_to(0, -area_h)
		cr.close_path()

		rgba.parse('blue')
		cr.set_source_rgba(*rgba)
		cr.set_line_width(self.data['block_size']/10)
		cr.set_line_join(cairo.LINE_JOIN_ROUND)
		cr.set_tolerance(0.1)
		cr.stroke()

	def draw_snake(self, cr):
		l = self.data['block_size']

		# draw food
		pix_sz = Vector(self.pix_food.get_width(), self.pix_food.get_height())
		food = self.snake.food * l + Vector(l, l)/2 - pix_sz/2
		Gdk.cairo_set_source_pixbuf(cr, self.pix_food, *food)

		cr.rectangle(*food, *pix_sz)
		cr.fill()

		# draw snake
		rgba = Gdk.RGBA()
		rgba.parse('black')
		cstr_00 = rgba.to_string()
		rgba.parse(self.data['bg_color'])
		cstr_bg = rgba.to_string()
		rgba.parse(self.data['fg_color'])
		cstr_fg = rgba.to_string()

		colorful = (cstr_fg == cstr_bg == cstr_00)

		if colorful:
			color = color_pool(self.snake.length)
		else:
			cr.set_source_rgba(*rgba)

		# scale the body block and center it in the grid
		s, r = (0.9, 0.2)
		xy_offset = (1-s)*l/2

		cr.save()
		# aligned to grid center
		cr.transform(cairo.Matrix(l, 0, 0, l, xy_offset, xy_offset))

		for block in self.snake.body:
			self.rect_round(cr, *block, s, s, r)
			if colorful:
				rgba.parse(color.__next__())
				cr.set_source_rgba(*rgba)
			cr.fill()

		if self.snake.is_died():
			cr.set_source_rgba(1, 0, 0, 1)
			cr.set_line_width(0.2)
			self.cross_mark(cr, *self.snake.head, s, s, r)

		cr.restore()

		if self.snake.is_died():
			self.draw_gameover(cr)

		if self.debug['show_graph']:
			self.draw_snake_graph(cr)
			self.draw_snake_graph_path(cr)

	def do_startup(self):
		Gtk.Application.do_startup(self)

	def do_activate(self):
		if not self.window:
			self.init_ui()
		else:
			self.window.present()


if __name__ == '__main__':
	random.seed()

	app = App()
	app.run(sys.argv)

# vi: set ts=4 noexpandtab foldmethod=indent :
