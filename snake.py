#!/usr/bin/env python
# coding: utf8

import sys
import random, math
from functools import wraps

import gi
gi.require_version('Gtk', '3.0')

from gi.repository import Gtk, Gdk, GLib
from gi.repository.GdkPixbuf import Pixbuf, PixbufRotation, InterpType
import cairo


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
		return self.is_aim_valid(aim)

	def snake_reset(self):
		self.body = [ Vector(int(self.area_w/2), int(self.area_h/2)) ]
		self.food = self.new_food()
		self.aim = Vector(0,1)

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

		# if got food, generate new
		if self.head == self.food:
			self.food = self.new_food()
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

	def get_fast_aim(self, direct=True):
		pd_cross = self.aim.pd_cross(self.vec_head2food)
		pd_dot = self.aim.pd_dot(self.vec_head2food)

		if direct:
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

	def get_next_aim(self):
		aim_fast = self.get_fast_aim()
		aim_choices = [self.aim, self.aim.T, -self.aim.T]

		# aim_fast is already in aim_choices
		aim_choices.remove(aim_fast)

		# switch random
		if random.randint(0,1):
			aim_t = aim_choices[0]
			aim_choices[0] = aim_choices[1]
			aim_choices[1] = aim_t

		aim_choices.insert(0, aim_fast)

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
		keyname = Gdk.keyval_name(event.keyval).lower()

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
		super().__init__(*args, application_id='rt.game.snake', **kwargs)

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
		self.timeout_id = None
		self.snake_aim_buf = None

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

	def timer_move(self, data):
		if not self.snake_aim_buf and self.data['tg_auto']:
			aim = self.snake.get_next_aim()
		else:
			aim = self.snake_aim_buf
			self.snake_aim_buf = None

		if self.snake.move(aim):
			# set timer for next move
			self.timeout_id = GLib.timeout_add(1000/self.data['speed'], self.timer_move, None)
			self.lb_length.set_text('{}'.format(self.snake.length))
		else:
			self.tg_run.set_sensitive(False)
			print('game over, died')

		self.draw.queue_draw()

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

	def draw_text(self, cr):
		# relative to the snake window, attention to the transform before
		area_w = self.data['block_size'] * self.data['block_area']['width']
		area_h = self.data['block_size'] * self.data['block_area']['height']

		text_go = 'GAME OVER'
		text_reset = 'Press "r" to reset'

		cr.set_source_rgba(1, 0, 1, 0.8)
		cr.select_font_face('Serif', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)

		cr.set_font_size(48)
		extent_go = cr.text_extents(text_go)

		cr.move_to((area_w - extent_go.width)/2, (area_h - extent_go.height)/2)
		cr.show_text(text_go)

		cr.set_font_size(20)
		extent_reset = cr.text_extents(text_reset)

		cr.move_to((area_w - extent_reset.width)/2, (area_h - extent_reset.height + extent_go.height)/2)
		cr.show_text(text_reset)

	def draw_snake(self, cr):
		l = self.data['block_size']

		# draw food
		pix_sz = Vector(self.pix_food.get_width(), self.pix_food.get_height())
		pos = self.snake.food * l + Vector(l, l)/2 - pix_sz/2
		Gdk.cairo_set_source_pixbuf(cr, self.pix_food, pos.x, pos.y)
		cr.rectangle(pos.x, pos.y, pix_sz.x, pix_sz.y)
		cr.fill()

		# draw snake
		# todo: auto color
		rgba = Gdk.RGBA()
		rgba.parse('black')
		cstr_00 = rgba.to_string()
		rgba.parse(self.data['bg_color'])
		cstr_bg = rgba.to_string()
		rgba.parse(self.data['fg_color'])
		cstr_fg = rgba.to_string()

		colorful = (cstr_fg == cstr_bg == cstr_00)

		if not colorful:
			cr.set_source_rgba(*rgba)

		s = 0.9
		ls, r = (s * l, 0.2 * l)
		pos_offset = (1-s)/2 * l * Vector(1,1)

		def color_pool(n):
			for i in range(0, n):
				# set an offset so that blocks are not black
				yield '#{:0>6}'.format(hex(int(i/n * 0xeeeeee+0x101010))[2:])

		color = color_pool(self.snake.length)

		for block in self.snake.body:
			# aligned to grid center
			pos = block * l + pos_offset
			self.rect_round(cr, pos.x, pos.y, ls, ls, r)
			if colorful:
				rgba.parse(color.__next__())
				cr.set_source_rgba(*rgba)
			cr.fill()

		if self.snake.is_died():
			rgba.parse('red')
			cr.set_source_rgba(*rgba)
			pos = self.snake.head * l + pos_offset
			self.rect_round(cr, pos.x, pos.y, ls, ls, r)
			cr.fill()

			self.draw_text(cr)

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
