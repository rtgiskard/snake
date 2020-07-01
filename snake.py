#!/usr/bin/env python
# coding: utf8

import sys

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
from gi.repository.GdkPixbuf import Pixbuf, PixbufRotation, InterpType

from functools import wraps
import random

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

class Rotation:
	KEEP = [[1, 0], [0, 1]]
	COUNTERCLOCKWISE = [[0, 1], [-1, 0]]
	CLOCKWISE = [[0, -1], [1, 0]]

class Dot:
	def __init__(self, x=0, y=0):
		self.x = x
		self.y = y

	def __repr__(self):
		return "({}, {})".format(self.x, self.y)

class Vector(Dot):
	def __init__(self, x=0, y=0):
		super().__init__(x, y)

	def __neg__(self):
		return Vector(-self.x, -self.y)

	def __eq__(self, rhs):
		return self.x == rhs.x and self.y == rhs.y

	def __add__(self, rhs):
		return Vector(self.x + rhs.x, self.y + rhs.y)

	def __sub__(self, rhs):
		return Vector(self.x - rhs.x, self.y - rhs.y)

	def __iadd__(self, rhs):
		self.x += rhs.x
		self.y += rhs.y
		return self

	def __isub__(self, rhs):
		self.x -= rhs.x
		self.y -= rhs.y
		return self

	# todo: x / int

	# the dot product: self \cdot vec
	def dprod(self, vec):
		return self.x * vec.x + self.y * vec.y

	# the cross product: self \times vec
	def cprod(self, vec):
		return self.x * vec.y - self.y * vec.x

	def trans_linear(self, matrix, inplace=True):
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

		self.body = [ Vector(int(width/2), int(height/2)) ]

		self.food = Vector(x, y)

		self.aim = Vector(0,1)

	@property
	def length(self):
		return len(self.body)

	@property
	def vec_head2aim(self):
		return Vector(self.aim - self.head)

	@property
	def head(self):
		return self.body[0]
	@head.setter
	def head(self, point):
		self.head.x = point.x
		self.head.y = point.y

	def is_inside(self, point):
		return ( point.x >= 0 and point.x < self.area_w
				and point.y >= 0 and point.y < self.area_h )

	def is_full(self):
		return len(self.body) == self.area_h * self.area_w

	def is_valid(self, aim):
		head_next = self.head + aim
		return self.is_inside(head_next) and head_next not in self.body

	def move(self, aim=None):
		if not aim:
			aim = self.aim

		# insert the new head, pop old tail
		self.body.insert(0, self.head + aim)
		self.body.pop()

	def food_new(self):
		if self.is_full():
			return None

		# make sure the food is not in body
		while self.food in self.body:
			self.food = Vector(random.randrange(0, width), random.randrange(0, height))

		return self.food

	def get_fast_aim(self):
		cprod = self.vec_head2aim.cprod(self.aim)
		if cprod > 0:
			matrix = Rotation.COUNTERCLOCKWISE
		elif cprod < 0:
			matrix = Rotation.CLOCKWISE
		else:				# cprod == 0: 同向，反向，到达
			dprod = self.vec_head2aim.dprod(self.aim)
			if dprod >= 0:		# 同向或到达，保持
				return self.aim
			else:				# 反向，随机转向
				matrix = random.choice([Rotation.COUNTERCLOCKWISE, Rotation.CLOCKWISE])

		return self.aim.trans_linear(matrix, inplace=False)

	def get_aim(self):
		pass

class Handler:
	@classmethod
	def on_draw(cls, widget, cr, app):
		context = widget.get_style_context()

		width = widget.get_allocated_width()
		height = widget.get_allocated_height()
		Gtk.render_background(context, cr, 0, 0, width, height)

		rgba = Gdk.RGBA()
		rgba.parse(app.data['bg_color'])
		cr.set_source_rgba(*rgba)
		cr.rectangle(0,0, width, height)
		cr.fill()

		rgba.parse(app.data['fg_color'])
		cr.set_source_rgba(*rgba)
		cr.rectangle(width/2, height/2 - 20, 10, 60)
		cr.fill()

	@classmethod
	def on_toggled(cls, widget, app):
		label_text = widget.get_label()
		widget_label = widget.get_child()
		widget_label.set_markup(app.toggle_text(label_text, widget.get_active()))

		if widget is app.tg_auto:
			app.data['tg_auto'] = widget.get_active()
		elif widget is app.tg_pause:
			app.data['tg_pause'] = widget.get_active()

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

		if keyname in [ 'up', 'down', 'left', 'right' ]:
			if KEY_PRESS:
				pixbuf = app.pix_arrow_key
			elif KEY_RELEASE:
				pixbuf = app.pix_arrow

			app.arrows[keyname].set_from_pixbuf(pixbuf.rotate_simple(app.map_rotation[keyname]))
		elif KEY_PRESS and keyname == 'p':
			state = app.tg_pause.get_active()
			app.tg_pause.set_active(not state)
		elif KEY_PRESS and keyname == 'a':
			state = app.tg_auto.get_active()
			app.tg_auto.set_active(not state)

		return True

class App(Gtk.Application):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, application_id='rt.game.snake', **kwargs)

		self.window = None

		self.data = {
				'snake_width': 8,
				'block_size': 10,
				'block_area': {'width':40, 'height':40},
				'block_area_limit': {'min':10, 'max':999},
				'block_area_pix_margin': 20,
				'block_area_scale': {'x':1, 'y':1},
				'block_area_list': ( '{0}x{0}'.format(i*20) for i in range(1, 11) ),
				'bg_color': 'black',
				'fg_color': 'grey',
				'tg_auto': False,
				'tg_pause': False,
				'speed': 8,
				'speed_adj': { 'value':1, 'lower':1, 'upper':20,
					'step_increment':1, 'page_increment':10, 'page_size':0 },
				'image_icon': './data/icon/snake.svg',
				'image_arrow': './data/pix/arrow.svg',
				'image_arrow_key': './data/pix/arrow-key.svg',
				}

		self.map_rotation = {
				'up': PixbufRotation.NONE,
				'down': PixbufRotation.UPSIDEDOWN,
				'left': PixbufRotation.COUNTERCLOCKWISE,
				'right': PixbufRotation.CLOCKWISE,
				}

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
		block_size = self.data['block_size']
		block_area = self.data['block_area']
		margin = self.data['block_area_pix_margin']
		area = [ block_size * block_area['width'] + margin,
				block_size * block_area['height'] + margin ]

		# get current monitor resolution
		display = Gdk.Display.get_default()
		monitor = Gdk.Display.get_monitor(display, 0)
		rect = Gdk.Monitor.get_geometry(monitor)
		area_max = (int(rect.width * 0.9), int(rect.height * 0.9))

		if area[0] > area_max[0]:
			self.data['block_area_scale']['x'] = area[0]/area_max[0]
			area[0] = area_max[0]
		else:
			self.data['block_area_scale']['x'] = 1

		if area[1] > area_max[1]:
			self.data['block_area_scale']['y'] = area[1]/area_max[1]
			area[1] = area_max[1]
		else:
			self.data['block_area_scale']['y'] = 1

		self.draw.set_size_request(*area)

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
				widget.set_rgba(rgba)
			elif widget is self.color_bg:
				rgba.parse(self.data['bg_color'])
				widget.set_rgba(rgba)

	def load_image(self):
		self.pix_icon = Pixbuf.new_from_file(self.data['image_icon'])
		self.pix_arrow = Pixbuf.new_from_file_at_size(self.data['image_arrow'], 28, 28)
		self.pix_arrow_key = Pixbuf.new_from_file_at_size(self.data['image_arrow_key'], 28, 28)

		img_logo = self.builder.get_object('IMG_SNAKE')
		img_logo.set_from_pixbuf(self.pix_icon.scale_simple(20, 20, InterpType.BILINEAR))

	def init_ui(self):
		self.builder = Gtk.Builder()
		self.builder.add_from_file('snake.ui')

		# load image resource
		self.load_image()

		# main window
		self.window = self.builder.get_object('Snake')
		self.window.set_title('Snake')
		self.window.set_icon(self.pix_icon)
		self.window.show_all()

		# attach the window to app
		self.window.set_application(self)

		# draw area
		self.draw = self.builder.get_object('DRAW')
		self.draw.connect('draw', Handler.on_draw, self)

		# toggle button
		self.tg_auto = self.builder.get_object('TG_AUTO')
		self.tg_pause = self.builder.get_object('TG_PAUSE')
		self.tg_auto.connect('toggled', Handler.on_toggled, self)
		self.tg_pause.connect('toggled', Handler.on_toggled, self)
		# set toggle status on init
		self.tg_auto.toggled()
		self.tg_pause.toggled()

		# spin of speed
		speed_adj = Gtk.Adjustment(**self.data['speed_adj'])
		self.bt_speed = self.builder.get_object('BT_SPEED')
		self.bt_speed.set_adjustment(speed_adj)
		self.bt_speed.connect('value-changed', Handler.on_spin_value_changed, self)
		# set default speed on init, which will emit value-changed
		self.bt_speed.set_value(self.data['speed'])

		# color box
		self.color_fg = self.builder.get_object('COLOR_FG')
		self.color_bg = self.builder.get_object('COLOR_BG')
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
			self.arrows[x].set_from_pixbuf(self.pix_arrow.rotate_simple(self.map_rotation[x]))

		self.window.connect('key-press-event', Handler.on_keyboard_event, self)
		self.window.connect('key-release-event', Handler.on_keyboard_event, self)

		# area: combo box
		area_size_store = Gtk.ListStore(str)

		for size in self.data['block_area_list']:
			area_size_store.append([size])

		self.area_combo = self.builder.get_object('AREA_COMBO')
		self.area_combo.set_model(area_size_store)
		self.area_combo.set_entry_text_column(0)
		self.area_combo.connect('changed', Handler.on_combo_changed, self)
		combo_entry = self.area_combo.get_child()
		combo_entry.connect('activate', Handler.on_combo_entry_activate, self)
		combo_entry.set_text(self.get_block_area_text())

		# set draw area size request on init
		self.req_draw_size_mini()

		# remove focus by default
		self.window.set_focus(None)

	def do_startup(self):
		Gtk.Application.do_startup(self)

	def do_activate(self):
		if not self.window:
			self.init_ui()
		else:
			self.window.present()

if __name__ == '__main__':
	app = App()
	app.run(sys.argv)

# vi: set ts=4 noexpandtab foldmethod=indent :
