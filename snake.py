#!/usr/bin/env python
# coding: utf8

import sys
import random
from functools import wraps

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
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

		self.body = [ Vector(int(width/2), int(height/2)) ]
		self.food = self.food_new()

		self.aim = Vector(0,1)

	@property
	def length(self):
		return len(self.body)

	@property
	def area_size(self):
		return self.area_w * self.area_h

	@property
	def vec_head2food(self):
		return Vector(self.food - self.head)

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
		return self.length == self.area_size

	def is_aim_valid(self, aim):
		head_next = self.head + aim
		return self.is_inside(head_next) and head_next not in self.body

	def is_aim_right(self, aim):
		# todo: more check
		return self.is_aim_valid(aim)

	def area_resize(self, width, height):
		self.area_w = width
		self.area_h = height

	def move(self, aim=None):
		if not aim:
			aim = self.aim

		# insert the new head, pop old tail
		self.body.insert(0, self.head + aim)

		# if got food, generate new
		if self.food == self.head:
			self.food = self.food_new()
		else:
			self.body.pop()

	def auto_move(self):
		next_aim = self.get_next_aim()
		if next_aim:
			self.move(next_aim)
			return True
		else:
			print('game over, died')
			return False

	def food_new(self):
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

	def get_fast_aim(self):
		pd_cross = self.aim.pd_cross(self.vec_head2food)
		if pd_cross > 0:
			matrix = TransMatrix.ROTATE_LEFT
		elif pd_cross < 0:
			matrix = TransMatrix.ROTATE_RIGHT
		else:				# pd_cross == 0: 同向，反向，到达
			pd_dot = self.aim.pd_dot(self.vec_head2food)
			if pd_dot >= 0:		# 同向或到达，保持
				return self.aim
			else:				# 反向，随机转向
				matrix = random.choice([TransMatrix.ROTATE_LEFT, TransMatrix.ROTATE_RIGHT])

		return self.aim.trans_linear(matrix, inplace=False)

	def get_next_aim(self):
		aim_fast = self.get_fast_aim()
		aim_choices = [self.aim, self.aim.T, -self.aim.T]

		# aim_fast is already in aim_choices
		aim_choices.remove(aim_fast)

		# switch random
		if radom.randint(0,1):
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

			# disactive combo when the snake is run
			app.area_combo.set_sensitive(not active_state)

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
			state = app.tg_run.get_active()
			app.tg_run.set_active(not state)
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
				'block_area_margin': 20,
				'block_area_scale': 1,
				'block_area_list': ( '{0}x{0}'.format(i*20) for i in range(1, 11) ),
				'bg_color': 'black',
				'fg_color': 'grey',
				'tg_auto': False,
				'tg_run': False,
				'speed': 8,
				'speed_adj': { 'value':1, 'lower':1, 'upper':20,
					'step_increment':1, 'page_increment':10, 'page_size':0 },
				'image_icon': './data/icon/snake.svg',
				'image_arrow': './data/pix/arrow.svg',
				'image_arrow_key': './data/pix/arrow-key.svg',
				'image_snake_food': './data/pix/bonus5.svg',
				}

		self.map_rotation = {
				'up': PixbufRotation.NONE,
				'down': PixbufRotation.UPSIDEDOWN,
				'left': PixbufRotation.COUNTERCLOCKWISE,
				'right': PixbufRotation.CLOCKWISE,
				}

		self.snake = Snake(self.data['block_area']['width'], self.data['block_area']['height'])

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
		margin = self.data['block_area_margin']

		# get current monitor resolution
		display = Gdk.Display.get_default()
		monitor = Gdk.Display.get_monitor(display, 0)
		rect = Gdk.Monitor.get_geometry(monitor)

		area_lim = (int(rect.width * 0.9), int(rect.height * 0.9))
		area = [ block_size * block_area['width'] + margin,
				block_size * block_area['height'] + margin ]

		if area[0] > area_lim[0]:
			scale_x = area_lim[0]/area[0]
		else:
			scale_x = 1

		if area[1] > area_lim[1]:
			scale_y = area_lim[1]/area[1]
		else:
			scale_y = 1

		# use the smaller scale
		scale = scale_x if scale_x < scale_y else scale_y
		self.data['block_area_scale'] = scale

		# snake resize
		self.snake.area_resize(block_area['width'], block_area['height'])

		# request for mini size
		self.draw.set_size_request(area[0] * scale, area[1] * scale)

		# queue draw
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
				widget.set_rgba(rgba)
			elif widget is self.color_bg:
				rgba.parse(self.data['bg_color'])
				widget.set_rgba(rgba)

	def load_widgets(self):
		self.builder = Gtk.Builder()
		self.builder.add_from_file('snake.ui')

		self.window = self.builder.get_object('Snake')
		self.draw = self.builder.get_object('DRAW')
		self.tg_auto = self.builder.get_object('TG_AUTO')
		self.tg_run = self.builder.get_object('TG_RUN')
		self.bt_speed = self.builder.get_object('BT_SPEED')
		self.color_fg = self.builder.get_object('COLOR_FG')
		self.color_bg = self.builder.get_object('COLOR_BG')
		self.area_combo = self.builder.get_object('AREA_COMBO')
		self.img_logo = self.builder.get_object('IMG_SNAKE')

	def load_image(self):
		self.pix_icon = Pixbuf.new_from_file(self.data['image_icon'])
		self.pix_food = Pixbuf.new_from_file(self.data['image_snake_food'])
		self.pix_arrow = Pixbuf.new_from_file_at_size(self.data['image_arrow'], 28, 28)
		self.pix_arrow_key = Pixbuf.new_from_file_at_size(self.data['image_arrow_key'], 28, 28)

		self.img_logo.set_from_pixbuf(self.pix_icon.scale_simple(20, 20, InterpType.BILINEAR))

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
			self.arrows[x].set_from_pixbuf(self.pix_arrow.rotate_simple(self.map_rotation[x]))

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

		# remove focus on init, show
		self.window.set_focus(None)
		self.window.show_all()

	def draw_init(self, cr):
		width = self.draw.get_allocated_width()
		height = self.draw.get_allocated_height()

		area_w = self.data['block_size'] * self.data['block_area']['width']
		area_h = self.data['block_size'] * self.data['block_area']['height']

		context = self.draw.get_style_context()
		Gtk.render_background(context, cr, 0, 0, width, height)

		# draw background
		rgba = Gdk.RGBA()
		rgba.parse(app.data['bg_color'])
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
		cr.set_line_width(self.data['block_size']/5)
		cr.set_line_join(cairo.LINE_JOIN_ROUND)
		cr.set_tolerance(0.1)
		cr.stroke()

	def draw_snake(self, cr):
		rgba = Gdk.RGBA()
		rgba.parse(app.data['fg_color'])
		cr.set_source_rgba(*rgba)
		cr.rectangle(60, 40, 10, 60)
		cr.fill()

		rgba.parse(app.data['fg_color'])
		cr.set_source_rgba(*rgba)
		cr.rectangle(80, 70, 10, 60)
		cr.fill()

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
