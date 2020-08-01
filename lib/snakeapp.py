#!/usr/bin/env python
# coding: utf8

import math
import numpy as np
import json

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('PangoCairo', '1.0')

from gi.repository import Gio, Gtk, Gdk, GLib
from gi.repository import Pango, PangoCairo
from gi.repository.GdkPixbuf import Pixbuf, PixbufRotation, InterpType
import cairo

from .datatypes import *
from .decorator import *
from .dataref import *
from .snake import Snake

from . import NAME, VERSION, AUTHOR, COPYRIGHT, LICENSE_TYPE

iprint = lambda msg: print(f'[INFO]: {msg}')

class Draw_Pack:
	def __init__(self, cr=None):
		self.cr = cr

	def rect_round(self, x, y, lx, ly, r):
		self.cr.move_to(x, y+r)
		self.cr.arc(x + r, y + r, r, math.pi, -math.pi/2)
		self.cr.rel_line_to(lx - 2*r, 0)
		self.cr.arc(x + lx - r, y + r, r, -math.pi/2, 0)
		self.cr.rel_line_to(0, ly - 2*r)
		self.cr.arc(x + lx - r, y + ly - r, r, 0, math.pi/2)
		self.cr.rel_line_to(-lx + 2*r, 0)
		self.cr.arc(x + r, y + ly - r, r, math.pi/2, math.pi)
		self.cr.close_path()

	def circle_mark(self, x, y, lx, ly, r):
		self.cr.arc(x+lx/2, y+ly/2, r, 0, 2*math.pi)
		self.cr.stroke()

	def cross_mark(self, x, y, lx, ly, r):
		self.cr.move_to(x+r, y+r)
		self.cr.line_to(x+lx-r, y+ly-r)
		self.cr.move_to(x+lx-r, y+r)
		self.cr.line_to(x+r, y+ly-r)
		self.cr.stroke()

class Handler:
	@classmethod
	def on_draw(cls, widget, cr, app):
		app.update_draw(cr)

		# stop event pass on
		return True

	@classmethod
	def on_toggled(cls, widget, app):
		"""get avtive state, sync to data"""
		active_state = widget.get_active()
		widget_label = widget.get_child()
		label_text = widget_label.get_text()
		widget_label.set_markup(app.toggle_text(label_text, active_state))

		if widget is app.tg_auto:
			app.data['tg_auto'] = active_state
		elif widget is app.tg_run:
			app.data['tg_run'] = active_state

			if active_state:
				# disable combo once snake active
				app.area_combo.set_sensitive(False)

				app.timeout_id = GLib.timeout_add(1000/app.data['speed'], app.timer_move, None)
			else:
				if app.timeout_id:
					GLib.source_remove(app.timeout_id)
					app.timeout_id = None

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
	def on_about_exit(cls, widget, response, app):
		widget.destroy()
		app.about_dialog = None

	@classmethod
	def on_reset(cls, action, param, app):
		app.reset_game(reset_all=True)

	@classmethod
	def on_about(cls, action, param, app):
		app.show_about_dialog()

	@classmethod
	def on_save(cls, action, param, app):
		app.save_or_load_game(is_save=True)

	@classmethod
	def on_load(cls, action, param, app):
		app.save_or_load_game(is_save=False)

	@classmethod
	def on_keyboard_event(cls, widget, event, app):
		"""
			keyshot:

			ui:
				esc:		unfocus widget
				tab:		switch focus
				h:			hide/unhide panel

			game control:
				r:			reset after gameover
				p:			toggle pause/continue
				a:			toggle auto/manual
				m:			switch auto mode
				s:			submode switch

				[]:			speed down/up
				←→↑↓:		direction control

			debug:
				t:			toggle trace display
				g:			toogle path and graph display
				G:			force rescan, then display path and graph
				x:			switch the display of (path, graph)
				R:			snake reseed

			accel:
				<Ctrl>R		reset game
				<Ctrl>S		pause and save game
				<Ctrl>L		pause and load game
		"""
		KeyName = Gdk.keyval_name(event.keyval)
		keyname = KeyName.lower()

		# if <Ctrl>, return and pass on
		if event.state & Gdk.ModifierType.CONTROL_MASK:
			return False

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

		# debug related keyshot
		if KEY_PRESS and keyname == 'h':
			app.panel.set_visible(not app.panel.get_visible())

		elif KEY_PRESS and keyname == 't':
			app.data['show_trace'] = not app.data['show_trace']
			app.draw.queue_draw()

		elif KEY_PRESS and keyname == 'x':
			"""
				display_md:
					True: regular: [path, graph]
					False: unsafe: [unsafe_path/col_path, graph_col]
			"""
			app.data['display_md'] = not app.data['display_md']

			iprint(f"display regular: {app.data['display_md']}")
			app.draw.queue_draw()

		elif KEY_PRESS and keyname == 'g':
			if KeyName == 'G':
				if not app.dpack.died:
					# scan only on alive
					app.snake.update_path_and_graph()
				app.data['show_graph'] = True
			else:
				app.data['show_graph'] = not app.data['show_graph']

			iprint(f"show map: {app.data['show_graph']}")
			app.draw.queue_draw()

		if app.dpack.died:
			if KEY_PRESS and keyname == 'r':
				app.reset_game()
				iprint('game reset')

			return True

		# forbid player mode related keyshot after died

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

		elif KEY_PRESS and KeyName == 'R':
			app.snake.reseed()
			iprint('snake random: reseed')

		elif KEY_PRESS and keyname == 's':
			app.data['sub_switch'] = not app.data['sub_switch']
			iprint(f"sub switch: {app.data['sub_switch']}")

		elif KEY_PRESS and keyname == 'm':
			automode_list = list(AutoMode)
			id_cur = automode_list.index(app.data['auto_mode'])
			id_next = (id_cur + 1) % len(automode_list)
			app.data['auto_mode'] = automode_list[id_next]
			iprint(f"auto_mode: {app.data['auto_mode'].name}")

		return True

class SnakeApp(Gtk.Application):
	def __init__(self, *args, **kwargs):
		# allow multiple instance
		super().__init__(*args, application_id='rt.game.snake',
				flags=Gio.ApplicationFlags.NON_UNIQUE, **kwargs)

		self.window = None

		self.data = {
				'snake_width': 8,
				'block_size': 16,
				'block_area': {'width':40, 'height':28},
				'block_area_limit': {'min':10, 'max':999},
				'block_area_scale': 1,
				'block_area_list': ( f'{i*20}x{i*20}' for i in range(1, 11) ),
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

				'auto_mode': AutoMode.GRAPH,
				'sub_switch': True,
				'show_trace': False,
				'show_graph': False,
				'display_md': True,
				}

		# which state to save on game save
		self.state_to_dump = (
				'block_area', 'bg_color', 'fg_color', 'speed', 'tg_auto',
				'auto_mode', 'sub_switch', 'show_graph', 'show_trace'
				)

		# 注意绘图座标系正负与窗口上下左右的关系
		self.map_arrow = {
				'up': (PixbufRotation.NONE, VECTORS.UP),
				'down': (PixbufRotation.UPSIDEDOWN, VECTORS.DOWN),
				'left': (PixbufRotation.COUNTERCLOCKWISE, VECTORS.LEFT),
				'right': (PixbufRotation.CLOCKWISE, VECTORS.RIGHT)
				}

		self.snake = Snake(self.data['block_area']['width'], self.data['block_area']['height'])
		self.snake_aim_buf = None

		# for share of draw parameters
		self.dpack = Draw_Pack()

		self.timeout_id = None
		self.about_dialog = None

	## game op: reset, save/load ##

	def reset_game(self, reset_all=False):
		# reset snake and app
		self.snake.snake_reset()
		self.snake_aim_buf = None

		# reset timeout id
		if self.timeout_id:
			GLib.source_remove(self.timeout_id)
			self.timeout_id = None

		# reset widgets
		self.tg_run.set_active(False)
		self.tg_auto.set_active(False)

		# re-activate widgets
		self.tg_run.set_sensitive(True)
		self.area_combo.set_sensitive(True)

		if reset_all:
			self.data['speed'] = 8
			self.data['fg_color'] = 'grey'
			self.data['bg_color'] = 'black'
			self.data['block_area'] = {'width':40, 'height':28}
			self.data['auto_mode'] = AutoMode.GRAPH
			self.data['sub_switch'] = True
			self.data['show_graph'] = False
			self.data['show_trace'] = False
			self.data['display_md'] = True

		self.init_state_from_data()

	def save_or_load_game(self, is_save=True):
		"""pause on save or load"""

		self.tg_run.set_active(False)

		filename = self.run_filechooser(is_save)

		if filename is None:
			return True

		if is_save:
			text = 'save'
			with open(filename, 'w') as fd:
				json.dump(self.dump_data_json(), fd)
				return True
		else:
			text = 'load'
			with open(filename, 'r') as fd:
				if self.load_data_json(json.load(fd)):
					return True

		# pop dialog for failed operation
		self.show_warning_dialog(f"Failed to {text} game")

		return False

	def dump_data_json(self):
		snake_data = self.snake.snake_dump()
		app_data = { x:self.data[x] for x in self.state_to_dump }

		# convert auto_mode to str
		app_data['auto_mode'] = app_data['auto_mode'].name

		return { 'snake': snake_data, 'app': app_data }

	def load_data_json(self, data):
		# reset current game
		self.reset_game(reset_all=True)

		# load snake first
		if not self.snake.snake_load(data['snake']):
			return False

		# load data for app, without verification
		# load only keys in data and state_to_dump
		for key in data['app'].keys():
			if key in self.state_to_dump:
				self.data[key] = data['app'][key]

		# convert auto_mode back to enum
		self.data['auto_mode'] = AutoMode[self.data['auto_mode']]

		# recover gui state, set snake length label
		self.init_state_from_data()

		# pause and de-sensitive combo_entry after restore
		self.tg_run.set_active(False)
		self.area_combo.set_sensitive(False)

		return True

	## the real snake ##

	#@count_func_time
	def timer_move(self, data):
		if self.data['tg_auto'] and not self.snake_aim_buf:
			aim = self.snake.get_auto_aim(self.data['auto_mode'], self.data['sub_switch'])
		else:
			aim = self.snake_aim_buf
			self.snake_aim_buf = None

		if self.snake.move(aim):
			""" if current function not end in time, the timeout callback will
			be delayed, which can be checked with time.process_time_ns()
			"""
			self.timeout_id = GLib.timeout_add(1000/self.data['speed'], self.timer_move, None)

			self.lb_length.set_text(f"{self.snake.length}")
			self.check_and_update_after_move()
		else:
			self.dpack.died = True
			self.timeout_id = None
			self.tg_run.set_sensitive(False)
			iprint('game over, died')

		self.draw.queue_draw()

	#@count_func_time
	def check_and_update_after_move(self):
		"""
		when to update:
		. eat food
		. off-path: head not in path
		. end of path
		. force update
		"""
		# if in graph-auto mode
		if self.data['tg_auto'] and self.data['auto_mode'] == AutoMode.GRAPH:
			# if eat food on move, off-path, or at end of path
			if self.snake.head not in self.snake.path[:-1]:
				self.snake.update_path_and_graph()


	## dialog related ##

	def run_filechooser(self, is_save=True):
		if is_save:
			dialog_action = Gtk.FileChooserAction.SAVE
			dialog_button = Gtk.STOCK_SAVE
		else:
			dialog_action = Gtk.FileChooserAction.OPEN
			dialog_button = Gtk.STOCK_OPEN

		dialog = Gtk.FileChooserDialog(
			'Select File', self.window, dialog_action,
			(
				Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
				dialog_button, Gtk.ResponseType.OK,
			),
		)

		self.filechooser_filter(dialog)

		if dialog.run() == Gtk.ResponseType.OK:
			filename = dialog.get_filename()
		else:
			filename = None

		dialog.destroy()

		return filename

	def filechooser_filter(self, dialog):
		filter_json = Gtk.FileFilter()
		filter_json.set_name('Json')
		filter_json.add_mime_type('application/json')
		dialog.add_filter(filter_json)

		filter_any = Gtk.FileFilter()
		filter_any.set_name('Any files')
		filter_any.add_pattern('*')
		dialog.add_filter(filter_any)

	def show_about_dialog(self):
		if self.about_dialog:
			self.about_dialog.present()
		else:
			about_dia = Gtk.AboutDialog()
			self.about_dialog = about_dia

			# the close button will not issue 'close' event
			about_dia.connect('response', Handler.on_about_exit, self)

			# about dialog
			about_dia.set_authors([AUTHOR])
			about_dia.set_program_name(NAME)
			about_dia.set_version(VERSION)
			about_dia.set_copyright(COPYRIGHT)
			about_dia.set_license_type(Gtk.License.__dict__[LICENSE_TYPE])
			about_dia.set_logo(self.pix_icon)
			about_dia.set_destroy_with_parent(True)
			about_dia.set_title(f"About {NAME}")

			about_dia.show()

	def show_warning_dialog(self, text):
		dialog = Gtk.MessageDialog(
			self.window,
			Gtk.DialogFlags.DESTROY_WITH_PARENT,
			Gtk.MessageType.WARNING,
			Gtk.ButtonsType.CLOSE,
			text,
		)

		# lambda itself is the callback function
		dialog.connect('response', lambda *args: dialog.destroy())
		dialog.show()

	## ui related op ##

	def init_state_from_data(self):
		"""called on init_ui, reset, load game"""

		# reset bt_speed
		self.bt_speed.set_value(self.data['speed'])

		# reset color
		self.set_color(self.color_fg, self.color_bg)

		# the toggle button
		self.tg_auto.set_active(self.data['tg_auto'])
		self.tg_run.set_active(self.data['tg_run'])

		# update the combo_entry for area size only, no more operation
		combo_entry = self.area_combo.get_child()
		combo_entry.set_text(self.get_block_area_text())

		# area resize, queue redraw, may reset snake if resize
		self.req_draw_size_mini()

		# reset length label from snake
		self.lb_length.set_text(f"{self.snake.length}")

	def get_block_area_text(self):
		area = self.data['block_area']
		return f"{area['width']}x{area['height']}"

	def sync_block_area_from_text(self, text):
		try:
			width, height = ( int(x) for x in text.split('x') )

			if width < self.data['block_area_limit']['min'] or \
				width > self.data['block_area_limit']['max'] or \
				height < self.data['block_area_limit']['min'] or \
				height > self.data['block_area_limit']['max']:
				raise Exception()
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
			return f'<small>{lstr}</small>/<big><b>{rstr}</b></big>'
		else:
			return f'<big><b>{lstr}</b></big>/<small>{rstr}</small>'

	def init_draw_pack(self):
		dp = self.dpack

		# dynamic value: the draw widget size
		dp.wd_w = lambda: self.draw.get_allocated_width()
		dp.wd_h = lambda: self.draw.get_allocated_height()

		# snake area size
		dp.area_w = self.data['block_area']['width']
		dp.area_h = self.data['block_area']['height']

		dp.scale = self.data['block_area_scale']

		dp.l = self.data['block_size']	# block grid side length, in pixel
		dp.s = 0.9						# body block side length, relative to dp.l
		dp.r = 0.2						# body block corner radius, relative to dp.l

		dp.fn = 'monospace'				# dist mao text font name
		dp.fs = 0.7						# dist map text font size, relative to dp.l

		# snake alive cache
		dp.died = self.snake.is_died()

		# color
		rgba = Gdk.RGBA()			# free rgba

		rgba.parse(self.data['bg_color'])
		dp.rgba_bg = tuple(rgba)
		rgba.parse(self.data['fg_color'])
		dp.rgba_fg = tuple(rgba)

		dp.rgba_mark = (*dp.rgba_bg[:3], 0.8)	# mark: use bg color, but set alpha to 0.8
		dp.rgba_trace = (*dp.rgba_bg[:3], 0.4)	# trace: use bg color, but set alpha to 0.4
		dp.rgba_path = None
		dp.rgba_path_0 = (0, 1, 1, 0.6)			# path: regular mode
		dp.rgba_path_1 = (1, 0, 1, 0.6)			# path: unsafe mode
		dp.rgba_text = (0, 1, 0, 0.8)			# text for dist map
		dp.rgba_over = (1, 0, 1, 0.8)			# game over text
		dp.rgba_edge = (0, 0, 1, 1)				# edge: blue
		dp.rgba_black = (0, 0, 0, 1)			# black reference

		# fg == bg == black: colorful
		if (dp.rgba_bg == dp.rgba_fg == dp.rgba_black):
			# the color gradient rely on snake's dynamic length
			dp.body_color = lambda i: Color_Grad(self.snake.length)[i]
		else:
			dp.body_color = lambda i: dp.rgba_fg

	def req_draw_size_mini(self):
		"""call on window resized or init_state_from_data"""

		blk_sz = self.data['block_size']
		area_w = self.data['block_area']['width']
		area_h = self.data['block_area']['height']

		# get current monitor resolution
		display = Gdk.Display.get_default()
		monitor = Gdk.Display.get_monitor(display, 0)
		rect = Gdk.Monitor.get_geometry(monitor)

		area_lim = (int(rect.width * 0.9), int(rect.height * 0.9))
		area = [ blk_sz * (area_w + 2), blk_sz * (area_h + 2) ]

		scale_x = min(area[0], area_lim[0])/area[0]
		scale_y = min(area[1], area_lim[1])/area[1]

		# use the smaller scale
		scale = min(scale_x, scale_y)
		self.data['block_area_scale'] = scale

		if self.snake.area_w != area_w or self.snake.area_h != area_h:
			# snake resize && reset is not match
			self.snake.area_resize(area_w, area_h, True)

		# init/sync draw pack
		self.init_draw_pack()

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

		self.init_draw_pack()

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
		self.builder.add_from_file('data/ui/snake.ui')

		self.window = self.builder.get_object('Snake')
		self.panel = self.builder.get_object('PANEL')
		self.header = self.builder.get_object('HEADER')

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

	def init_menu(self):
		menu_items = [
				('Reset', Handler.on_reset, ['<Ctrl>R']),
				('Save', Handler.on_save, ['<Ctrl>S']),
				('Load', Handler.on_load, ['<Ctrl>L']),
				('About', Handler.on_about, [])
			]

		menu = Gio.Menu()

		for item in menu_items:
			action_name = item[0].lower()
			action_dname = 'app.' + action_name
			action_label = item[0]
			action_callback = item[1]
			action_accels = item[2]

			action = Gio.SimpleAction.new(action_name, None)
			action.connect('activate', action_callback, self)
			self.add_action(action)

			self.set_accels_for_action(action_dname, action_accels)
			menu.append(action_label, action_dname)

		self.set_app_menu(menu)

	def init_ui(self):
		self.load_widgets()		# load widgets
		self.load_image()		# load image resource
		self.init_menu()		# init app menu

		# attach the window to app
		self.window.set_application(self)

		# header bar
		self.header.set_decoration_layout('menu:minimize,close')

		# main window
		self.window.set_title(NAME)
		self.window.set_icon(self.pix_icon)

		# connect keyevent
		self.window.connect('key-press-event', Handler.on_keyboard_event, self)
		self.window.connect('key-release-event', Handler.on_keyboard_event, self)

		# draw area
		self.draw.connect('draw', Handler.on_draw, self)

		# toggle button
		self.tg_auto.connect('toggled', Handler.on_toggled, self)
		self.tg_run.connect('toggled', Handler.on_toggled, self)
		# init via toggle, set_active() only trigger if state changed
		self.tg_auto.toggled()
		self.tg_run.toggled()

		# spin of speed
		speed_adj = Gtk.Adjustment(**self.data['speed_adj'])
		self.bt_speed.set_adjustment(speed_adj)
		self.bt_speed.connect('value-changed', Handler.on_spin_value_changed, self)

		# color box
		self.color_fg.set_title('前景色')
		self.color_bg.set_title('背景色')
		self.color_fg.connect('color-set', Handler.on_color_set, self)
		self.color_bg.connect('color-set', Handler.on_color_set, self)

		# arrow image
		self.arrows = {}
		for x in [ 'up', 'down', 'left', 'right' ]:
			self.arrows[x] = self.builder.get_object(f'IMG_{x.upper()}')
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

		# init gui state from data
		self.init_state_from_data()

		# to avoid highlight in the entry
		#self.bt_speed.grab_focus_without_selecting()
		# or just avoid focus on those with entry
		self.tg_run.grab_focus()

		self.window.show_all()
		# remove focus on init, must after show
		self.window.set_focus(None)

	## Draw related ##

	#@count_func_time
	def update_draw(self, cr):
		"""op pack for update draw"""

		dp = self.dpack
		dp.cr = cr

		self.draw_init(dp)
		self.draw_snake(dp)

	def draw_init(self, dp):
		cr = dp.cr

		# draw background
		cr.set_source_rgba(*dp.rgba_bg)
		cr.rectangle(0,0, dp.wd_w(), dp.wd_h())
		cr.fill()

		# or use theme-provided background
		#context = self.draw.get_style_context()
		#Gtk.render_background(context, cr, 0, 0, dp.wd_w(), dp.wd_h())

		# make sure center is center
		translate = (
				(dp.wd_w() - dp.scale * dp.l * dp.area_w)/2,
				(dp.wd_h() - dp.scale * dp.l * dp.area_h)/2
				)
		cr.transform(cairo.Matrix(dp.scale, 0, 0, dp.scale, *translate))

		cr.set_line_join(cairo.LINE_JOIN_ROUND)
		cr.set_tolerance(0.2)

		cr.save()
		cr.scale(dp.l, dp.l)

		# draw the edge
		cr.move_to(0, 0)
		cr.rel_line_to(0, dp.area_h)
		cr.rel_line_to(dp.area_w, 0)
		cr.rel_line_to(0, -dp.area_h)
		cr.close_path()

		cr.set_source_rgba(*dp.rgba_edge)
		cr.set_line_width(0.1)
		cr.stroke()
		cr.restore()

	def draw_snake(self, dp):
		cr = dp.cr

		# food
		if self.snake.food:
			pix_sz = Vector(self.pix_food.get_width(), self.pix_food.get_height())
			food = self.snake.food * dp.l + (Vector(dp.l, dp.l) - pix_sz)/2
			Gdk.cairo_set_source_pixbuf(cr, self.pix_food, *food)

			cr.rectangle(*food, *pix_sz)
			cr.fill()

		cr.save()

		# aligned to grid center
		xy_offset = (1-dp.s)*dp.l/2
		cr.transform(cairo.Matrix(dp.l, 0, 0, dp.l, xy_offset, xy_offset))

		# snake body
		for i in range(self.snake.length):
			dp.rect_round(*self.snake.body[i], dp.s, dp.s, dp.r)
			cr.set_source_rgba(*dp.body_color(i))
			cr.fill()

		# head mark
		cr.set_source_rgba(*dp.rgba_mark)

		if dp.died:
			cr.set_line_width(0.2)
			dp.cross_mark(*self.snake.head, dp.s, dp.s, dp.r)
		else:
			cr.set_line_width(0.12)
			dp.circle_mark(*self.snake.head, dp.s, dp.s, dp.s/4)

		cr.restore()

		if self.data['show_trace']:
			self.draw_snake_trace(dp)

		if self.data['show_graph']:
			if self.data['display_md']:
				dp.rgba_path = dp.rgba_path_0
				path, graph = (self.snake.path, self.snake.graph)
			else:
				dp.rgba_path = dp.rgba_path_1
				path, graph = (self.snake.path_col, self.snake.graph_col)
				if len(path) == 0:
					path = self.snake.path_unsafe

			self.draw_snake_graph_cairo(dp, graph)
			self.draw_snake_path(dp, path)

		if dp.died:
			self.draw_gameover(dp)

	def draw_snake_trace(self, dp):
		cr = dp.cr

		cr.save()
		cr.transform(cairo.Matrix(dp.l, 0, 0, dp.l, dp.l/2, dp.l/2))

		cr.move_to(*self.snake.body[0])
		for pos in self.snake.body[1:]:
			cr.line_to(*pos)

		cr.set_source_rgba(*dp.rgba_trace)
		cr.set_line_width(0.1)
		cr.stroke()
		cr.restore()

	def draw_snake_path(self, dp, path):
		# graph path exist and not empty
		if path is None or len(path) == 0:
			return False

		cr = dp.cr

		cr.save()
		cr.transform(cairo.Matrix(dp.l, 0, 0, dp.l, dp.l/2, dp.l/2))

		cr.move_to(*path[0])
		for pos in path[1:]:
			cr.line_to(*pos)

		cr.set_source_rgba(*dp.rgba_path)
		cr.set_line_width(0.2)
		cr.stroke()
		cr.restore()

	def draw_snake_graph_pango(self, dp, graph):
		if graph is None:
			return False

		cr = dp.cr

		cr.save()
		cr.translate(dp.l/2, dp.l/2)

		font_desc = Pango.FontDescription.from_string(dp.fn)

		# set obsolute size in pixel and scaled to pango size
		font_desc.set_absolute_size(dp.fs * dp.l * Pango.SCALE)
		font_desc.set_weight(Pango.Weight.NORMAL)

		pg_layout = PangoCairo.create_layout(cr)
		pg_layout.set_font_description(font_desc)

		cr.set_source_rgba(*dp.rgba_text)

		for x,y in np.transpose(graph.nonzero()):
			dist = graph[x, y]
			pg_layout.set_text(str(dist), -1)

			# without scale, use pixel size directly
			width, height = pg_layout.get_pixel_size()

			cr.move_to(x * dp.l - width/2, y * dp.l - height/2)
			PangoCairo.show_layout(cr, pg_layout)

		cr.restore()

	def draw_snake_graph_cairo(self, dp, graph):
		if graph is None:
			return False

		cr = dp.cr

		cr.save()
		cr.transform(cairo.Matrix(dp.l, 0, 0, dp.l, dp.l/2, dp.l/2))

		cr.set_source_rgba(*dp.rgba_text)
		cr.select_font_face(dp.fn, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
		cr.set_font_size(dp.fs)

		for x,y in np.transpose(graph.nonzero()):
			dist = graph[x, y]
			extent = cr.text_extents(str(dist))
			cr.move_to(x - extent.width/2, y + extent.height/2)
			cr.show_text(str(dist))

		cr.restore()

	def draw_gameover(self, dp):
		cr = dp.cr

		text_go = 'GAME OVER'
		text_reset = 'Press "r" to reset'

		cr.set_source_rgba(*dp.rgba_over)
		cr.select_font_face('Serif', cairo.FONT_SLANT_OBLIQUE, cairo.FONT_WEIGHT_BOLD)
		cr.set_font_size(48)
		extent_go = cr.text_extents(text_go)

		# litte above center
		cr.move_to((dp.l * dp.area_w - extent_go.width)/2,
				(dp.l * dp.area_h - extent_go.height)/2)
		cr.show_text(text_go)

		cr.select_font_face(dp.fn, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
		cr.set_font_size(20)
		extent_reset = cr.text_extents(text_reset)

		cr.move_to((dp.l * dp.area_w - extent_reset.width)/2,
				(dp.l * dp.area_h - extent_reset.height + extent_go.height)/2)
		cr.show_text(text_reset)

	## App actions ##

	def do_startup(self):
		Gtk.Application.do_startup(self)

	def do_activate(self):
		if not self.window:
			self.init_ui()
		else:
			self.window.present()

# vi: set ts=4 noexpandtab foldmethod=indent foldignore= :
