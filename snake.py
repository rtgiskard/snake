#!/usr/bin/env python
# coding: utf8

import sys

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

from functools import wraps

def echo_func(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		print("calling {} ..".format(func.__name__))
		return func(*args, **kwargs)

	return wrapper

def echo_func_count(func):
	counter = [0]

	@wraps(func)
	def wrapper(*args, **kwargs):
		print("calling {} x{} ..".format(func.__name__, counter[0]))
		counter[0] += 1
		return func(*args, **kwargs)

	return wrapper

class Handler:
	@classmethod
	#@echo_func_count
	def on_draw(cls, widget, cr, app):
		context = widget.get_style_context()

		width = widget.get_allocated_width()
		height = widget.get_allocated_height()
		Gtk.render_background(context, cr, 0, 0, width, height)

		bg_rgba = app.color_bg.get_rgba()
		cr.set_source_rgba(*bg_rgba)
		cr.rectangle(0,0, width, height)
		cr.fill()

		fg_rgba = app.color_fg.get_rgba()
		cr.set_source_rgba(*fg_rgba)
		cr.rectangle(width/2, height/2, 10, 60)
		cr.fill()

	@classmethod
	def on_toggled(cls, widget, app):
		def tg_text(label, active):
			try:
				lstr, rstr = label.split('/')
			except:
				return label

			if active:
				return "<small>{}</small>/<big><b>{}</b></big>".format(lstr, rstr)
			else:
				return "<big><b>{}</b></big>/<small>{}</small>".format(lstr, rstr)

		label = widget.get_label()
		widget_label = widget.get_child()
		widget_label.set_markup(tg_text(label, widget.get_active()))

	@classmethod
	def on_combo_changed(self, widget, app):
		t_iter = widget.get_active_iter()
		if t_iter is not None:
			model = widget.get_model()
			app.data["area_size"][0], app.data["area_size"][1] = (
					int(x) for x in model[t_iter][0].split('x') )

	@classmethod
	def on_combo_entry_activate(self, widget, app):
		entry_text = widget.get_text()
		try:
			wlen,hlen = ( int(x) for x in entry_text.split('x') )

			assert wlen >= 10 and wlen < 1000
			assert hlen >= 10 and hlen < 1000
		except:
			widget.set_text("{}x{}".format(*app.data["area_size"]))
			return None

		app.data["area_size"][0], app.data["area_size"][1] = ( wlen, hlen )

		# append the data to model if not found
		new_size = "{}x{}".format(*app.data["area_size"])
		model = app.area_combo.get_model()
		t_iter = model.get_iter_first()

		while t_iter is not None:
			# set input row active
			if model[t_iter][0] == new_size:
				app.area_combo.set_active_iter(t_iter)
				break
			else:
				t_iter = model.iter_next(t_iter)
		else:
			t_iter = model.append([new_size])
			app.area_combo.set_active_iter(t_iter)


	@classmethod
	def on_colorset(cls, widget, app):
		pass

class App(Gtk.Application):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, application_id="rt.game.snake", **kwargs)

		self.window = None

		# todo: check the initial value
		self.data = {
				"area_size": [0, 0],
				"bg_color": None,
				"fg_color": None,
				"tg_auto": False,
				"tg_pause": False,
				"speed": 2
				}

	def init_ui(self):
		self.builder = Gtk.Builder()
		self.builder.add_from_file("snake.ui")

		# main window
		self.window = self.builder.get_object("Snake")
		self.window.show_all()

		"""attach the window to app"""
		self.window.set_application(self)

		# draw area
		self.draw = self.builder.get_object("DRAW")
		self.draw.connect("draw", Handler.on_draw, self)
		#self.window.set_size_request(40, 40)
		#self.window.resize()

		# toggle button
		self.tg_auto = self.builder.get_object("TG_AUTO")
		self.tg_pause = self.builder.get_object("TG_PAUSE")
		self.tg_auto.connect("toggled", Handler.on_toggled, self)
		self.tg_pause.connect("toggled", Handler.on_toggled, self)
		# emit toggled signal
		self.tg_auto.toggled()
		self.tg_pause.toggled()

		# spin of speed
		speed_adj = Gtk.Adjustment(**{ "value": 2, "lower":1, "upper":20,
			"step_increment":1, "page_increment":10, "page_size":0 })
		self.bt_speed = self.builder.get_object("BT_SPEED")
		self.bt_speed.set_adjustment(speed_adj)

		# color box
		self.color_fg = self.builder.get_object("COLOR_FG")
		self.color_bg = self.builder.get_object("COLOR_BG")
		#self.color_fg.connect("color-set", Handler.on_colorset, self)
		#self.color_bg.connect("color-set", Handler.on_colorset, self)
		self.color_fg.set_title("前景色")
		self.color_bg.set_title("背景色")

		# area: combo box
		area_size_store = Gtk.ListStore(str)
		area_size_list = [ '10x10', '20x20', '40x40', '60x60', '80x80', '100x100' ]

		for size in area_size_list:
			area_size_store.append([size])

		self.area_combo = self.builder.get_object("AREA")
		self.area_combo.set_model(area_size_store)
		self.area_combo.set_entry_text_column(0)
		self.area_combo.set_active(2)
		self.area_combo.connect("changed", Handler.on_combo_changed, self)
		self.area_combo.get_child().connect("activate", Handler.on_combo_entry_activate, self)


	#@echo_func
	def do_startup(self):
		Gtk.Application.do_startup(self)

	#@echo_func
	def do_activate(self):
		# limit to one instance
		if not self.window:
			self.init_ui()
		else:
			self.window.present()

if __name__ == "__main__":
	app = App()
	app.run(sys.argv)

# vi: set ts=4 noexpandtab foldmethod=indent :
