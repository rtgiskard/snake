#!/usr/bin/env python
# coding: utf8

import sys

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

from functools import wraps

def echo_func(func):
    """went wrong with wraps"""
    #@wraps
    def wrapper(*args, **kwargs):
        print("calling {} ..".format(func.__name__))
        return func(*args, **kwargs)

    return wrapper

class Handler:
    @classmethod
    def on_draw(cls, widget, cr):
        context = widget.get_style_context()

        width = widget.get_allocated_width()
        height = widget.get_allocated_height()
        Gtk.render_background(context, cr, 0, 0, width, height)

        cr.set_source_rgba(0,0,0,1)
        cr.rectangle(0,0, width, height)
        cr.fill()

    @classmethod
    def on_toggled(cls, widget, data):
        pass

    @classmethod
    def on_destroy(cls, *args):
        Gtk.main_quit()

class App(Gtk.Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                application_id="rt.game.snake",
                **kwargs)

        self.window = None

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
        self.draw.connect("draw", Handler.on_draw)
        #self.draw.set_size_request(40, 40)

        # spin of speed
        speed_adj = Gtk.Adjustment(**{ "value": 8, "lower":1, "upper":40,
            "step_increment":1, "page_increment":10, "page_size":0 })
        self.bt_speed = self.builder.get_object("BT_SPEED")
        self.bt_speed.set_adjustment(speed_adj)


    def do_startup(self):
        """replace wtth super(): loop?"""
        Gtk.Application.do_startup(self)

    def do_activate(self):
        if not self.window:
            self.init_ui()

        self.window.present()

if __name__ == "__main__":
    app = App()
    app.run(sys.argv)
