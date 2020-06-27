#!/usr/bin/env python
# coding: utf8

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

builder = Gtk.Builder()
builder.add_from_file("snake.ui")

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
    def on_destroy(cls, *args):
        Gtk.main_quit()

class App:
    def __init__(self):
        win = builder.get_object("Snake")
        win.connect("destroy", Handler.on_destroy)
        win.show_all()

        draw = builder.get_object("DRAW")
        draw.connect("draw", Handler.on_draw)
        #draw.set_size_request(40, 40)

app = App()
Gtk.main()
