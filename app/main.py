#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:03:29 2025

@author: fran
"""
from bokeh.io import curdoc
from bokeh.layouts import row
from .ui import controls, plot_container
from .callbacks import setup_callbacks

setup_callbacks(controls, plot_container)
layout = row(controls, plot_container)
curdoc().add_root(layout)

