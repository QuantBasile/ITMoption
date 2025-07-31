#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 11:12:05 2025

@author: fran
"""

import subprocess
import time
import requests

def test_bokeh_app_launch():
    # Lanza el servidor Bokeh en un puerto no estándar para test
    proc = subprocess.Popen(
        ["bokeh", "serve", "app/main.py", "--port", "5007", "--allow-websocket-origin=localhost:5007"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Espera unos segundos a que arranque
        time.sleep(3)
        # Hace un request a la página principal
        resp = requests.get("http://localhost:5007/main")
        # Si responde 200, ¡todo bien!
        assert resp.status_code == 200
    finally:
        proc.terminate()
        proc.wait()
