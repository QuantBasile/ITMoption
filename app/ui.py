#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:04:29 2025

@author: fran
"""

from bokeh.layouts import column, row
from bokeh.models import (
    Div, Spinner, Select, Button, CustomJS, DatePicker, Toggle,
    ColumnDataSource, DataTable, TableColumn, HTMLTemplateFormatter
)
import numpy as np

# Descriptions
desc_spot = Div(text="The current price of the underlying asset", width=300, visible=False)
desc_strike = Div(text="The option's strike price", width=300, visible=False)
desc_market_option_price = Div(text="The observed market price of the option controlling 1 stock", width=300, visible=False)
desc_r = Div(text="The risk-free interest rate in % (5%=0.05)", width=300, visible=False)
desc_div_amount = Div(text="The absolute dividend amount (if any)", width=300, visible=False)
desc_shock = Div(text="A value representing market shock", width=300, visible=False)
desc_num_stocks = Div(text="The number of stocks for the calculation", width=300, visible=False)
desc_vol_post = Div(text="The volatility after dividends (if you want to change it)", width=300, visible=False)
desc_model = Div(text="Choose 'cont' for continuous or 'lin' for linear model.", width=300, visible=False)
desc_start_date = Div(text="The date from which the calculation begins", width=300, visible=False)
desc_dividend_date = Div(text="The date on which the dividend is paid (if applicable)", width=300, visible=False)
desc_maturity_date = Div(text="The expiration date of the option or contract", width=300, visible=False)
desc_n_steps = Div(text="Roughly daily steps until maturity", width=300, visible=False)
desc_timeSteps = Div(text="Base number of steps in the binomial tree (CRR solver)", width=300, visible=False)

# Inputs
spot_input = Spinner(title="Spot", low=0, high=5000, step=0.01, value=100.0)
strike_input = Spinner(title="Strike", low=0, high=5000, step=0.1, value=90.0)
market_option_price_input = Spinner(title="Market Option Price", low=0, high=10000, step=0.1, value=12.0)
r_input = Spinner(title="Risk-Free Interest Rate (r)", low=-1, high=1, step=0.001, value=0.05)
div_amount_input = Spinner(title="Dividend Amount", low=0, high=1000, step=0.01, value=3.0)
shock_input = Spinner(title="Shock", low=0, high=1000, step=0.1, value=0.0)
num_stocks_input = Spinner(title="Number of Stocks", low=1, high=10000, step=1, value=1800)
vol_post_input = Spinner(title="Volatility Post Dividends", low=0.10, high=1, step=0.01, value=0.29646960452469723)
timeSteps_input = Spinner(title="timeSteps", low=1, high=10000, step=1, value=200)
n_steps_input = Spinner(title="n_steps", low=1, high=10000, step=1, value=380)
engine_input = Select(title="Binomial Engine", options=["CRR", "JR", "Tian", "EQP"], value="CRR")
model_input = Select(title="Model", options=["cont", "lin"], value="cont")
start_date_input = DatePicker(title="Start Date", value="2025-02-14", min_date="2020-01-01", max_date="2030-12-31")
dividend_date_input = DatePicker(title="Dividend Date", value="2025-06-05", min_date="2020-01-01", max_date="2030-12-31")
maturity_date_input = DatePicker(title="Maturity Date", value="2025-12-05", min_date="2020-01-01", max_date="2030-12-31")
use_single_vol_toggle = Toggle(label="Use Single Volatility", button_type="default", active=False)
start_button = Button(label="Start", button_type="success", width=100)
vol_pre_display = Div(text="", width=300)

def make_info_button(desc_div):
    button = Button(label="Info", width=50)
    button.js_on_click(CustomJS(args=dict(div=desc_div), code="""div.visible = !div.visible;"""))
    return button

row_spot = column(row(spot_input, make_info_button(desc_spot)), desc_spot)
row_strike = column(row(strike_input, make_info_button(desc_strike)), desc_strike)
row_market_option_price = column(row(market_option_price_input, make_info_button(desc_market_option_price)), desc_market_option_price)
row_r = column(row(r_input, make_info_button(desc_r)), desc_r)
row_div_amount = column(row(div_amount_input, make_info_button(desc_div_amount)), desc_div_amount)
row_shock = column(row(shock_input, make_info_button(desc_shock)), desc_shock)
row_num_stocks = column(row(num_stocks_input, make_info_button(desc_num_stocks)), desc_num_stocks)
row_vol_post = column(row(vol_post_input, make_info_button(desc_vol_post)), desc_vol_post)
row_model = column(row(model_input, make_info_button(desc_model)), desc_model)
row_start_date = column(row(start_date_input, make_info_button(desc_start_date)), desc_start_date)
row_dividend_date = column(row(dividend_date_input, make_info_button(desc_dividend_date)), desc_dividend_date)
row_maturity_date = column(row(maturity_date_input, make_info_button(desc_maturity_date)), desc_maturity_date)
row_n_steps = column(row(n_steps_input, make_info_button(desc_n_steps)), desc_n_steps)
row_timeSteps = column(row(timeSteps_input, make_info_button(desc_timeSteps)), desc_timeSteps)

inputs = column(
    row_spot, row_strike, row_market_option_price, row_r, row_div_amount, row_shock,
    row_num_stocks, row_vol_post, row_model, row_start_date, row_dividend_date,
    row_maturity_date, row_n_steps, row_timeSteps, engine_input
)

controls = column(inputs, use_single_vol_toggle, start_button, vol_pre_display)
plot_container = column()

# Export ALL widgets for callbacks.py
__all__ = [k for k in locals() if not k.startswith("_")]
