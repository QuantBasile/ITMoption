#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:05:04 2025

@author: fran
"""
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, DataTable, TableColumn, Button, HTMLTemplateFormatter
)
import datetime
import numpy as np
import QuantLib as ql
from .ui import (
    spot_input, strike_input, market_option_price_input, r_input, div_amount_input, shock_input,
    num_stocks_input, vol_post_input, model_input, n_steps_input, timeSteps_input, engine_input,
    start_date_input, dividend_date_input, maturity_date_input, use_single_vol_toggle,
    start_button, vol_pre_display, plot_container
)
from .logic.pricing import compute_implied_vol, compute_option_price, compute_post_div_matrix

html_formatter = HTMLTemplateFormatter(template='<div style="text-align:right; font-family: monospace;"><%= value %></div>')

def setup_callbacks(controls, plot_container):
    # These must be global to survive toggle
    table_source = ColumnDataSource(data={})
    table_toggle_button = Button(label="Show Full Table", button_type="primary", width=200)
    post_div_source = ColumnDataSource(data={"IV ↓ / Spot →": []})
    post_div_table = DataTable(source=post_div_source, columns=[], width=1200, height=400)

    def start_calculation():
        spot = spot_input.value
        strike = strike_input.value
        market_option_price = market_option_price_input.value
        r = r_input.value
        div_amount = div_amount_input.value
        shock = shock_input.value
        num_stocks = num_stocks_input.value
        vol_post = vol_post_input.value
        model = model_input.value
        n_steps = n_steps_input.value
        timeSteps = timeSteps_input.value
        start_date = start_date_input.value
        dividend_date = dividend_date_input.value
        maturity_date_val = maturity_date_input.value

        calendar = ql.TARGET()
        day_count = ql.Actual365Fixed()
        today = ql.DateParser.parseISO(start_date)
        div_date = ql.DateParser.parseISO(dividend_date)
        maturity_date = ql.DateParser.parseISO(maturity_date_val)
        div_time = day_count.yearFraction(today, div_date)

        vol_pre = compute_implied_vol(
            market_option_price, today, spot, strike, market_option_price, r, div_amount,
            div_date, maturity_date, timeSteps, engine_input.value
        )
        vol_pre_display.text = f"Initial implied volatility: {vol_pre:.4f}"

        dates = []
        underlying_prices = []
        option_prices = []
        immediate_exercise_values = []
        portfolio_values = []
        adj_spot = []
        discounted_dividends = []
        stock_gains = []

        for i in range(n_steps + 1):
            d = calendar.advance(today, ql.Period(i, ql.Days))
            if d > maturity_date:
                break
            dates.append(d)
            t = day_count.yearFraction(today, d)
            current_vol = vol_pre if (use_single_vol_toggle.active or t < div_time) else vol_post

            if t < div_time:
                S_t = spot * np.exp(r * t)
            else:
                S_t = (spot * np.exp(r * div_time) - div_amount - shock) * np.exp(r * (t - div_time))
            underlying_prices.append(S_t)

            remaining_t = day_count.yearFraction(d, maturity_date)
            if remaining_t <= 1e-8:
                opt_price = max(S_t - strike, 0.0)
                adj_spot.append(spot)
            else:
                opt_price, adjusted_spot_val = compute_option_price(
                    current_vol, d, t, div_date, model, today, spot, shock, r, div_amount,
                    strike, maturity_date, timeSteps, engine_input.value
                )
                adj_spot.append(adjusted_spot_val)
            option_prices.append(opt_price)

            stock_gain = S_t - strike
            stock_gains.append(stock_gain)
            discounted_div = div_amount * np.exp(-r * (div_time - t)) if d < div_date else 0.0
            discounted_dividends.append(discounted_div)
            immediate_ex = stock_gain + discounted_div
            immediate_exercise_values.append(immediate_ex)
            portfolio_value = max(opt_price, immediate_ex) * num_stocks
            portfolio_values.append(portfolio_value)

        py_dates = [datetime.date(d.year(), d.month(), d.dayOfMonth()) for d in dates]
        py_dates_dt = [datetime.datetime.combine(d, datetime.time.min) for d in py_dates]

        # -------- Plots
        from bokeh.plotting import figure
        p = figure(title="Underlying vs Adjusted", x_axis_type="datetime", width=800, height=400)
        p.line(py_dates_dt, adj_spot, line_color="green", line_dash="dashed", line_width=2,
               legend_label="Adjusted for Option Price Calculation")
        p.line(py_dates_dt, underlying_prices, color='black', line_dash="solid", legend_label='Underlying Price')
        p.xaxis.axis_label = "Time"
        p.yaxis.axis_label = "Price"

        p2 = figure(title="Portfolio Values Over Time", x_axis_type="datetime", width=800, height=400)
        p2.line(py_dates_dt, portfolio_values, line_color="blue", line_width=2, legend_label="Portfolio Value Over Time")
        p2.xaxis.axis_label = "Time"
        p2.yaxis.axis_label = "Portfolio Value"

        p3 = figure(title="Option Price vs Immediate Exercise Value Over Time", x_axis_type="datetime", width=800, height=400)
        p3.line(py_dates_dt, option_prices, legend_label="Option Price", color="red")
        p3.line(py_dates_dt, immediate_exercise_values, legend_label="Immediate Exercise Value", color="blue", line_dash="dashed")
        p3.xaxis.axis_label = "Time"
        p3.yaxis.axis_label = "Price"

        # -------- Data Table + Toggle
        time_str = [d.strftime("%Y-%m-%d") for d in py_dates_dt]
        data = {
            "Time": time_str,
            "Underlying Price": [f"{val:.4f}" for val in underlying_prices],
            "Underlying Prices used for Option Price": [f"{val:.4f}" for val in adj_spot],
            "Option Price": [f"{val:.4f}" for val in option_prices],
            "Discounted Dividend Gain": [f"{val:.4f}" for val in discounted_dividends],
            "Stock Gain": [f"{val:.4f}" for val in stock_gains],
            "Exercise Gain": [f"{val:.4f}" for val in immediate_exercise_values],
            "Portfolio Value": [f"{val:.4f}" for val in portfolio_values],
        }

        # Reduced data
        calc_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        div_py = datetime.date(div_date.year(), div_date.month(), div_date.dayOfMonth())
        maturity_py = datetime.date(maturity_date.year(), maturity_date.month(), maturity_date.dayOfMonth())
        ex_tag_minus1_qldate = calendar.advance(div_date, ql.Period(-1, ql.Days))
        ex_tag_minus1 = datetime.date(ex_tag_minus1_qldate.year(), ex_tag_minus1_qldate.month(), ex_tag_minus1_qldate.dayOfMonth())
        indices = [i for i, d in enumerate(py_dates) if d in [calc_date, ex_tag_minus1, div_py, maturity_py]]
        reduced_data = {key: [vals[i] for i in indices] for key, vals in data.items()}

        table_source.data = reduced_data  # Default: reduced
        columns = [
            TableColumn(field="Time", title="Time"),
            TableColumn(field="Underlying Price", title="Underlying Price"),
            TableColumn(field="Underlying Prices used for Option Price", title="Underlying Prices used for Option Price"),
            TableColumn(field="Option Price", title="Option Price"),
            TableColumn(field="Discounted Dividend Gain", title="Discounted Dividend Gain"),
            TableColumn(field="Stock Gain", title="Stock Gain"),
            TableColumn(field="Exercise Gain", title="Exercise Gain"),
            TableColumn(field="Portfolio Value", title="Portfolio Value"),
        ]
        data_table = DataTable(source=table_source, columns=columns, width=1200, height=400)

        def toggle_table():
            if table_toggle_button.label == "Show Full Table":
                table_toggle_button.label = "Show Reduced Table"
                table_source.data = data
            else:
                table_toggle_button.label = "Show Full Table"
                table_source.data = reduced_data

        table_toggle_button.on_click(toggle_table)

        # -------- Post Dividend Matrix Table
        post_div_matrix = compute_post_div_matrix(
            today, spot, shock, r, div_amount, strike, maturity_date, timeSteps, vol_pre, div_date
        )
        post_div_source.data = post_div_matrix
        cols = []
        for key in list(post_div_matrix.keys()):
            cols.append(TableColumn(field=key, title=key, formatter=html_formatter))
        post_div_table.columns = cols

        # -------- Layout
        plots_column = column(p3, p, p2)
        table_column = column(table_toggle_button, data_table, post_div_table)
        plot_container.children = [row(plots_column, table_column)]

    start_button.on_click(start_calculation)
