#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 16:29:44 2025

@author: fran
"""

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Div, Spinner, Select, Button, CustomJS, DatePicker, Toggle
import QuantLib as ql
import numpy as np
import datetime
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from bokeh.models import HTMLTemplateFormatter

html_formatter = HTMLTemplateFormatter(template='<div style="text-align:right; font-family: monospace;"><%= value %></div>')



# Inputs ------------------------------------------------------------------------------------------------------------------------
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
use_single_vol_toggle = Toggle(label="Use Single Volatility", button_type="default", active=False)
vol_pre_display = Div(text="", width=300)

# Numeric inputs
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

# Model dropdown (only "cont" or "lin")
model_input = Select(title="Model", options=["cont", "lin"], value="cont")

# Date inputs
start_date_input = DatePicker(title="Start Date", value="2025-02-14", min_date="2020-01-01", max_date="2030-12-31")
dividend_date_input = DatePicker(title="Dividend Date", value="2025-06-05", min_date="2020-01-01", max_date="2030-12-31")
maturity_date_input = DatePicker(title="Maturity Date", value="2025-12-05", min_date="2020-01-01", max_date="2030-12-31")

# A utility function to create an "Info" button that toggles the Div's .visible property
def make_info_button(desc_div):
    button = Button(label="Info", width=50)
    button.js_on_click(CustomJS(args=dict(div=desc_div), code="""div.visible = !div.visible;"""))
    return button

# Create info buttons
button_spot = make_info_button(desc_spot)
button_strike = make_info_button(desc_strike)
button_market_option_price = make_info_button(desc_market_option_price)
button_r = make_info_button(desc_r)
button_div_amount = make_info_button(desc_div_amount)
button_shock = make_info_button(desc_shock)
button_num_stocks = make_info_button(desc_num_stocks)
button_vol_post = make_info_button(desc_vol_post)
button_model = make_info_button(desc_model)
button_start_date = make_info_button(desc_start_date)
button_dividend_date = make_info_button(desc_dividend_date)
button_maturity_date = make_info_button(desc_maturity_date)
button_n_steps = make_info_button(desc_n_steps)
button_timeSteps = make_info_button(desc_timeSteps)

# Lay out each input with its Info button and the (hidden) description below
row_spot = column(row(spot_input, button_spot), desc_spot)
row_strike = column(row(strike_input, button_strike), desc_strike)
row_market_option_price = column(row(market_option_price_input, button_market_option_price), desc_market_option_price)
row_r = column(row(r_input, button_r), desc_r)
row_div_amount = column(row(div_amount_input, button_div_amount), desc_div_amount)
row_shock = column(row(shock_input, button_shock), desc_shock)
row_num_stocks = column(row(num_stocks_input, button_num_stocks), desc_num_stocks)
row_vol_post = column(row(vol_post_input, button_vol_post), desc_vol_post)
row_model = column(row(model_input, button_model), desc_model)
row_start_date = column(row(start_date_input, button_start_date), desc_start_date)
row_dividend_date = column(row(dividend_date_input, button_dividend_date), desc_dividend_date)
row_maturity_date = column(row(maturity_date_input, button_maturity_date), desc_maturity_date)
row_n_steps = column(row(n_steps_input, button_n_steps), desc_n_steps)
row_timeSteps = column(row(timeSteps_input, button_timeSteps), desc_timeSteps)

inputs = column(
    row_spot,
    row_strike,
    row_market_option_price,
    row_r,
    row_div_amount,
    row_shock,
    row_num_stocks,
    row_vol_post,
    row_model,
    row_start_date,
    row_dividend_date,
    row_maturity_date,
    row_n_steps,
    row_timeSteps,
    engine_input
)

#------------------------------------------------------------------------------------------------------------------------------
# Set up ---------------------------------------------------------------------------------------------------------------------
start_button = Button(label="Start", button_type="success", width=100)

def compute_implied_vol(target_price, eval_date, spot, strike, market_option_price, r, div_amount, div_date, maturity_date, timeSteps):
    ql.Settings.instance().evaluationDate = eval_date
    calendar = ql.TARGET()
    day_count = ql.Actual365Fixed()

    adjusted_spot_value = spot - div_amount * np.exp(-r * day_count.yearFraction(eval_date, div_date))
    adjusted_spot = ql.SimpleQuote(adjusted_spot_value)
    
    print(f"The adjusted spot was {adjusted_spot.value()}")

    vol_handle = ql.SimpleQuote(0.2)
    vol_curve = ql.BlackConstantVol(eval_date, calendar, ql.QuoteHandle(vol_handle), day_count)
    bs_process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(adjusted_spot),  
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, r, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.0, day_count)),
        ql.BlackVolTermStructureHandle(vol_curve)
    )

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    exercise = ql.AmericanExercise(eval_date, maturity_date)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.BinomialVanillaEngine(bs_process, engine_input.value, timeSteps))

    try:
        implied_vol = option.impliedVolatility(target_price, bs_process, 1e-6, 100, 0.01, 3.0)
        return implied_vol
    except RuntimeError as e:
        print(f"Implied Volatility Solver Failed: {e}")
        return np.nan

def compute_option_price(vol, eval_date, t, div_date, model, today, spot, shock, r, div_amount, strike, maturity_date, timeSteps):
    ql.Settings.instance().evaluationDate = eval_date
    calendar = ql.TARGET()
    day_count = ql.Actual365Fixed()

    if eval_date < div_date:
        if model == "cont":
            factor_div = 1 / day_count.dayCount(eval_date, div_date)**2
        else:
            factor_div = 1 - day_count.dayCount(eval_date, div_date) / day_count.dayCount(today, div_date)
        factor_shock = 0
    else:
        factor_div = 1
        factor_shock = 1
    
    adjusted_spot_value = (spot - shock * factor_shock) * np.exp(r * t) - (
        div_amount * np.exp(-r * day_count.yearFraction(eval_date, div_date)) * factor_div
    )
    
    adjusted_spot = ql.SimpleQuote(adjusted_spot_value)
    vol_curve = ql.BlackConstantVol(eval_date, calendar, vol, day_count)
    bs_process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(adjusted_spot),
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, r, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.0, day_count)),
        ql.BlackVolTermStructureHandle(vol_curve)
    )

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    exercise = ql.AmericanExercise(today, maturity_date)
    option = ql.VanillaOption(payoff, exercise)
    engine = ql.BinomialVanillaEngine(bs_process, engine_input.value, timeSteps)
    option.setPricingEngine(engine)

    return option.NPV(), adjusted_spot.value()

def compute_post_div_matrix(eval_date, spot, shock, r, div_amount, strike, maturity_date, timeSteps, vol_pre, div_date):
    """
    Computes a matrix of option prices for different implied volatilities (IVs) and underlying prices.
    - Option prices are formatted with fixed-width (8 characters, 2 decimals) so decimals align.
    - The column corresponding to the pre-dividend IV (closest match) is colored blue and bold.
    - The row corresponding to the ex-tag underlying price (S_after, with no change) is colored blue and bold.
    """
    calendar = ql.TARGET()
    day_count = ql.Actual365Fixed()

    # Define the ranges:
    implied_volas = np.arange(0.10, 0.85, 0.05)  # IV from 0.10 to 0.80
    spot_changes = np.arange(-0.20, 0.225, 0.025)  # Variation from -20% to +20%
    
    # Calculate S_after: underlying price on the ex-tag day (if no shock occurs)
    div_time = day_count.yearFraction(eval_date, div_date)
    S_after = spot * np.exp(r * div_time) - div_amount

    # Now compute underlying values based on S_after instead of the initial spot
    underlying_values = [S_after * (1 + change) for change in spot_changes]

    # Build initial data dictionary with fixed-width formatting.
    matrix_data = {}
    matrix_data["IV ↓ / Spot →"] = [f"{S:8.2f}" for S in underlying_values]
    
    # Determine which row corresponds to S_after (i.e. no change, ideally change==0)
    differences = [abs(S - S_after) for S in underlying_values]
    closest_idx = int(np.argmin(differences))
    row_bold = [i == closest_idx for i in range(len(underlying_values))]
    matrix_data["row_bold"] = row_bold  # temporary key for formatting

    # For each implied volatility, compute option prices.
    for vol in implied_volas:
        col_key = f"IV {vol:.2f}"
        prices = []
        for S in underlying_values:
            # Compute the option price at t=0 (post-dividend evaluation)
            option_price, _ = compute_option_price(vol, eval_date, 0, eval_date, "cont", eval_date, S, shock, r, div_amount, strike, maturity_date, timeSteps)
            prices.append(f"{option_price:8.2f}")  # fixed-width format with 2 decimals
        matrix_data[col_key] = prices

    # Bold and color the column corresponding to the pre-dividend IV.
    iv_keys = [key for key in matrix_data if key.startswith("IV") and "↓" not in key]
    pre_vol_str = f"IV {vol_pre:.2f}"
    if pre_vol_str not in matrix_data and iv_keys:
        pre_vol_str = min(iv_keys, key=lambda k: abs(float(k.split()[1]) - vol_pre))
    if pre_vol_str in matrix_data:
        matrix_data[pre_vol_str] = [f"<span style='color:blue; font-weight:bold;'>{val}</span>" for val in matrix_data[pre_vol_str]]

    # Bold and color the row where the underlying equals S_after.
    for key in list(matrix_data.keys()):
        if key == "row_bold":
            continue
        new_col = []
        for i, val in enumerate(matrix_data[key]):
            if matrix_data["row_bold"][i]:
                new_col.append(f"<span style='color:blue; font-weight:bold;'>{val}</span>")
            else:
                new_col.append(val)
        matrix_data[key] = new_col

    # Remove the temporary key.
    matrix_data.pop("row_bold")
    
    return matrix_data


# ColumnDataSource for PostDividend Matrix
post_div_source = ColumnDataSource(data={"IV ↓ / Spot →": []})  # Initially empty

# Create table columns dynamically
post_div_columns = [TableColumn(field="IV ↓ / Spot →", title="IV ↓ / Spot →")]
for vol in np.arange(0.10, 0.85, 0.05):
    post_div_columns.append(TableColumn(field=f"IV {vol:.2f}", title=f"IV {vol:.2f}"))

# Create the DataTable for PostDividend Matrix
post_div_table = DataTable(source=post_div_source, columns=post_div_columns, width=1200, height=400)


def start_calculation():
    # Read parameters from widgets
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
    maturity_date = maturity_date_input.value

    calendar = ql.TARGET()
    day_count = ql.Actual365Fixed()
    today = ql.DateParser.parseISO(start_date)
    div_date = ql.DateParser.parseISO(dividend_date)
    maturity_date = ql.DateParser.parseISO(maturity_date)
    div_time = day_count.yearFraction(today, div_date)
    
    vol_pre = compute_implied_vol(market_option_price, today, spot, strike, market_option_price, r, div_amount, div_date, maturity_date, timeSteps)
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
        
        if use_single_vol_toggle.active:
            current_vol = vol_pre
        else:
            current_vol = vol_pre if t < div_time else vol_post
        
        # Simulate the Underlying Price (deterministic growth)
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
            opt_price, adjusted_spot = compute_option_price(current_vol, d, t, div_date, model, today, spot, shock, r, div_amount, strike, maturity_date, timeSteps)
            adj_spot.append(adjusted_spot)
        option_prices.append(opt_price)
        
        # Compute Immediate Exercise Value and Portfolio Value
        stock_gain = S_t - strike
        stock_gains.append(stock_gain)
        discounted_div = div_amount * np.exp(-r * (div_time - t)) if d < div_date else 0.0
        discounted_dividends.append(discounted_div)
        immediate_ex = stock_gain + discounted_div
        immediate_exercise_values.append(immediate_ex)
        portfolio_value = max(opt_price, immediate_ex) * num_stocks
        portfolio_values.append(portfolio_value)
        
    # Compute the PostDividend Matrix
    post_div_matrix = compute_post_div_matrix(today, spot, shock, r, div_amount, strike, maturity_date, timeSteps, vol_pre, div_date)
    post_div_source.data = post_div_matrix  # Update the table source
    
    post_div_source.data = post_div_matrix

    cols = []
    for key in list(post_div_matrix.keys()):
        cols.append(TableColumn(field=key, title=key, formatter=html_formatter))
    post_div_table.columns = cols


    
    
    py_dates = [datetime.date(d.year(), d.month(), d.dayOfMonth()) for d in dates]   
    py_dates_dt = [datetime.datetime.combine(d, datetime.time.min) for d in py_dates]
    
    # First Plot: Adjusted Spot Price vs Underlying Price
    p = figure(title="Underlying vs Adjusted", x_axis_type="datetime", width=800, height=400)
    p.xaxis.axis_label = "Time"
    p.yaxis.axis_label = "Price"
    p.line(py_dates_dt, adj_spot, line_color="green", line_dash="dashed", line_width=2,
           legend_label="Adjusted for Option Price Calculation")
    p.line(py_dates_dt, underlying_prices, color='black', line_dash="solid", legend_label='Underlying Price')
    
    # Second Plot: Portfolio Values Over Time
    p2 = figure(title="Portfolio Values Over Time", x_axis_type="datetime", width=800, height=400)
    p2.xaxis.axis_label = "Time"
    p2.yaxis.axis_label = "Portfolio Value"
    p2.line(py_dates_dt, portfolio_values, line_color="blue", line_width=2,
            legend_label="Portfolio Value Over Time")
    
    # Third Plot: Option Price vs Immediate Exercise Value Over Time
    p3 = figure(title="Option Price vs Immediate Exercise Value Over Time", x_axis_type="datetime", width=800, height=400)
    p3.xaxis.axis_label = "Time"
    p3.yaxis.axis_label = "Price"
    p3.line(py_dates_dt, option_prices, legend_label="Option Price", color="red")
    p3.line(py_dates_dt, immediate_exercise_values, legend_label="Immediate Exercise Value", color="blue", line_dash="dashed")
      
    # Data Table
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
    source = ColumnDataSource(data=data)
    
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
    data_table = DataTable(source=source, columns=columns, width=1200, height=400)
    
    #new table------------------------------------------------------------------------------------------------
    # Convert simulation dates to Python dates
    py_dates = [datetime.date(d.year(), d.month(), d.dayOfMonth()) for d in dates]

    # Define the 4 key dates:
    calc_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    div_py = datetime.date(div_date.year(), div_date.month(), div_date.dayOfMonth())
    maturity_py = datetime.date(maturity_date.year(), maturity_date.month(), maturity_date.dayOfMonth())
    ex_tag_minus1_qldate = calendar.advance(div_date, ql.Period(-1, ql.Days))
    ex_tag_minus1 = datetime.date(ex_tag_minus1_qldate.year(), ex_tag_minus1_qldate.month(), ex_tag_minus1_qldate.dayOfMonth())

    # Identify indices corresponding to these dates
    indices = [i for i, d in enumerate(py_dates) if d in [calc_date, ex_tag_minus1, div_py, maturity_py]]

    # Build the reduced data dictionary using only these indices
    reduced_data = {key: [vals[i] for i in indices] for key, vals in data.items()}

    # Create a global data source (defaulting to reduced view)
    global table_source
    table_source = ColumnDataSource(data=reduced_data)
    
    # Create a toggle button for switching table view
    global table_toggle_button
    table_toggle_button = Button(label="Show Full Table", button_type="primary", width=200)
    
    def toggle_table():
        if table_toggle_button.label == "Show Full Table":
            table_toggle_button.label = "Show Reduced Table"
            table_source.data = data  # full table data
        
        else:
            table_toggle_button.label = "Show Full Table"
            table_source.data = reduced_data

    table_toggle_button.on_click(toggle_table)

    data_table = DataTable(source=table_source, columns=columns, width=1200, height=400)

    
    # Update the plot container with the three plots and the table
    plots_column = column(p3, p, p2)
    table_column = column(table_toggle_button, data_table, post_div_table)  # Add PostDividend table here
    plot_container.children = [row(plots_column, table_column)]




# Plot container for the plots and table on the right
plot_container = column()

# Controls (inputs and button) on the left
use_single_vol_toggle.js_on_change("active", CustomJS(args=dict(vol_post=vol_post_input), code="vol_post.disabled = cb_obj.active;"))
controls = column(inputs, use_single_vol_toggle, start_button, vol_pre_display)

start_button.on_click(start_calculation)

# Arrange the layout: controls on the left and plots/table on the right
layout = row(controls, plot_container)
curdoc().add_root(layout)
