#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:05:54 2025

@author: fran
"""

import QuantLib as ql
import numpy as np

def compute_implied_vol(target_price, eval_date, spot, strike, market_option_price, r, 
                        div_amount, div_date, maturity_date, timeSteps, engine, 
                        exercise_type="american",option_type="call"):
    ql.Settings.instance().evaluationDate = eval_date
    calendar = ql.TARGET()
    day_count = ql.Actual365Fixed()
    #adjusted_spot_value = spot - div_amount * np.exp(-r * day_count.yearFraction(eval_date, div_date))
    #adjusted_spot = ql.SimpleQuote(adjusted_spot_value)
    adjusted_spot = ql.SimpleQuote(spot)
    vol_handle = ql.SimpleQuote(0.3)
    vol_curve = ql.BlackConstantVol(eval_date, calendar, ql.QuoteHandle(vol_handle), day_count)
    bs_process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(adjusted_spot),
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.0, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, r, day_count)),
        ql.BlackVolTermStructureHandle(vol_curve)
    )
    if option_type == "call":
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    elif option_type == "put":
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, strike)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    if exercise_type == "american":
        exercise = ql.AmericanExercise(eval_date, maturity_date)
    elif exercise_type == "european":
        exercise = ql.EuropeanExercise(maturity_date)
    else:
        raise ValueError("exercise_type must be 'american' or 'european'")
    
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(getattr(ql, 'BinomialVanillaEngine')(bs_process, engine, timeSteps))
    try:
        implied_vol = option.impliedVolatility(target_price, bs_process, 1e-6, 100, 0.01, 3.0)
        return implied_vol
    except RuntimeError:
        return np.nan

def compute_option_price(vol, eval_date, t, div_date, today, spot, shock, r, div_amount, strike, 
                         maturity_date, timeSteps, engine, exercise_type="american",option_type="call"):
    
    if eval_date > maturity_date:
        raise ValueError("Evaluation date (eval_date) cannot be after option maturity.")
    if div_date > maturity_date:
        raise ValueError("Dividend date (div_date) cannot be after option maturity.")
    
    ql.Settings.instance().evaluationDate = eval_date
    calendar = ql.TARGET()
    day_count = ql.Actual365Fixed()
    if eval_date < div_date:
        if 1 == 1:
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
    #print(vol)
    bs_process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(adjusted_spot),
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.0, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, r, day_count)),       #rate free risk
        ql.BlackVolTermStructureHandle(vol_curve)
    )
    
    if option_type == "call":
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    elif option_type == "put":
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, strike)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    if exercise_type == "american":
        exercise = ql.AmericanExercise(today, maturity_date)
    elif exercise_type == "european":
        exercise = ql.EuropeanExercise(maturity_date)
    else:
        raise ValueError("Unknown exercise_type")
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(getattr(ql, 'BinomialVanillaEngine')(bs_process, engine, timeSteps))
    return option.NPV(), adjusted_spot.value()

def compute_post_div_matrix(eval_date, spot, shock, r, div_amount, strike, maturity_date, timeSteps, 
                            vol_pre, div_date):
    calendar = ql.TARGET()
    day_count = ql.Actual365Fixed()
    implied_volas = np.arange(0.10, 0.85, 0.05)
    spot_changes = np.arange(-0.20, 0.225, 0.025)
    div_time = day_count.yearFraction(eval_date, div_date)
    S_after = spot * np.exp(r * div_time) - div_amount
    underlying_values = [S_after * (1 + change) for change in spot_changes]
    matrix_data = {}
    matrix_data["IV ↓ / Spot →"] = [f"{S:8.2f}" for S in underlying_values]
    differences = [abs(S - S_after) for S in underlying_values]
    closest_idx = int(np.argmin(differences))
    row_bold = [i == closest_idx for i in range(len(underlying_values))]
    matrix_data["row_bold"] = row_bold
    for vol in implied_volas:
        col_key = f"IV {vol:.2f}"
        prices = []
        for S in underlying_values:
            option_price, _ = compute_option_price(
                vol, eval_date, 0, eval_date, eval_date, S, shock, r, div_amount,
                strike, maturity_date, timeSteps, "CRR"
            )
            prices.append(f"{option_price:8.2f}")
        matrix_data[col_key] = prices
    iv_keys = [key for key in matrix_data if key.startswith("IV") and "↓" not in key]
    pre_vol_str = f"IV {vol_pre:.2f}"
    if pre_vol_str not in matrix_data and iv_keys:
        pre_vol_str = min(iv_keys, key=lambda k: abs(float(k.split()[1]) - vol_pre))
    if pre_vol_str in matrix_data:
        matrix_data[pre_vol_str] = [f"<span style='color:blue; font-weight:bold;'>{val}</span>" for val in matrix_data[pre_vol_str]]
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
    matrix_data.pop("row_bold")
    return matrix_data
