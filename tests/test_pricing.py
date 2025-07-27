#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:51:43 2025

@author: fran
"""

import pytest
import QuantLib as ql
import numpy as np
from scipy.stats import norm

from app.logic.pricing import compute_option_price

@pytest.mark.parametrize(
    "r, vol, timeSteps, engine, model, div_amount, shock, div_offset_days, t_offset_days",
    [
        # Baseline: no dividend, t=0
        (0.05, 0.2, 1000, "CRR", "cont", 0.0, 0.0, 100, 0),
        # Baseline: no dividend, t=180
        (0.05, 0.2, 1000, "CRR", "cont", 0.0, 0.0, 100, 180),
        # Dividend before maturity, t=0
        (0.05, 0.2, 1000, "CRR", "cont", 5.0, 0.0, 100, 0),
        # Dividend before maturity, t=180
        (0.05, 0.2, 1000, "CRR", "cont", 5.0, 0.0, 100, 180),
        # Dividend and shock, t=90
        (0.03, 0.25, 500, "JR", "lin", 2.0, 1.0, 120, 90),
        # Zero rates, t=0
        (0.0,  0.2, 1000, "EQP", "cont", 0.0, 0.0, 50, 0),
        # Zero rates, t=100
        (0.0,  0.2, 1000, "EQP", "cont", 0.0, 0.0, 50, 100),
        # Large dividend, near maturity
        (0.07, 0.15, 2000, "Tian", "lin", 10.0, 0.0, 200, 300),
        # Shock only, t=50
        (0.03, 0.25, 500, "CRR", "cont", 0.0, 5.0, 90, 50),
        # Lin model, t=60
        (0.05, 0.18, 1000, "JR", "lin", 3.0, 2.0, 60, 60),
    ]
)
def test_option_price_binomial_vs_black_scholes(
    r, vol, timeSteps, engine, model, div_amount, shock, div_offset_days, t_offset_days
):
    spot = 100
    strike = 100
    today = ql.Date(1, 1, 2025)
    maturity_date = ql.Date(1, 1, 2026)
    day_count = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = today

    # Calcula la fecha de dividendo (hoy + offset)
    div_date = today + div_offset_days
    if div_date >= maturity_date:
        div_date = today + 1  # El dividendo nunca después del vencimiento

    # Calcula la fecha de evaluación (t offset)
    eval_date = today + t_offset_days
    if eval_date >= maturity_date:
        eval_date = today  # Si t_offset demasiado grande, vuelve a hoy
    t = day_count.yearFraction(today, eval_date)

    price, adj_spot = compute_option_price(
        vol, eval_date, t, div_date, model, today, spot, shock, r,
        div_amount, strike, maturity_date, timeSteps, engine, exercise_type="european"
    )

    # Black-Scholes analytical price solo si no hay dividendos ni shocks y modelo cont
    T = day_count.yearFraction(eval_date, maturity_date)
    if div_amount == 0.0 and shock == 0.0 and model == "cont":
        d1 = (np.log(spot / strike) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        bs_price = spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        print(f"[Analytic] t={t_offset_days}, T={T:.4f}, price={price:.4f}, bs={bs_price:.4f}")
        assert abs(price - bs_price) < 0.5
    else:
        print(f"[Binomial] t={t_offset_days}, price={price:.4f}, model={model}, div={div_amount}, shock={shock}")
        assert price > 0





# def test_implied_vol_simple_case():
#     # Known case: price produced with known vol, implied vol should recover it
#     spot = 100
#     strike = 100
#     r = 0.05
#     div_amount = 0.0
#     shock = 0.0
#     vol = 0.2
#     today = ql.Date(1, 1, 2025)
#     eval_date = today
#     maturity_date = ql.Date(1, 1, 2026)
#     t = 0.0
#     model = "cont"
#     timeSteps = 100
#     engine = "CRR"
#     price, _ = compute_option_price(
#         vol, eval_date, t, eval_date, model, today, spot, shock, r,
#         div_amount, strike, maturity_date, timeSteps, engine
#     )
#     implied_vol = compute_implied_vol(
#         price, eval_date, spot, strike, price, r, div_amount,
#         eval_date, maturity_date, timeSteps, engine
#     )
#     assert abs(implied_vol - vol) < 0.02  # Within 2% error

# def test_option_price_dividends():
#     # Test a dividend scenario: option price should decrease
#     spot = 100
#     strike = 90
#     vol = 0.25
#     r = 0.03
#     div_amount = 5.0
#     shock = 0.0
#     today = ql.Date(1, 1, 2025)
#     eval_date = today
#     maturity_date = ql.Date(1, 7, 2025)
#     div_date = ql.Date(1, 4, 2025)
#     t = 0.0
#     model = "cont"
#     timeSteps = 150
#     engine = "CRR"
#     # With dividend
#     price_with_div, _ = compute_option_price(
#         vol, eval_date, t, div_date, model, today, spot, shock, r,
#         div_amount, strike, maturity_date, timeSteps, engine
#     )
#     # No dividend
#     price_no_div, _ = compute_option_price(
#         vol, eval_date, t, div_date, model, today, spot, shock, r,
#         0.0, strike, maturity_date, timeSteps, engine
#     )
#     assert price_with_div < price_no_div

