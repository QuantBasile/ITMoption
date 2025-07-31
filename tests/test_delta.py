#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 13:08:28 2025

@author: fran
"""

import pytest
import QuantLib as ql
import numpy as np
from app.logic.pricing import compute_option_price

import pytest
import QuantLib as ql
import numpy as np

@pytest.mark.parametrize("eval_offset", range(0, 101, 30))
@pytest.mark.parametrize("maturity_offset", [180, 365])  # days to maturity (6 months, 1 year)
@pytest.mark.parametrize("r", np.arange(0, 0.2, 0.05))
@pytest.mark.parametrize("vol", np.arange(0.1, 0.5, 0.15))
@pytest.mark.parametrize("div_offset_days", [0, 100, 10000])
@pytest.mark.parametrize("option_type", ["put", "call"])
@pytest.mark.parametrize("exercise_type", ["european", "american"])
def test_delta_vs_fd(eval_offset, maturity_offset, r, vol, div_offset_days, option_type, exercise_type):
    spot = 100
    strike = 100
    div_amount=0
    shock=0

    # Dates
    today = ql.Date(1, 8, 2025)
    eval_date = today + eval_offset  # offset in days from today
    div_date = today + div_offset_days
    maturity_date = today + maturity_offset

    # If dividend or eval after maturity, skip (invalid test)
    if eval_date > maturity_date or div_date > maturity_date:
        pytest.skip("Invalid: eval_date/div_date after maturity.")

    t = ql.Actual365Fixed().yearFraction(today, eval_date)
    engine = "CRR"
    timeSteps = 200

    # QuantLib delta
    price, adj_spot, ql_delta = compute_option_price(
        vol, eval_date, t, div_date, today, spot, shock, r, div_amount, strike, 
        maturity_date, timeSteps, engine, 
        exercise_type=exercise_type, option_type=option_type, return_greeks=True
    )

    # Finite diff delta
    eps = 0.01 * spot
    price_up, _, = compute_option_price(
        vol, eval_date, t, div_date, today, spot + eps, shock, r, div_amount, strike,
        maturity_date, timeSteps, engine, exercise_type, option_type, return_greeks=False
    )
    price_down, _, = compute_option_price(
        vol, eval_date, t, div_date, today, spot - eps, shock, r, div_amount, strike,
        maturity_date, timeSteps, engine, exercise_type, option_type, return_greeks=False
    )
    fd_delta = (price_up - price_down) / (2 * eps)

    print(f"\n{option_type}-{exercise_type}, spot={spot}, strike={strike}, r={r:.2f}, vol={vol:.2f}, div={div_amount}, shock={shock}, ttm={maturity_offset}d, eval+{eval_offset}, div+{div_offset_days}")
    print(f"QuantLib Delta:   {ql_delta:.8f}")
    print(f"FD Delta:         {fd_delta:.8f}")
    print(f"Diff:             {ql_delta - fd_delta:.2e}")

    # Accept small numerical difference (binomial, FD, etc)
    assert np.isclose(ql_delta, fd_delta, rtol=0, atol=3-2)
