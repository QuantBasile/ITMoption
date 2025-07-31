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
from app.logic.pricing import compute_implied_vol

today = ql.Date(1, 1, 2025)
maturity_date = ql.Date(1, 1, 2026)
day_count = ql.Actual365Fixed()

@pytest.mark.parametrize("eval_offset", range(0, 101, 30))
@pytest.mark.parametrize("r", np.arange(0, 0.2, 0.05))
@pytest.mark.parametrize("vol", np.arange(0.1, 0.5, 0.15))
@pytest.mark.parametrize("div_amount", [0,  5])
@pytest.mark.parametrize("shock", [0, 10])
@pytest.mark.parametrize("div_offset_days", [0, 100,10000])
@pytest.mark.parametrize("option_type", ["put","call"])
def test_compute_option_price(eval_offset, r, vol,  div_amount, shock, div_offset_days,option_type):
    timeSteps=1000
    engine="CRR"
    spot = 100
    strike = 100
    eval_date = today + eval_offset
    div_date = today + div_offset_days
    t = day_count.yearFraction(today, eval_date) #desde today a eval date
    T = day_count.yearFraction(eval_date, maturity_date) #desde eval date a expiration

    if eval_date > maturity_date or div_date > maturity_date:
        with pytest.raises(ValueError):
            compute_option_price(
                vol, eval_date, day_count.yearFraction(today, eval_date), div_date, today, spot, shock, r,
                div_amount, strike, maturity_date, timeSteps, engine, exercise_type="european", option_type=option_type
            )
        return
    
    else:
        price, adj_spot = compute_option_price(
            vol, eval_date, t, div_date, today, spot, shock, r,
            div_amount, strike, maturity_date, timeSteps, engine, exercise_type="european", option_type=option_type
        )

    # Black-Scholes analytical price solo si no hay dividendos ni shocks
    if div_amount == 0.0 and shock == 0.0:
        d1 = (np.log(spot * np.exp(r * t) / strike) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        
        if option_type == "call":
            bs_price = spot * np.exp(r * t) * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            bs_price = strike * np.exp(-r * T) * norm.cdf(-d2) - spot * np.exp(r * t) * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        #bs_price = (spot * np.exp(r * t)) * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        print(f"[Analytic] eval_offset={eval_offset}, price={price:.4f}, bs={bs_price:.4f}")
        assert abs(price - bs_price) < 0.5
    else:
        print(f"[Binomial] eval_offset={eval_offset}, price={price:.4f}, div={div_amount}, shock={shock}")
        assert price > 0


@pytest.mark.parametrize("option_type", ["call","put"])
@pytest.mark.parametrize("spot", [80, 100, 120])
@pytest.mark.parametrize("strike", [80, 100, 120])
@pytest.mark.parametrize("r", [0.0, 0.03, 0.05,0.1])
@pytest.mark.parametrize("vol", [0.15, 0.2, 0.3,0.4,0.5])
@pytest.mark.parametrize("div_amount", [0.0])
@pytest.mark.parametrize("shock", [0.0])
@pytest.mark.parametrize("eval_offset_days", [0, 50])
def test_implied_vol_parametric(option_type, spot, strike, r, vol, div_amount, shock,  eval_offset_days):
    moneyness = spot / strike
    deep_ITM_OTM = False
    if option_type == "call" and (moneyness > 1.15 or moneyness < 0.85):
        deep_ITM_OTM = True
    elif option_type == "put" and (moneyness < 0.85 or moneyness > 1.15):
        deep_ITM_OTM = True
    if deep_ITM_OTM:
        pytest.skip(f"Option is deep ITM/OTM (moneyness={moneyness:.2f}), implied vol not robust.")
        return
    
    today = ql.Date(1, 1, 2025)
    eval_date = today + eval_offset_days
    maturity_date = ql.Date(1, 1, 2026)
    div_date = today + 100  # For simplicity, fixed div date at +100d (make sure < maturity!)
    if div_date > maturity_date:
        div_date = today + 1
    t = day_count.yearFraction(today, eval_date)
    timeSteps = 1000
    engine = "CRR"

    # Only test for cases where eval_date < maturity_date
    if eval_date > maturity_date or div_date > maturity_date:
        pytest.skip("Invalid test case: eval_date or div_date after maturity")
        

    # 1. Generate a theoretical option price with known vol
    price, _ = compute_option_price(
        vol, eval_date, t, div_date, today, spot, shock, r,
        div_amount, strike, maturity_date, timeSteps, engine,
        exercise_type="european", option_type=option_type
    )
    if price < 0.1:
        pytest.skip(f"Option price too small ({price:.4f}), implied vol not meaningful.")
        return

    # 2. Try to recover implied volatility from that price
    implied_vol = compute_implied_vol(
        price, eval_date, spot, strike, price, r, div_amount,
        div_date, maturity_date, timeSteps, engine, exercise_type="european",option_type=option_type
    )

    print(f"type={option_type}, spot={spot}, strike={strike}, r={r}, vol={vol}, div_amount={div_amount}, shock={shock},  eval_offset={eval_offset_days} | price={price:.4f} | implied_vol={implied_vol:.4f}")

    assert abs(implied_vol - vol) < 0.06  # Within 2% error


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("div_amount", [1, 5, 10])
@pytest.mark.parametrize("spot", [80, 100, 120])
@pytest.mark.parametrize("strike", [80, 100, 120])
@pytest.mark.parametrize("r", [0.01, 0.05])
@pytest.mark.parametrize("vol", [0.15, 0.2, 0.3])
@pytest.mark.parametrize("shock", [0.0, 0])
@pytest.mark.parametrize("eval_offset_days", [0, 50])
@pytest.mark.parametrize("div_offset_days", [30, 100, 200])
def test_option_price_dividends(
    option_type, div_amount, spot, strike, r, vol, shock, eval_offset_days, div_offset_days
):
    today = ql.Date(1, 1, 2025)
    eval_date = today + eval_offset_days
    maturity_date = ql.Date(1, 1, 2026)
    div_date = today + div_offset_days
    t = day_count.yearFraction(today, eval_date)
    timeSteps = 1000
    engine = "CRR"

    # Saltar fechas inválidas
    if eval_date > maturity_date or div_date > maturity_date:
        pytest.skip("Invalid test case: eval_date or div_date after maturity")
        return

    # Deep ITM/OTM moneyness
    moneyness = spot / strike
    if option_type == "call" and (moneyness > 1.15 or moneyness < 0.85):
        pytest.skip(f"Call deep ITM/OTM (moneyness={moneyness:.2f})")
        return
    if option_type == "put" and (moneyness < 0.85 or moneyness > 1.15):
        pytest.skip(f"Put deep ITM/OTM (moneyness={moneyness:.2f})")
        return

    # Precios con y sin dividendo
    price_no_div, _ = compute_option_price(
        vol, eval_date, t, div_date, today, spot, shock, r,
        0.0, strike, maturity_date, timeSteps, engine,
        exercise_type="european", option_type=option_type
    )
    price_with_div, _ = compute_option_price(
        vol, eval_date, t, div_date, today, spot, shock, r,
        div_amount, strike, maturity_date, timeSteps, engine,
        exercise_type="european", option_type=option_type
    )

    # Salta precios bajos (opción sin valor o numéricamente inestable)
    min_price_threshold = 0.1
    if price_no_div < min_price_threshold or price_with_div < min_price_threshold:
        pytest.skip(f"Option price too small (no_div={price_no_div:.4f}, with_div={price_with_div:.4f})")
        return

    print(f"type={option_type}, spot={spot}, strike={strike}, r={r}, vol={vol}, "
          f"div_amount={div_amount}, shock={shock}, eval_offset={eval_offset_days}, "
          f"div_offset={div_offset_days} | price_no_div={price_no_div:.4f}, price_with_div={price_with_div:.4f}")

    # Test robusto:
    if option_type == "call":
        assert price_with_div < price_no_div - 1e-6, (
            f"Call: Price did not decrease with dividend. "
            f"no_div={price_no_div:.4f}, with_div={price_with_div:.4f}"
        )
    elif option_type == "put":
        # Permite igual o subir (subida puede ser mínima en el borde)
        assert price_with_div > price_no_div - 1e-6, (
            f"Put: Price did not increase (or remain) with dividend. "
            f"no_div={price_no_div:.4f}, with_div={price_with_div:.4f}"
        )

