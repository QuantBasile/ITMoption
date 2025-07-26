#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:06:16 2025

@author: fran
"""
import QuantLib as ql

def to_ql_date(date_str):
    return ql.DateParser.parseISO(date_str)

