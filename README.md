# ITMoption
Calculate Options Price till Maturity

bokeh serve app --show

ITMoption/
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # Bokeh application entry point
│   ├── ui.py                    # All widget and layout creation
│   ├── callbacks.py             # Callback logic (e.g. `start_calculation`)
│   └── logic/
│       ├── __init__.py
│       ├── pricing.py           # Option pricing functions using QuantLib
│       └── utils.py             # Date conversion, transformations, etc.
│
├── tests/
│   ├── __init__.py
│   └── test_pricing.py          # Unit tests for pricing models
│
├── static/                      # (Optional) CSS, JS, or assets
│
├── requirements.txt             # All required packages
├── README.md                    # Project overview and usage
├── LICENSE
└── run.sh                       # Launch script (optional)

