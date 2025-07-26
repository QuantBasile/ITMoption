## Project Structure

```text
ITMoption/
    app/
        __init__.py
        main.py         # Bokeh app entry point
        ui.py           # Widget/layout creation
        callbacks.py    # Callback logic
        logic/
            __init__.py
            pricing.py  # Option pricing logic
            utils.py    # Date/utility functions
    tests/
        __init__.py
        test_pricing.py # Unit tests
    static/             # (Optional) assets
    requirements.txt    # Dependencies
    README.md           # Project info
    LICENSE
    run.sh              # Launch script (optional)
```
