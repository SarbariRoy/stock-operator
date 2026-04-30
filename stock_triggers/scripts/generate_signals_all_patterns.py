"""Compatibility wrapper for the ST signal generator.

Canonical implementation lives in stock_triggers/scripts/short_term/generate_st_signals.py.
"""

from stock_triggers.scripts.short_term.generate_st_signals import *  # noqa: F401,F403
from stock_triggers.scripts.short_term.generate_st_signals import main


if __name__ == "__main__":
    main()
