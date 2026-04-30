"""Compatibility wrapper for the LT signal generator.

Canonical implementation lives in stock_triggers/scripts/long_term/generate_lt_signals.py.
"""

from stock_triggers.scripts.long_term.generate_lt_signals import *  # noqa: F401,F403
from stock_triggers.scripts.long_term.generate_lt_signals import main


if __name__ == "__main__":
    main()
