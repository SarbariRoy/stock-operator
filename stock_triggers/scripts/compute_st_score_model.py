"""Compatibility wrapper for ST logistic model training.

Canonical implementation lives in stock_triggers/scripts/short_term/train_st_logistic_model.py.
"""

from stock_triggers.scripts.short_term.train_st_logistic_model import *  # noqa: F401,F403
from stock_triggers.scripts.short_term.train_st_logistic_model import main


if __name__ == "__main__":
    main()
