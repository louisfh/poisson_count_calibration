"""Families of estimators beyond the original set, one module per family.

Each module registers its methods with countcal.methods.register when imported.
"""

from . import calibrated  # noqa: F401
from . import intervals  # noqa: F401
from . import designs_learned  # noqa: F401
