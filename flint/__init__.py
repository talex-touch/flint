"""Flint — an on-device decision model for selection and preference scenarios.

Flint is not a chat model and it does not generate text. Given a piece of
context and a small closed set of options, it returns the option the evidence
picks, together with a confidence it can be held to. It is built for the
decisions an application makes constantly and cannot afford to send anywhere:
which of these candidates the user meant, which of these fields this value
belongs to, which of these actions follows from this state.

This package holds the parts that are ours — how a labelled corpus becomes
decision records, how a mixture is built and audited, what the evaluation
measures, and how a checkpoint is trained and served. The architecture is not
ours and is not reimplemented here; `flint.engine` is the only module that talks
to it.

Only the lightweight modules are imported here. `flint.metrics`,
`flint.mixture` and `flint.scenarios` run on NumPy and the standard library, so
the parts worth reading carefully can be exercised without a GPU stack.
"""

__version__ = "0.1.0"

from . import metrics, mixture, scenarios  # noqa: F401

__all__ = ["metrics", "mixture", "scenarios", "__version__"]
