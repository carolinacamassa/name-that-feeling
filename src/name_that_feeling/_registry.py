"""inspect-ai entry-point registrations for this package.

Loaded via ``[project.entry-points.inspect_ai]`` in ``pyproject.toml`` so inspect
discovers our custom providers. Keep this import-light: each ``@modelapi`` callback
imports its heavy implementation lazily, so merely listing the entry point costs nothing
until the ``tinker`` provider is actually used.
"""

from __future__ import annotations

from inspect_ai.model import modelapi


@modelapi(name="tinker")
def tinker():
    from name_that_feeling.serving.tinker_provider import TinkerAPI

    return TinkerAPI
