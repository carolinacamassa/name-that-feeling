"""inspect-ai evaluation tasks for the ``<emotion>``-tag work.

STUB -- not implemented yet.

Planned ``inspect_ai`` ``@task`` definitions, all measured on held-out material and
comparing a trained variant against the untouched base (see
``experiments/01-pilot/description.md`` section 6):

- format compliance (response opens with a single well-formed ``<emotion>`` tag),
- held-out emotion generalization (sensible tag for emotions never trained),
- capability preservation (ordinary-task quality holds, neutral default fires),
- spontaneous expression (emotion leakage into the visible reply vs base).

These target the served endpoint (:mod:`name_that_feeling.serving`).
"""
