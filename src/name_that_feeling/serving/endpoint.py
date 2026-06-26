"""Modal inference endpoint for a trained ``<emotion>``-tag adapter.

STUB -- not implemented yet.

Planned: serve ``allenai/Olmo-3-7B-Instruct`` plus a LoRA adapter loaded from the
``name-that-feeling-checkpoints`` Volume (written by
:func:`name_that_feeling.training.axolotl_sft.train_sft`), exposing a web endpoint
so the local eval scripts / inspect-ai tasks can call the trained model and compare
it against the untouched base. See ``experiments/01-pilot/description.md`` sections 5-6.

Reuse :mod:`name_that_feeling.infra` for the checkpoints Volume, HF cache, and secret;
give this its own ``modal.App`` (e.g. ``"name-that-feeling-serving"``) since a
deployed endpoint has a different lifecycle than an ephemeral training run.
"""
