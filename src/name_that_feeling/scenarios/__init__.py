"""Scenario generation for the emotion-tag training data.

Current stage: **per-emotion candidate triage** (experiment 00). For each emotion
in the taxonomy we ask an open model whether that emotion can genuinely be the
*assistant's own* reaction to something a user might say -- and, when it can, to
sketch a few concrete situations where it would. Emotions that only ever make
sense as the *user's* state (or have no analogue for a text-only assistant) are
skipped, with a reason. The output is the OK/not-OK + scenario artifact that
re-grounds the borrowed (reader-of-stories) taxonomy in the assistant frame.

The logic lives in ``candidates.py``; it calls an open model through the HF
Inference router (``name_that_feeling.hf_router``) and runs locally.
"""
