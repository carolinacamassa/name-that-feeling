"""On-policy SFT data generation.

Two decoupled steps, mirroring the src<->experiments split:

- ``completions`` -- the expensive Modal step: generate the visible assistant reply
  to each user message with the same model + chat template the emotion probe reads
  at, so replies are on-policy. Optionally conditioned on a system prompt (Option 3).
- ``sft`` -- the cheap, re-runnable step: populate the ``<emotion>`` tag from the
  probe projections and render axolotl-ready ``{"messages": [...]}``. Kept separate
  so tag-population strategy (granularity, count, wording, train-time framing) can be
  swept without regenerating a single completion.
"""
