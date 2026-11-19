"""IFEval (Zhou et al. 2023, arXiv:2311.07911) scoring for stored replies.

IFEval is 541 prompts, each carrying one to three verifiable instructions ("no
commas", "at least 300 words", "end with the exact phrase ..."), checked by code
rather than by a judge. The checks come from the ``instruction_following_eval``
package (josejg's runtime port of the google-research reference implementation,
Apache 2.0), which also ships the 541 prompts, so nothing is downloaded and the
prompt text is the reference file's byte for byte.

Two things this module adds over the package's own ``evaluate_instruction_following``:
it scores one reply at a time and returns the per-instruction booleans, so an
experiment can bootstrap over prompts, break results down by instruction category and
compare models prompt by prompt; and it keeps the reference's two regimes side by side:

- ``strict``: the reply as generated must satisfy every instruction;
- ``loose``: the reference's leniency, the reply also passes if it satisfies the
  instructions after stripping markdown asterisks and/or the first or last line
  (a "Sure, here is ..." preamble or a trailing remark).

The four headline numbers of the reference are the prompt-level accuracy (share of
prompts where every instruction is followed) and the instruction-level accuracy
(share of instructions followed), each in both regimes.
"""

from __future__ import annotations

from instruction_following_eval import get_examples
from instruction_following_eval.evaluation import ensure_nltk_resource, test_instruction_following

REGIMES = ("strict", "loose")


def load_prompts() -> list[dict]:
    """The 541 reference prompts in file order: ``{"key", "prompt", "instruction_id_list", "kwargs"}``."""
    return get_examples()


def instruction_category(instruction_id: str) -> str:
    """``"punctuation:no_comma" -> "punctuation"``: the reference's category prefix."""
    return instruction_id.split(":", 1)[0]


def score_reply(example: dict, reply: str) -> dict[str, list[bool]]:
    """Per-instruction pass booleans for one reply, ``{"strict": [...], "loose": [...]}``,
    in the order of ``example["instruction_id_list"]``. An empty reply fails everything."""
    ensure_nltk_resource()
    return {
        regime: list(test_instruction_following(example, reply, strict=(regime == "strict")).follow_instruction_list)
        for regime in REGIMES
    }
