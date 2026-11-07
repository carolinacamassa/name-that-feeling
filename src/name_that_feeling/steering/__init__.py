"""Activation steering: add a direction to the residual stream during a forward pass.

``residual`` holds the forward hook, the position masks, the per-token norm statistics
that fix the unit of a steering strength, and the loader that turns stored emotion
vectors into a steering direction. Everything here is plain torch / numpy and runs
inside whichever Modal container already holds the model (``serving.persona_sampler``),
so importing the package locally needs neither.
"""
