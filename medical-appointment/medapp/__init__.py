"""The medical-appointment system.

One flat package, one module per component of the glossary in ``CONTEXT.md``.
Two seams reach into it: ``pipeline.predict`` speaks the wire protocol, and
``answerer.answer`` is the pure one that takes Segments and returns Verdicts.
"""
