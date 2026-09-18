"""The evaluation harness: everything that measures the system, and nothing the
served request path may import.

Folds, the transcript cache and the measurement scripts live here. ``medapp``
never reaches into this package — a Conversation arrives once at request time
and there is nothing cached or split about it.
"""
