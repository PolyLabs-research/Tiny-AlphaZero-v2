"""Array-backed MCTS (spec §6).

Deliberately free of submodule imports: ``python -m tinyalphazero.mcts.search``
would otherwise import ``search`` here first and then re-execute it as
``__main__``, which runpy warns about and which would give the module two
distinct identities.
"""
