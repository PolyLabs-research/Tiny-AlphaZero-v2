"""Array-backed MCTS (spec §6).

Deliberately free of submodule imports. The original reason was
``python -m tinyalphazero.mcts.search``: a package import of ``search`` followed
by runpy re-executing it as ``__main__`` gives the module two identities. That
CLI is now ``scripts/run_suite.py`` (REVIEW-2026-10-21 E11), but the rule stays:
importing one submodule should not pull in the others.
"""
