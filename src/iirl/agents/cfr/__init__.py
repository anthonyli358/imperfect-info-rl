"""Counterfactual Regret Minimization (CFR) for small imperfect-information games."""

from iirl.agents.cfr.cfr import CFRAgent, CFRSolver, regret_matching

__all__ = ["CFRAgent", "CFRSolver", "regret_matching"]
