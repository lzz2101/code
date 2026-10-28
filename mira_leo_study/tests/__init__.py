"""Tests for the synthetic MIRA-LEO study."""
"""Simulation tests are opt-in so code verification never runs a simulator."""

import os
import unittest


simulation_test = unittest.skipUnless(
    os.environ.get("MIRA_LEO_RUN_SIMULATION_TESTS") == "1",
    "simulation disabled; set MIRA_LEO_RUN_SIMULATION_TESTS=1 to opt in",
)
