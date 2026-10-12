"""Config-driven evaluation harness for the G1 locomotion stack.

Baselines, tasks, perturbations and metrics are registries. A new controller
(RL, VLA) or a new disturbance (an obstacle, once vision exists) is a class
plus a YAML entry.
"""
