"""Small helper shared by the example scripts."""

import os
import sys


def add_repo_root_to_path():
    """Put the repo root on sys.path so the clustering package imports.

    The examples live in examples/ but the package sits at the repo root, so
    this lets them run directly (python examples/walkthrough.py) without
    installing anything.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
