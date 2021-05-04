# -*- coding: utf-8 -*-

"""Top-level package for OpenAI Gym Environments for Donkey Car."""
import os

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()

__author__ = """Tawn Kramer"""
__email__ = "tawnkramer@gmail.com"
__version__ = __version__