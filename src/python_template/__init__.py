"""
Project description.
"""

import importlib_metadata

__author__ = "Pierre-Alexandre Dufrene"
__email__ = "pierre-alexandre.dufrene@usherbrooke.ca"
__copyright__ = "Copyright 2026, Pierre-Alexandre Dufrene"
__license__ = "Apache 2.0"
__url__ = "https://github.com/dufp2002/credit-card-fraud"
__package__ = "credit-card-fraud"
__version__ = importlib_metadata.version(__package__)

import warnings

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")
