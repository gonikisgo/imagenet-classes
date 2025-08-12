"""
ImageNet Classes Package

A Python package for managing and retrieving ImageNet class names and mappings.
Supports ImageNet-1k.
"""

from .class_loader import ClassDictionary

__version__ = "0.1.0"
__author__ = "Illia Volkov, Nikita Kisel"
__email__ = "kiselnik@fel.cvut.cz"

__all__ = ["ClassDictionary"]
