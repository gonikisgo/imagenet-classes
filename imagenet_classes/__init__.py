"""
ImageNet Classes Package

A Python package for managing and retrieving ImageNet-1k mappings among integer class IDs, string class IDs, and human-readable class names.
"""

from .class_loader import ClassDictionary

__version__ = "0.2.1"
__author__ = "Illia Volkov, Nikita Kisel"
__email__ = "kiselnik@fel.cvut.cz"

__all__ = [
    "ClassDictionary",
    "get_1k_class_name",
    "get_1k_clean_name",
    "get_1k_gpt_name",
    "get_gpt_class_description",
    "get_gpt_class_guidelines",
    "imagenet21k_to_1k",
    "imagenet1k_to_21k",
    "val_image_to_1k_label",
    "val_image_to_21k_key",
]

_default = ClassDictionary()


def get_1k_class_name(key: int):
    return _default.get_1k_class_name(key)


def get_1k_clean_name(key: int):
    return _default.get_1k_clean_name(key)


def get_1k_gpt_name(key: int):
    return _default.get_1k_gpt_name(key)


def get_gpt_class_description(key: int):
    return _default.get_gpt_class_description(key)


def get_gpt_class_guidelines(key: int):
    return _default.get_gpt_class_guidelines(key)


def imagenet21k_to_1k(key: str):
    return _default.imagenet21k_to_1k(key)


def imagenet1k_to_21k(key: int):
    return _default.imagenet1k_to_21k(key)


def val_image_to_1k_label(key: str):
    return _default.val_image_to_1k_label(key)


def val_image_to_21k_key(key: str):
    return _default.val_image_to_21k_key(key)
