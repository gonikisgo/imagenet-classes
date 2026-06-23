#!/usr/bin/env python3
"""
Basic usage example for the imagenet-classes package.
"""

import imagenet_classes as ic
from imagenet_classes import ClassDictionary


def main():
    print("ImageNet Classes Package - Basic Usage Example")
    print("=" * 50)

    # --- Class name methods ---
    print("\n1. Class name retrieval:")

    # Raw synset label (comma-separated synonyms as in the dataset)
    print(f"   get_1k_class_name(0)  → {ic.get_1k_class_name(0)}")
    print(f"   get_1k_class_name(2)  → {ic.get_1k_class_name(2)}")

    # Single curated human-readable name
    print(f"   get_1k_clean_name(0)  → {ic.get_1k_clean_name(0)}")
    print(f"   get_1k_clean_name(2)  → {ic.get_1k_clean_name(2)}")

    # --- GPT descriptions and guidelines ---
    print("\n2. GPT descriptions and guidelines:")
    print(f"   get_gpt_class_description(0) → {ic.get_gpt_class_description(0)}")
    print(f"   get_gpt_class_guidelines(0)  → {ic.get_gpt_class_guidelines(0)}")

    # --- 1k / 21k mappings ---
    print("\n3. 1k ↔ 21k mappings:")
    print(f"   imagenet1k_to_21k(0)              → {ic.imagenet1k_to_21k(0)}")
    print(f"   imagenet21k_to_1k('n01440764')    → {ic.imagenet21k_to_1k('n01440764')}")

    print("\n4. Validation image → 1k label / 21k key:")
    img = 'ILSVRC2012_val_00015416.JPEG'
    print(f"   val_image_to_1k_label('{img}')  → {ic.val_image_to_1k_label(img)}")
    print(f"   val_image_to_21k_key('{img}') → {ic.val_image_to_21k_key(img)}")

    # --- ClassDictionary static utilities ---
    print("\n5. Label ↔ name dict utilities:")
    classes = ['tench', 'goldfish', 'great white shark']
    print(f"   classes = {classes}")
    print(f"   create_label_to_name_dict → {ClassDictionary.create_label_to_name_dict(classes)}")
    print(f"   create_name_to_label_dict → {ClassDictionary.create_name_to_label_dict(classes)}")


if __name__ == "__main__":
    main()
