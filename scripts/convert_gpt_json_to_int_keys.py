#!/usr/bin/env python3
"""
Convert gpt_5.4_... JSON from string class-name keys to integer label keys.

Usage:
    python scripts/convert_gpt_json_to_int_keys.py

Output:
    class_mapping/gpt_5.4_classes_names_descriptions.json
"""

import json
import os
import sys

import numpy as np

CURRENT_DIR = os.path.join(os.path.dirname(__file__), '..', 'class_mapping')
NPY_FILE = os.path.join(CURRENT_DIR, 'imagenet2012_classes.npy')
JSON_IN = os.path.join(
    CURRENT_DIR,
    'gpt_5.4_description_with_image_and_name_class_naming_gpt54_collage_high_detail_latin_vs_common_batch_request.json',
)
JSON_OUT = os.path.join(
    CURRENT_DIR,
    'gpt_5.4_classes_names_descriptions.json',
)


def main():
    try:
        class2name = np.load(NPY_FILE, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: {NPY_FILE} not found.")
        sys.exit(1)

    with open(JSON_IN, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Build reverse: string class name -> int label
    name2int = {v: k for k, v in class2name.items()}

    result = {}
    unmatched = []
    for class_str, entry in raw.items():
        if class_str in name2int:
            result[name2int[class_str]] = entry
        else:
            unmatched.append(class_str)

    # Sort by int key for readability
    result = {int(k): v for k, v in sorted(result.items())}

    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(result)} entries to {JSON_OUT}")
    if unmatched:
        print(f"Warning: {len(unmatched)} entries had no matching int label:")
        for s in unmatched:
            print(f"  {s!r}")


if __name__ == '__main__':
    main()
