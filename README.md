# imagenet-classes

A Python package for managing and retrieving ImageNet-1k (ImageNet2012) class names and 1k/21k mappings.

> **v0.2.0 — breaking rename:** The package was previously distributed as `class_mapping` with imports from `class_mapping`. Starting in 0.2.0 the package is renamed to `imagenet_classes`. All imports must be updated accordingly (see [Migration from v0.1.x](#migration-from-v01x)).

## Installation

```bash
pip install imagenet-classes
```

Or from source:

```bash
git clone https://github.com/gonikisgo/imagenet-classes.git
cd imagenet-classes
pip install -e .
```

## Quick Start

```python
import imagenet_classes as ic

# Get the raw synset label string for class 388
print(ic.get_1k_class_name(388))
# → 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca'

# Get a single curated human-readable name for class 388
print(ic.get_1k_clean_name(388))
# → 'giant panda'

# 1k/21k key conversions
print(ic.imagenet1k_to_21k(388))             # → 'n02510455'
print(ic.imagenet21k_to_1k('n02510455'))     # → 388
```

## Class name methods

There are two methods for retrieving human-readable names. They differ in the source data and the level of curation:

### `get_1k_class_name(key: int) -> str | None`

Returns the original ImageNet-1k label string exactly as it appears in the dataset annotation — a comma-separated list of WordNet synonyms for the synset. Many entries contain five or more synonyms and a scientific name.

```python
ic.get_1k_class_name(0)   # → 'tench, Tinca tinca'
ic.get_1k_class_name(2)   # → 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias'
```

Use this when you need the canonical ImageNet label as-is (e.g. for reproducibility with prior work that used the original synset strings).

### `get_1k_clean_name(key: int) -> str | None`

Returns a single, manually curated human-readable name — one short English phrase per class, with no synonyms or scientific names.

```python
ic.get_1k_clean_name(0)   # → 'tench'
ic.get_1k_clean_name(2)   # → 'great white shark'
```

Use this when displaying class names in a UI or feeding them to a model as plain text, where a clean, unambiguous single name is preferable.

## GPT descriptions and guidelines

Each class has a GPT-generated textual description and classification guideline based on the class content.

```python
ic.get_gpt_class_description(0)
# → 'A thick-bodied freshwater fish with a smooth-looking body covered in very
#    small scales, usually colored olive-green, bronze, or brownish gold...'

ic.get_gpt_class_guidelines(0)
# → 'Classify as tench when the fish is deep-bodied and olive or bronze colored,
#    with tiny scales giving a slick matte appearance...'
```

## 1k/21k mappings

### ImageNet-1k ↔ ImageNet-21k

Each ImageNet-1k integer label (0–999) corresponds to a WordNet synset key (e.g. `'n01440764'`) that is also used in ImageNet-21k.

```python
ic.imagenet1k_to_21k(0)              # → 'n01440764'
ic.imagenet21k_to_1k('n01440764')    # → 0
```

### Validation image → ImageNet-1k label / ImageNet-21k key

Maps a validation split image filename to its ImageNet-1k integer label or ImageNet-21k string key.

```python
ic.val_image_to_1k_label('ILSVRC2012_val_00015416.JPEG')   # → 13
ic.val_image_to_21k_key('ILSVRC2012_val_00015416.JPEG')    # → 'n01534433'
```

## Using `ClassDictionary` directly

All module-level functions above delegate to a shared `ClassDictionary` instance. You can also instantiate `ClassDictionary` directly, which is useful when you want to swap in a different curated names file:

```python
from imagenet_classes import ClassDictionary

cd = ClassDictionary(clean_names_json_filename='my_clean_names.json')
print(cd.get_1k_clean_name(0))
```

`ClassDictionary` also provides two static utilities for building label ↔ name dictionaries from an arbitrary class list:

```python
classes = ['tench', 'goldfish', 'great white shark']

ClassDictionary.create_label_to_name_dict(classes)
# → {0: 'tench', 1: 'goldfish', 2: 'great white shark'}

ClassDictionary.create_name_to_label_dict(classes)
# → {'tench': 0, 'goldfish': 1, 'great white shark': 2}
```

## Migration from v0.1.x

| Before (v0.1.x) | After (v0.2.0+) |
|---|---|
| `from class_mapping import ClassDictionary` | `from imagenet_classes import ClassDictionary` or `import imagenet_classes as ic` |
| `class_dict.get_class_name(key)` | `ic.get_1k_class_name(key)` |
| `class_dict.get_custom_class_name(key)` | `ic.get_1k_clean_name(key)` |
| `class_dict.get_class_1k(key)` | `ic.imagenet21k_to_1k(key)` |

If you previously instantiated `ClassDictionary()` yourself, you can drop it and use the module-level functions (`ic.get_1k_class_name`, `ic.get_1k_clean_name`, etc.) — they delegate to a shared instance internally. The only reason to keep using `ClassDictionary` directly is if you need to supply a custom clean names file:

```python
from imagenet_classes import ClassDictionary

cd = ClassDictionary(clean_names_json_filename='/path/to/my_clean_names.json')
cd.get_1k_clean_name(0)   # uses your file instead of the bundled one
```

The file must be a plain JSON list of 1000 strings — no keys, no mapping, just names in label order (index 0 = label 0, index 1 = label 1, …):

```json
["tench", "goldfish", "great white shark", ...]
```

## License

MIT — see [LICENSE](LICENSE).
