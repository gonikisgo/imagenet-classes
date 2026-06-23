import json
import os

_GPT_JSON_FILENAME = 'imagenet1k_gpt_names_descriptions.json'
_CLEAN_NAMES_JSON_FILENAME = 'imagenet1k_clean_names.json'


class ClassDictionary:
    def __init__(self, clean_names_json_filename=_CLEAN_NAMES_JSON_FILENAME):
        _dir = os.path.dirname(__file__)
        self._paths = {
            'label_to_names': os.path.join(_dir, 'imagenet1k_label_to_names.json'),
            '21k_to_1k':      os.path.join(_dir, 'imagenet21k_to_1k.json'),
            '1k_to_21k':      os.path.join(_dir, 'imagenet1k_to_21k.json'),
            'val_to_1k':      os.path.join(_dir, 'val_image_to_1k_label.json'),
            'gpt':            os.path.join(_dir, _GPT_JSON_FILENAME),
            'clean_names':    os.path.join(_dir, clean_names_json_filename),
        }
        self._cache = {}

    def _load_json(self, key, int_keys=False):
        if key not in self._cache:
            try:
                with open(self._paths[key], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._cache[key] = {int(k): v for k, v in data.items()} if int_keys else data
            except Exception as e:
                print(f"Error loading '{self._paths[key]}': {e}")
                self._cache[key] = None
        return self._cache[key]

    def get_1k_class_name(self, key):
        """
        Retrieves the original ImageNet-1k class names for a given integer label.

        Args:
            key (int): The ImageNet-1k integer label (0–999).

        Returns:
            list[str] | None: A list of class names if the label exists, None otherwise.
        """
        data = self._load_json('label_to_names', int_keys=True)
        return data.get(key) if data else None

    def get_1k_clean_name(self, key):
        """
        Retrieves the clean curated name for a given ImageNet-1k integer label.

        Falls back to the GPT-generated name if the label is not present in the curated data.

        Args:
            key (int): The ImageNet-1k integer label (0–999).

        Returns:
            str | None: The class name if found, None otherwise.
        """
        clean = self._load_json('clean_names')
        if clean and 0 <= key < len(clean):
            return clean[key]
        return None

    def get_1k_gpt_name(self, key):
        """
        Retrieves the GPT-generated name for a given ImageNet-1k integer label.

        Falls back to get_1k_clean_name, then to get_1k_class_name if the clean
        names file is missing or incorrectly specified.

        Args:
            key (int): The ImageNet-1k integer label (0–999).

        Returns:
            str | None: The GPT name if found, otherwise the best available fallback.
        """
        gpt = self._load_json('gpt')
        if gpt:
            names = gpt.get(str(key), {}).get('name', [])
            if names:
                return ", ".join(names)
        result = self.get_1k_clean_name(key)
        if result is not None:
            return result
        names = self.get_1k_class_name(key)
        return ", ".join(names) if names else None

    def get_gpt_class_description(self, key):
        """
        Retrieves the GPT-generated description for a given int label.

        Args:
            key (int): The ImageNet-1k integer label.

        Returns:
            str | None: The description if found, None otherwise.
        """
        gpt = self._load_json('gpt')
        entry = gpt.get(str(key)) if gpt else None
        return entry.get('description') if entry else "No description available."

    def get_gpt_class_guidelines(self, key):
        """
        Retrieves the GPT-generated classification guidelines for a given int label.

        Args:
            key (int): The ImageNet-1k integer label.

        Returns:
            str | None: The guidelines if found, None otherwise.
        """
        gpt = self._load_json('gpt')
        entry = gpt.get(str(key)) if gpt else None
        return entry.get('guidelines') if entry else "No guidelines available."

    def imagenet21k_to_1k(self, key):
        """
        Maps an ImageNet-21k string key to its corresponding ImageNet-1k integer label.

        Args:
            key (str): The ImageNet-21k class key (e.g. "n01440764").

        Returns:
            int | None: The ImageNet-1k integer label if the key exists, None otherwise.
        """
        data = self._load_json('21k_to_1k')
        return data.get(key) if data else None

    def imagenet1k_to_21k(self, key):
        """
        Maps an ImageNet-1k integer label to its corresponding ImageNet-21k string key.

        Args:
            key (int): The ImageNet-1k integer label (0–999).

        Returns:
            str | None: The ImageNet-21k class key if the label exists, None otherwise.
        """
        data = self._load_json('1k_to_21k', int_keys=True)
        return data.get(key) if data else None

    def val_image_to_1k_label(self, key):
        """
        Maps a validation image filename to its ImageNet-1k integer label.

        Args:
            key (str): The validation split image filename (e.g. "ILSVRC2012_val_00015416.JPEG").

        Returns:
            int | None: The ImageNet-1k integer label if the filename exists, None otherwise.
        """
        data = self._load_json('val_to_1k')
        return data.get(key) if data else None

    def val_image_to_21k_key(self, key):
        """
        Maps a validation image filename to its ImageNet-21k string key.

        Args:
            key (str): The validation split image filename (e.g. "ILSVRC2012_val_00015416.JPEG").

        Returns:
            str | None: The ImageNet-21k class key if the filename exists, None otherwise.
        """
        label_1k = self.val_image_to_1k_label(key)
        return self.imagenet1k_to_21k(label_1k) if label_1k is not None else None

    @staticmethod
    def create_label_to_name_dict(class_names):
        return {idx: cls for idx, cls in enumerate(class_names)}

    @staticmethod
    def create_name_to_label_dict(class_list):
        return {cls: idx for idx, cls in enumerate(class_list)}
