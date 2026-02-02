import copy
import logging
import sys
from ast import literal_eval
from typing import List
from typing import Union, IO, Dict, Any

import yaml
from ml_collections import ConfigDict

from src.utils import jload

_VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}

# py2 allow for str and unicode
_PY2 = sys.version_info.major == 2
if _PY2:
    _VALID_TYPES = _VALID_TYPES.union({unicode})  # noqa: F821


class CfgNode(ConfigDict):
    """compact yacs config node api"""

    @classmethod
    def _open_cfg(cls, filename: str) -> Union[IO[str], IO[bytes]]:
        """
        Defines how a config file is opened. May be overridden to support
        different file schemas.
        """
        return open(filename, "r")

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def freeze(self):
        return self.lock()

    def defrost(self):
        return self.unlock()

    def clone(self):
        return copy.deepcopy(self)

    def merge_from_other_cfg(self, cfg_other: "CfgNode"):
        return self.update(cfg_other)

    def update(self, cfg_other):
        if isinstance(cfg_other, dict):
            cfg_other = CfgNode(cfg_other)
        super().update(cfg_other)

    def merge_from_list(self, cfg_list: List[str]):
        """Merge config (keys, values) in a list (e.g., from command line) into
                        this CfgNode. For example, `cfg_list = ['FOO.BAR', 0.5]`.
        """
        if len(cfg_list) == 0:
            return

        assert len(cfg_list) % 2 == 0, f"Override list has odd length: {cfg_list}; it must be a list of pairs"
        keys = cfg_list[0::2]
        values = cfg_list[1::2]
        cfg_dict = {}
        for k, v in zip(keys, values):
            try:
                v = literal_eval(v)
            except ValueError:
                pass
            except SyntaxError:
                pass

            if v in ("true", "True", "TRUE"):
                v = True

            if v in ("false", "False", "FALSE"):
                v = False

            if v in ("null", "Null", "NULL", "none", "None", "NONE"):
                v = None

            value = _check_and_coerce_cfg_value_type(v, self[k], k)
            cfg_dict[k] = value
        self.update_from_flattened_dict(cfg_dict)

    @classmethod
    def load_yaml_with_base(
            cls, filename: str, allow_unsafe: bool = False
    ) -> Dict[str, Any]:
        """
        Just like `yaml.load(open(filename))`, but inherit attributes from its
            `_BASE_`.

        Args:
            filename (str or file-like object): the file name or file of the current config.
                Will be used to find the base config file.
            allow_unsafe (bool): whether to allow loading the config file with
                `yaml.unsafe_load`.

        Returns:
            (dict): the loaded yaml
        """
        with cls._open_cfg(filename) as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                if not allow_unsafe:
                    raise
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Loading config {} with yaml.unsafe_load. Your machine may "
                    "be at risk if the file contains malicious content.".format(
                        filename
                    )
                )
                f.close()
                with cls._open_cfg(filename) as f:
                    cfg = yaml.unsafe_load(f)
        return cfg

    @classmethod
    def load_from_file(self, filename, allow_unsafe: bool = False):
        if filename.endswith("json"):
            loaded_cfg = jload(filename)
        else:
            loaded_cfg = self.load_yaml_with_base(filename, allow_unsafe=allow_unsafe)

        loaded_cfg = CfgNode(loaded_cfg)
        return loaded_cfg

    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = False) -> None:
        """
        Merge configs from a given yaml file.

        Args:
            cfg_filename: the file name of the yaml config.
            allow_unsafe: whether to allow loading the config file with
                `yaml.unsafe_load`.
        """
        loaded_cfg = self.load_from_file(cfg_filename, allow_unsafe=allow_unsafe)
        self.merge_from_other_cfg(loaded_cfg)

    def dump(self, **kwargs):
        return self.to_yaml(**kwargs)

    def is_frozen(self):
        return self.is_locked


def _check_and_coerce_cfg_value_type(replacement, original, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # If either of them is None, allow type conversion to one of the valid types
    if (replacement_type == type(None) and original_type in _VALID_TYPES) or (
            original_type == type(None) and replacement_type in _VALID_TYPES
    ):
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]
    # For py2: allow converting from str (bytes) to a unicode string
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


CN = CfgNode
