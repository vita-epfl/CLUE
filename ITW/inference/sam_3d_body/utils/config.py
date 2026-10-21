# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

from typing import Dict

import yaml

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
    from omegaconf.errors import InterpolationResolutionError
    from yacs.config import CfgNode as CN
except ImportError:
    DictConfig = ListConfig = OmegaConf = InterpolationResolutionError = None

    class CN(dict):
        """Small yacs-like config node used when yacs/omegaconf are unavailable."""

        def __init__(self, init_dict=None, new_allowed=True):
            super().__init__()
            self.__dict__["new_allowed"] = new_allowed
            for key, value in (init_dict or {}).items():
                self[key] = self._wrap(value)

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key, value):
            if key == "new_allowed":
                self.__dict__[key] = value
            else:
                self[key] = self._wrap(value)

        def _wrap(self, value):
            if isinstance(value, dict) and not isinstance(value, CN):
                return CN(value, new_allowed=self.new_allowed)
            if isinstance(value, list):
                return [self._wrap(item) for item in value]
            return value

        def merge_from_other_cfg(self, other):
            for key, value in other.items():
                if key in self and isinstance(self[key], CN) and isinstance(value, CN):
                    self[key].merge_from_other_cfg(value)
                else:
                    self[key] = self._wrap(value)

        def defrost(self):
            return None

        def freeze(self):
            return None


def resolve_omegaconf_to_dict(conf):
    """
    Recursively convert an OmegaConf object to a dictionary, resolving interpolations
    where possible and leaving unsupported ones as-is.
    """
    if OmegaConf is None:
        return conf
    if isinstance(conf, DictConfig):
        result = {}
        for k, v in conf.items():
            try:
                result[k] = resolve_omegaconf_to_dict(v)
            except InterpolationResolutionError:
                result[k] = OmegaConf.to_container(v, resolve=False)
        return result
    elif isinstance(conf, ListConfig):
        result = []
        for item in conf:
            try:
                result.append(resolve_omegaconf_to_dict(item))
            except InterpolationResolutionError:
                result.append(OmegaConf.to_container(item, resolve=False))
        return result
    else:
        if OmegaConf.is_config(conf):
            try:
                return OmegaConf.to_container(conf, resolve=True)
            except InterpolationResolutionError:
                return OmegaConf.to_container(conf, resolve=False)
        return conf


def to_lower(x: Dict) -> Dict:
    """
    Convert all dictionary keys to lowercase
    Args:
      x (dict): Input dictionary
    Returns:
      dict: Output dictionary with all keys converted to lowercase
    """
    return {k.lower(): v for k, v in x.items()}


def get_config(config_file: str) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
    Returns:
      Config as a yacs-style CfgNode object.
    """
    cfg = CN(new_allowed=True)

    if OmegaConf is None:
        with open(config_file, "r", encoding="utf-8") as handle:
            conf_dict = yaml.safe_load(handle) or {}
    else:
        conf = OmegaConf.load(config_file)
        conf_dict = resolve_omegaconf_to_dict(conf)

    cfg.merge_from_other_cfg(CN(conf_dict))
    cfg.freeze()
    return cfg
