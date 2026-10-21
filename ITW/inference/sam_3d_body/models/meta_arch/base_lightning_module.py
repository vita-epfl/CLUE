# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch.nn as nn


class BaseLightningModule(nn.Module):
    """Minimal PyTorch replacement for the upstream Lightning base class.

    The vendored model is used as a normal ``nn.Module``.  These methods are
    kept so upstream model code that calls Lightning helpers still imports and
    runs without requiring pytorch-lightning.
    """

    def save_hyperparameters(self, *args, **kwargs):
        return None

    def _log_metric(self, *args, **kwargs):
        return None

    def _log_image(self, *args, **kwargs):
        return None

    def _log_hist(self, *args, **kwargs):
        return None
