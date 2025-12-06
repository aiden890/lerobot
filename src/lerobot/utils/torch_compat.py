#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Compatibility helpers for bridging gaps between different PyTorch releases.

The upstream `transformers` package expects PyTorch >= 2.6 to expose
`TransformGetItemToIndex` from `torch._dynamo._trace_wrapped_higher_order_op`.
Older torch builds used in robotics deployments might not provide that symbol
even if their version string reports 2.6+, which results in an ImportError
during `transformers` import. This module provides a lightweight fallback
implementation so that the rest of the stack can keep running.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


def ensure_transform_getitem_to_index() -> None:
    """
    Ensure `torch._dynamo._trace_wrapped_higher_order_op.TransformGetItemToIndex`
    exists by lazily patching the module with a compatible fallback when needed.
    """

    try:
        module = import_module("torch._dynamo._trace_wrapped_higher_order_op")
    except Exception:
        # Torch is not available, nothing to patch.
        return

    if hasattr(module, "TransformGetItemToIndex"):
        return

    try:
        _install_transform_getitem_to_index(module)
    except Exception:
        # Intentionally swallow the error to avoid breaking environments where
        # torch internals differ substantially. Worst case, transformers will
        # raise the original ImportError which is still actionable.
        return


def _install_transform_getitem_to_index(module: ModuleType) -> None:
    """
    Register a simplified but compatible version of TransformGetItemToIndex.
    """

    import torch
    import torch.utils._pytree as pytree
    from torch.overrides import TorchFunctionMode

    class _ModIndex(torch.autograd.Function):
        """
        Custom autograd function mimicking PyTorch's newer scatter-based index op.
        """

        @staticmethod
        def forward(ctx: Any, x: torch.Tensor, indices: list[torch.Tensor]) -> torch.Tensor:
            ctx.save_for_backward(*indices)
            ctx.input_shape = x.shape
            return torch.ops.aten.index(x, indices)

        @staticmethod
        def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, None]:
            indices = ctx.saved_tensors
            grad = torch.zeros(ctx.input_shape, dtype=grad_out.dtype, device=grad_out.device)
            grad = torch.ops.aten.index_put(grad, indices, grad_out, accumulate=True)
            return grad, None

    mod_index = _ModIndex.apply

    class TransformGetItemToIndex(TorchFunctionMode):
        """
        Minimal drop-in replacement for the upstream helper used by transformers.
        """

        def __torch_function__(
            self,
            func: Any,
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            if func == torch.Tensor.__getitem__ and len(args) >= 2:
                index_args = pytree.tree_leaves(args[1])
                if index_args and all(isinstance(arg, torch.Tensor) for arg in index_args):
                    return mod_index(args[0], index_args)
            return func(*args, **(kwargs or {}))

    setattr(module, "TransformGetItemToIndex", TransformGetItemToIndex)
