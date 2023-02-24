# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP HookManager."""

import torch


class HookManager:
    """HookManager class to manage hooks."""
    class _HookModule(torch.nn.Module):
        """A module purely forward input tensor."""
        def forward(self, x):
            """Pass input tensor to output."""
            return x

    class _RemoveHandle:
        """A remove handle to remove a hook from HookManager."""
        def __init__(self, manager, hook_id):
            """Constructor.

            Args:
                manager (HookManager): Hook manager to remove.
                hook_id: Hook id.
            """
            self.manager = manager
            self.hook_id = hook_id

        def remove(self):
            """Remove hook id from hook manager."""
            if self.hook_id in self.manager.hooks:
                del self.manager.hooks[self.hook_id]

    def __init__(self):
        """Constructor to initialize members variables."""
        self.hooks = dict()
        self.history_counter = 0

    def __call__(self, *args, **kwargs):
        """Override __call__ to call all hooks.

        Args:
            *args (list): Arguments.
            **kwargs (dict): Keyword arguments.
        """
        for hook in self.hooks.values():
            hook(*args, **kwargs)

    def register_hook(self, fn):
        """Register a funcrtion to hook manager.

        Args:
            fn: a function should have the following signature: hook(module, input, output) -> None.

        Returns:
            _RemoveHandle: A remove handle for deleting this hook.
        """
        hook_id = self.history_counter
        self.hooks[hook_id] = self._create_hook(fn)
        self.history_counter += 1
        return self._get_hook_deleter(hook_id)

    def _create_hook(self, fn):
        """Create a hook.

        Args:
            fn: A function should have the following signature: hook(module, input, output) -> None.

        Returns:
            _HookModule: An instance of _HoolModule.
        """
        mod = self._HookModule()
        mod.register_forward_hook(fn)
        return mod

    def _get_hook_deleter(self, hook_id):
        """Get hook deleter for deleting hook.

        Args:
            hook_id (int): hook id.
        """
        return self._RemoveHandle(self, hook_id)
