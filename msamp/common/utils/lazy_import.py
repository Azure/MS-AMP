# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Lazy import utility."""

import importlib


class LazyImport:
    """Lazy import Python moduels, only import when modules are used."""
    def __init__(self, name, attr=None, callback=None):
        """Init lazy import class.

        Args:
            name (str): Python module name.
            attr (str, optional): Function or class name in the module. Defaults to None.
            callback (callable, optional): Callback function. Defaults to None.
        """
        self._module = None
        self._name = name
        self._attr = attr
        self._callback = callback

    def _import(self):
        """Import the needed module when it is used."""
        if self._module is None:
            self._module = importlib.import_module(self._name)
            if self._attr is not None:
                self._module = getattr(self._module, self._attr)
            if self._callback is not None:
                self._callback()

    def __getattr__(self, item):
        """Override __getattr__.

        Args:
            item (str): Attribute name.

        Returns:
            Any: Attribute value.
        """
        self._import()
        return getattr(self._module, item)

    def __dir__(self):
        """Override __dir__.

        Returns:
            List[str]: The list of attributes.
        """
        self._import()
        return dir(self._module)

    def __call__(self, *args, **kwargs):
        """Override __call__.

        Args:
            *args (list): Arguments.
            **kwargs (dict): Keyword arguments.

        Returns:
            Any: The return value of the function.
        """
        self._import()
        return self._module(*args, **kwargs)
