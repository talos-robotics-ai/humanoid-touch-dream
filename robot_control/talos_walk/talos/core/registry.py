"""RobotRegistry — plug-in discovery: ``get_robot("h2") -> RobotSpec``.

A robot is a package ``talos.robots.<name>`` exposing ``build_spec() -> RobotSpec``.
Specs are validated once and cached per process.
"""

from __future__ import annotations

import pkgutil
from importlib import import_module

from talos.core.robot import RobotSpec


class RobotRegistry:
    _cache: dict[str, RobotSpec] = {}

    @classmethod
    def available(cls) -> list[str]:
        import talos.robots as pkg
        return sorted(m.name for m in pkgutil.iter_modules(pkg.__path__) if m.ispkg)

    @classmethod
    def get(cls, name: str) -> RobotSpec:
        if name not in cls._cache:
            try:
                mod = import_module(f"talos.robots.{name}")
            except ModuleNotFoundError as e:
                raise KeyError(f"unknown robot '{name}'; available: {cls.available()}") from e
            spec = mod.build_spec()
            if not isinstance(spec, RobotSpec):
                raise TypeError(f"talos.robots.{name}.build_spec() must return a RobotSpec")
            cls._cache[name] = spec.validate()
        return cls._cache[name]

    @classmethod
    def register(cls, spec: RobotSpec) -> None:
        """Register a spec built in code (tests, ad-hoc robots)."""
        cls._cache[spec.name] = spec.validate()


def get_robot(name: str) -> RobotSpec:
    return RobotRegistry.get(name)
