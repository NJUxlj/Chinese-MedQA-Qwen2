# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import collections
from dataclasses import FrozenInstanceError, dataclass, field, fields
from typing import Any


# BaseConfig class inherits from collections.abc.Mapping, which means it can act like a dictionary
@dataclass(frozen=True)
class BaseConfig(collections.abc.Mapping):
    """The BaseConfig provides dict-like interface for a dataclass config.

    By default all fields in the config is not mutable, unless specified in
    "_mutable_fields". The BaseConfig class implements the Mapping Abstract Base Class.
    This allows instances of this class to be used like dictionaries.
    """

    _mutable_fields = {"extra"}
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value associated with the given key. If the key does not exist, return the default value.

        Args:
            key (str): The attribute name to retrieve.
            default (Any, optional): The value to return if the attribute does not exist. Defaults to None.

        Returns:
            Any: The value of the attribute or the default value.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __getitem__(self, key: str):
        """Implement the [] operator for the class. Allows accessing attributes like dictionary items.

        Args:
            key (str): The attribute name to retrieve.

        Returns:
            Any: The value of the attribute.

        Raises:
            AttributeError: If the attribute does not exist.
            TypeError: If the key type is not string
        """
        return getattr(self, key)

    def __iter__(self):
        """Implement the iterator protocol. Allows iterating over the attribute names of the instance.

        Yields:
            str: The name of each field in the dataclass.
        """
        for f in fields(self):
            yield f.name

    def __len__(self):
        """
        Return the number of fields in the dataclass.

        Returns:
            int: The number of fields in the dataclass.
        """
        return len(fields(self))


@dataclass(frozen=True)
class ProfilerConfig(BaseConfig):
    """Worker profiler config. Currently only support Nsight system profiler.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        discrete (bool): True for each task has its own database, False for all tasks in one training step
          share one database.
        all_ranks (bool): Whether to profile all ranks.
        ranks (list[int]): The ranks that will be profiled. Defaults to [].
    """

    discrete: bool = False

    all_ranks: bool = False

    ranks: list[int] = field(default_factory=list)

    def union(self, other: "ProfilerConfig") -> "ProfilerConfig":
        return ProfilerConfig(
            all_ranks=self.all_ranks or other.all_ranks,
            ranks=list(set(self.ranks or []) | set(other.ranks or [])),
            discrete=self.discrete or other.discrete,
        )

    def intersect(self, other: "ProfilerConfig") -> "ProfilerConfig":
        return ProfilerConfig(
            all_ranks=self.all_ranks and other.all_ranks,
            ranks=list(set(self.ranks or []) & set(other.ranks or [])),
            discrete=self.discrete and other.discrete,
        )

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert isinstance(self.ranks, (set, list, tuple)), (
            f"Profiler ranks must be of type list, got {type(self.ranks)}"
        )
