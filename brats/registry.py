"""Single source of truth for walking the algorithm metadata registry.

The public enum API (``brats.constants``) and the YAML metadata under
``brats/data/meta`` are bound together by ``ALGORITHM_META_FILE_BY_ENUM``. This
module turns that binding into a flat, iterable view of every registered
algorithm so that tooling (verification suite, contract tests) does not have to
re-implement the walk.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from brats.constants import ALGORITHM_META_FILE_BY_ENUM, Algorithms
from brats.utils.algorithm_config import AlgorithmData, load_algorithms


@dataclass(frozen=True)
class AlgorithmRecord:
    """A single algorithm declared in the metadata registry."""

    enum_cls: type[Algorithms]
    """Enum class the algorithm belongs to."""
    member: str
    """Metadata key / enum value of the algorithm, e.g. ``"BraTS25_1"``."""
    data: AlgorithmData
    """Parsed metadata for the algorithm."""
    meta_path: Path
    """Metadata YAML file the algorithm was declared in."""

    @property
    def algorithm(self) -> Algorithms:
        """The enum member for this algorithm."""
        return self.enum_cls(self.member)

    @property
    def track(self) -> str:
        """Metadata file stem for the algorithm's track, e.g. ``"inpainting"``."""
        return self.meta_path.stem


def iter_algorithms() -> Iterator[AlgorithmRecord]:
    """Yield every registered algorithm as an :class:`AlgorithmRecord`.

    Yields:
        AlgorithmRecord: One record per algorithm in ``ALGORITHM_META_FILE_BY_ENUM``
    """
    for enum_cls, meta_path in ALGORITHM_META_FILE_BY_ENUM.items():
        for member, data in load_algorithms(meta_path).items():
            yield AlgorithmRecord(
                enum_cls=enum_cls, member=member, data=data, meta_path=meta_path
            )
