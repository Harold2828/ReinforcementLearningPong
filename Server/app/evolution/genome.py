from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Genome:
    """Ordered hidden-layer widths; depth is in [1, 4] per SPEC-02/SPEC-04."""

    hiddenWidths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.hiddenWidths:
            raise ValueError("a genome must have at least one hidden layer")
        if any(width <= 0 for width in self.hiddenWidths):
            raise ValueError(f"hidden widths must be positive, got {self.hiddenWidths}")

    @property
    def depth(self) -> int:
        return len(self.hiddenWidths)

    def validate(self, min_layers: int, max_layers: int, allowed_widths: set[int]) -> None:
        if not min_layers <= self.depth <= max_layers:
            raise ValueError(
                f"hidden layer count {self.depth} out of range [{min_layers}, {max_layers}]"
            )
        unsupported = sorted({w for w in self.hiddenWidths if w not in allowed_widths})
        if unsupported:
            raise ValueError(f"unsupported hidden widths {unsupported}")

    def to_json(self) -> dict:
        return {"schemaVersion": "genome-v1", "hiddenWidths": list(self.hiddenWidths)}