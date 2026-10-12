"""The charts' colors, shared by the report's figures (matplotlib) and the dashboard (Altair).

Validated with the dataviz skill's palette validator against each surface: slots 1-3 pass on
every pair and slots 1-7 on adjacent pairs, in both modes. A chart that can put any two series
side by side (lines that cross) therefore uses at most 3 slots. This module imports no plotting
library, so the dashboard does not need matplotlib.
"""

from dataclasses import dataclass

__all__ = ["EVERY_PAIR", "THEMES", "Theme"]

EVERY_PAIR = 3  # the slots that pass the validator on every pair, in both modes


@dataclass(frozen=True)
class Theme:
    """A surface's colors: chart chrome, ink, and the categorical slots in their fixed order."""

    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    rest: str  # the folded band of the weights stack: a neutral, not a slot


THEMES = {
    "light": Theme(
        surface="#fcfcfb",
        ink="#0b0b0b",
        secondary="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"),
        rest="#898781",
    ),
    "dark": Theme(
        surface="#1a1a19",
        ink="#ffffff",
        secondary="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"),
        rest="#898781",
    ),
}
