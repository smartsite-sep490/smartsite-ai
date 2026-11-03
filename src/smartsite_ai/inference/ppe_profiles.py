"""Immutable detector taxonomies, not model evaluation or release assertions."""

from dataclasses import dataclass
from typing import Literal

ExperimentalPpeProfile = Literal["native-ppe-10", "unified-ppe-15"]


@dataclass(frozen=True, slots=True)
class PpeModelProfile:
    """Declared evidence families; absence never supplies an explicit negative."""

    name: str
    class_map: tuple[tuple[int, str], ...]
    present_items: tuple[str, ...]
    explicit_missing_items: tuple[str, ...]
    experimental: bool


LEGACY_PPE_5_PROFILE = PpeModelProfile(
    name="legacy-ppe-5",
    class_map=(
        (0, "Person"),
        (1, "Hardhat"),
        (2, "NO-Hardhat"),
        (3, "Safety Vest"),
        (4, "NO-Safety Vest"),
    ),
    present_items=("HARD_HAT", "SAFETY_VEST"),
    explicit_missing_items=("HARD_HAT", "SAFETY_VEST"),
    experimental=False,
)

NATIVE_PPE_10_PROFILE = PpeModelProfile(
    name="native-ppe-10",
    class_map=(
        (0, "Person"),
        (1, "Hardhat"),
        (2, "NO-Hardhat"),
        (3, "Safety Vest"),
        (4, "Gloves"),
        (5, "NO-Gloves"),
        (6, "Boots"),
        (7, "NO-Boots"),
        (8, "Goggles"),
        (9, "NO-Goggles"),
    ),
    present_items=("HARD_HAT", "SAFETY_VEST", "GLOVES", "BOOTS", "GOGGLES"),
    explicit_missing_items=("HARD_HAT", "GLOVES", "BOOTS", "GOGGLES"),
    experimental=True,
)

# Eight detector object families; only the existing five wearer observation
# families are eligible for the immutable technical ingestion contract.
UNIFIED_PPE_15_PROFILE = PpeModelProfile(
    name="unified-ppe-15",
    class_map=tuple(
        enumerate(
            (
                "Person",
                "Hardhat",
                "NO-Hardhat",
                "Safety Vest",
                "NO-Safety Vest",
                "Gloves",
                "NO-Gloves",
                "Boots",
                "NO-Boots",
                "Goggles",
                "NO-Goggles",
                "Mask",
                "NO-Mask",
                "fullbody-Harness",
                "Lanyard-Hook",
            )
        )
    ),
    present_items=NATIVE_PPE_10_PROFILE.present_items,
    explicit_missing_items=("HARD_HAT", "SAFETY_VEST", "GLOVES", "BOOTS", "GOGGLES"),
    experimental=True,
)


def experimental_ppe_profile(name: ExperimentalPpeProfile | None) -> PpeModelProfile:
    if name is None:
        return LEGACY_PPE_5_PROFILE
    if name == "native-ppe-10":
        return NATIVE_PPE_10_PROFILE
    if name == "unified-ppe-15":
        return UNIFIED_PPE_15_PROFILE
    raise ValueError("unsupported experimental PPE profile")
