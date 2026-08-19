"""Deterministic small document fixtures without external image dependencies."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

SYNTHETIC_CASES = (
    "white_page_dark_background",
    "white_page_light_background",
    "dark_page_light_background",
    "rotated_quadrilateral",
    "perspective_quadrilateral",
    "partial_cut_off_page",
    "multiple_rectangle_distractors",
    "shadow_gradient",
    "low_contrast_receipt",
    "blurred_page",
    "glare_patch",
    "dirty_background",
)


@dataclass(frozen=True)
class SyntheticDocumentSample:
    """A small Gray8 sample plus compatible geometry/mask/quality annotations."""

    sample_id: str
    case: str
    width: int
    height: int
    pixels: tuple[int, ...]
    mask: tuple[int, ...]
    quad: tuple[tuple[float, float], ...]
    quality_labels: dict[str, float]
    cleanup_mask: tuple[int, ...]
    restoration_target: tuple[int, ...] | None

    def pgm_bytes(self) -> bytes:
        """Encode a tiny portable Gray8 PGM without adding a Pillow dependency."""
        return f"P5\n{self.width} {self.height}\n255\n".encode("ascii") + bytes(self.pixels)

    def sha256(self) -> str:
        """Hash encoded image bytes for an auditable DatasetManifest sample record."""
        return hashlib.sha256(self.pgm_bytes()).hexdigest()

    def annotations(self) -> dict[str, object]:
        """Return annotations using the names accepted by the dataset contract."""
        return {
            "quad": [[x, y] for x, y in self.quad],
            "mask": {"width": self.width, "height": self.height},
            "keypoints": [[x, y] for x, y in self.quad],
            "quality_labels": self.quality_labels,
            "cleanup_mask": {"width": self.width, "height": self.height},
        }


class SyntheticDocumentGenerator:
    """Creates deterministic in-memory pages for test-only model baselines."""

    def __init__(self, *, width: int = 96, height: int = 72, seed: int = 7) -> None:
        if width < 16 or height < 16:
            raise ValueError("synthetic dimensions must be at least 16 pixels")
        self.width = width
        self.height = height
        self.seed = seed

    def generate(self, case: str, *, index: int = 0) -> SyntheticDocumentSample:
        """Create one named fixture with no global random-state mutation."""
        if case not in SYNTHETIC_CASES:
            raise ValueError(f"unsupported synthetic case: {case}")
        rng = random.Random(f"{self.seed}:{case}:{index}")
        background, page = self._colors(case)
        quad = self._quad(case, rng)
        pixels = [background] * (self.width * self.height)
        mask = [0] * len(pixels)
        for y in range(self.height):
            for x in range(self.width):
                pixel_index = y * self.width + x
                if _inside_convex_quad(x + 0.5, y + 0.5, quad):
                    pixels[pixel_index] = page
                    mask[pixel_index] = 255
        self._draw_case_effects(case, pixels, mask, rng)
        cleanup_mask = self._cleanup_mask(case, rng)
        target = (
            tuple(page if value else background for value in mask)
            if case in {"shadow_gradient", "glare_patch", "blurred_page"}
            else None
        )
        return SyntheticDocumentSample(
            sample_id=f"synthetic-{case}-{index:03d}",
            case=case,
            width=self.width,
            height=self.height,
            pixels=tuple(pixels),
            mask=tuple(mask),
            quad=tuple(quad),
            quality_labels=self._quality_labels(case),
            cleanup_mask=tuple(cleanup_mask),
            restoration_target=target,
        )

    def generate_suite(self, count_per_case: int = 1) -> list[SyntheticDocumentSample]:
        """Generate a stable suite across all documented fixture cases."""
        if count_per_case < 1:
            raise ValueError("count_per_case must be positive")
        return [
            self.generate(case, index=index)
            for case in SYNTHETIC_CASES
            for index in range(count_per_case)
        ]

    def write_pgm(self, sample: SyntheticDocumentSample, directory: Path) -> Path:
        """Write a fixture only when a caller deliberately provides a temporary output directory."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{sample.sample_id}.pgm"
        path.write_bytes(sample.pgm_bytes())
        return path

    def _colors(self, case: str) -> tuple[int, int]:
        if case == "white_page_light_background":
            return 180, 240
        if case == "dark_page_light_background":
            return 225, 48
        if case == "low_contrast_receipt":
            return 178, 194
        return 30, 230

    def _quad(self, case: str, rng: random.Random) -> list[tuple[float, float]]:
        left, top = 15.0, 10.0
        right, bottom = self.width - 15.0, self.height - 10.0
        if case == "rotated_quadrilateral":
            return [(left + 7, top), (right, top + 6), (right - 7, bottom), (left, bottom - 6)]
        if case == "perspective_quadrilateral":
            return [(left + 9, top + 4), (right - 2, top), (right - 10, bottom), (left, bottom - 4)]
        if case == "partial_cut_off_page":
            return [(-7, top + 4), (right, top), (right - 5, bottom), (-3, bottom - 2)]
        if case == "multiple_rectangle_distractors":
            return [
                (left + 5, top + 3),
                (right - 6, top + 2),
                (right - 8, bottom - 4),
                (left + 4, bottom - 5),
            ]
        jitter = rng.uniform(-1.0, 1.0)
        return [
            (left + jitter, top),
            (right, top + jitter),
            (right - jitter, bottom),
            (left, bottom - jitter),
        ]

    def _draw_case_effects(
        self, case: str, pixels: list[int], mask: list[int], rng: random.Random
    ) -> None:
        if case == "multiple_rectangle_distractors":
            self._fill_rect(pixels, 2, 2, 14, 15, 210)
            self._fill_rect(pixels, self.width - 17, 3, self.width - 3, 17, 215)
        if case == "shadow_gradient":
            for y in range(self.height):
                for x in range(self.width):
                    idx = y * self.width + x
                    if mask[idx]:
                        pixels[idx] = max(0, pixels[idx] - int(80 * x / self.width))
        if case == "glare_patch":
            self._fill_rect(
                pixels,
                self.width // 2 - 8,
                self.height // 2 - 6,
                self.width // 2 + 9,
                self.height // 2 + 7,
                255,
            )
        if case == "blurred_page":
            original = list(pixels)
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    idx = y * self.width + x
                    if mask[idx]:
                        pixels[idx] = (
                            sum(
                                original[(y + dy) * self.width + x + dx]
                                for dy in (-1, 0, 1)
                                for dx in (-1, 0, 1)
                            )
                            // 9
                        )
        if case == "dirty_background":
            for _ in range(20):
                x, y = rng.randrange(self.width), rng.randrange(self.height)
                idx = y * self.width + x
                if not mask[idx]:
                    pixels[idx] = min(255, pixels[idx] + rng.randrange(20, 70))

    def _cleanup_mask(self, case: str, rng: random.Random) -> list[int]:
        mask = [0] * (self.width * self.height)
        if case in {"glare_patch", "dirty_background"}:
            for _ in range(8):
                x, y = rng.randrange(self.width), rng.randrange(self.height)
                mask[y * self.width + x] = 255
        return mask

    @staticmethod
    def _quality_labels(case: str) -> dict[str, float]:
        labels = {
            "blur": 0.0,
            "shadow": 0.0,
            "glare": 0.0,
            "curvature": 0.0,
            "noise": 0.0,
            "faded": 0.0,
        }
        if case == "blurred_page":
            labels["blur"] = 1.0
        if case == "shadow_gradient":
            labels["shadow"] = 1.0
        if case == "glare_patch":
            labels["glare"] = 1.0
        if case == "low_contrast_receipt":
            labels["faded"] = 1.0
        if case == "partial_cut_off_page":
            labels["curvature"] = 0.0
        return labels

    def _fill_rect(
        self, pixels: list[int], left: int, top: int, right: int, bottom: int, value: int
    ) -> None:
        for y in range(max(0, top), min(self.height, bottom)):
            for x in range(max(0, left), min(self.width, right)):
                pixels[y * self.width + x] = value


def _inside_convex_quad(x: float, y: float, quad: list[tuple[float, float]]) -> bool:
    signs = []
    for index, (ax, ay) in enumerate(quad):
        bx, by = quad[(index + 1) % len(quad)]
        signs.append((bx - ax) * (y - ay) - (by - ay) * (x - ax))
    return all(value >= 0 for value in signs) or all(value <= 0 for value in signs)
