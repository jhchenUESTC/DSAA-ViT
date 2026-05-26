#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess PAM-4 eye-diagram density matrices for DSAA-ViT.

This script converts an exported eyeMatrix.csv into:
1) a colormap-rendered RGB eye diagram;
2) a 224 x 224 RGB image for the ViT backbone;
3) a patch-aligned density prior for attention modulation;
4) optional visualization figures for the RGB rendering and density prior.

Preprocessing rules:
- Zero-density pixels are mapped to white.
- Nonzero density values in [1, 100] are uniformly quantized into 15 color levels.
- The rendered RGB image is resized to 224 x 224 using bicubic interpolation.
- The raw density matrix is also resized to 224 x 224 using bicubic interpolation.
- The patch-aligned density prior is obtained by averaging each 16 x 16 patch and
  then min-max normalizing the patch-level prior to [0, 1].

Example:
    python preprocess_eye.py \
        --input "D:/.../sample_data/eyeMatrix_17.csv" \
        --out-dir "D:/.../outputs"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image


ADS_COLORMAP_15 = np.array(
    [
        [0, 0, 255],        # blue
        [0, 191, 255],      # deep sky blue
        [0, 245, 255],      # cyan-like
        [151, 255, 255],    # light cyan
        [193, 255, 193],    # pale green
        [154, 255, 154],    # light green
        [0, 255, 0],        # green
        [192, 255, 62],     # green-yellow
        [202, 255, 112],    # light green-yellow
        [255, 236, 139],    # light yellow
        [255, 255, 0],      # yellow
        [255, 215, 0],      # gold
        [255, 193, 37],     # golden orange
        [255, 127, 36],     # orange
        [238, 106, 80],     # highest, salmon/orange-red
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for eye-diagram preprocessing."""

    image_size: int = 224
    patch_size: int = 16
    density_min: float = 0.0
    density_max: float = 100.0
    save_outputs: bool = True
    show_figures: bool = False


def read_density_matrix(csv_path: str | Path) -> np.ndarray:
    """Read an exported eyeMatrix.csv as a float32 matrix."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

    matrix = pd.read_csv(csv_path, header=None).values.astype(np.float32)

    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D density matrix, got shape {matrix.shape}.")

    return matrix


def extract_sample_id(path: str | Path) -> str:
    """
    Extract a sample ID from the input file name.

    Supported examples:
    - eyeMatrix_17.csv   -> 17
    - eyeMatrix_821.csv  -> 821
    - eyeMatrix.csv      -> 0
    """
    file_stem = Path(path).stem  # e.g., eyeMatrix_17 or eyeMatrix

    match = re.fullmatch(r"eyeMatrix_(\d+)", file_stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    if file_stem.lower() == "eyematrix":
        return "0"

    raise ValueError(
        f"Cannot extract sample ID from file name: {Path(path).name}. "
        "Expected 'eyeMatrix.csv' or 'eyeMatrix_<number>.csv'."
    )


def matrix_to_colormap_rgb(
    matrix: np.ndarray,
    density_min: float = 0.0,
    density_max: float = 100.0,
) -> Image.Image:
    """
    Convert a density matrix to a colormap-rendered RGB eye-diagram image.

    Mapping rule:
    - 0-density pixels are mapped to white.
    - Nonzero density values are uniformly quantized into 15 color levels.
    """
    if density_max <= density_min:
        raise ValueError("density_max must be larger than density_min.")

    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.clip(matrix, density_min, density_max)

    height, width = matrix.shape
    rgb = np.ones((height, width, 3), dtype=np.uint8) * 255

    positive_mask = matrix > density_min
    if np.any(positive_mask):
        # Map nonzero values in (0, density_max] into 15 levels: 0 ... 14.
        level = np.ceil(matrix * 15.0 / density_max).astype(np.int32) - 1
        level = np.clip(level, 0, 14)
        rgb[positive_mask] = ADS_COLORMAP_15[level[positive_mask]]

    return Image.fromarray(rgb, mode="RGB")


def resize_rgb_image(rgb_img: Image.Image, image_size: int = 224) -> Image.Image:
    """Resize the rendered RGB image using bicubic interpolation."""
    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    return rgb_img.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)


def extract_patch_density_prior(
    matrix: np.ndarray,
    image_size: int = 224,
    patch_size: int = 16,
    density_min: float = 0.0,
    density_max: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a patch-aligned density prior from the raw density matrix.

    Steps:
    1) Clip the density matrix to [density_min, density_max].
    2) Resize it to image_size x image_size using bicubic interpolation.
    3) Normalize the resized density map to [0, 1].
    4) Average each patch_size x patch_size region to obtain an Hp x Wp prior.
    5) Apply min-max normalization to the patch-level prior.
    """
    if density_max <= density_min:
        raise ValueError("density_max must be larger than density_min.")
    if image_size <= 0 or patch_size <= 0:
        raise ValueError("image_size and patch_size must be positive.")
    if image_size % patch_size != 0:
        raise ValueError(
            f"image_size={image_size} must be divisible by patch_size={patch_size}."
        )

    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.clip(matrix, density_min, density_max)

    density_img = Image.fromarray(matrix)
    density_img_resized = density_img.resize(
        (image_size, image_size),
        resample=Image.Resampling.BICUBIC,
    )

    density_resized = np.asarray(density_img_resized, dtype=np.float32)
    density_resized = np.clip(density_resized, density_min, density_max)
    density_resized_norm = (density_resized - density_min) / (density_max - density_min)

    h_p = image_size // patch_size
    w_p = image_size // patch_size

    patches = density_resized_norm.reshape(h_p, patch_size, w_p, patch_size)
    density_prior = patches.mean(axis=(1, 3))

    prior_min = float(density_prior.min())
    prior_max = float(density_prior.max())

    if prior_max > prior_min:
        density_prior_norm = (density_prior - prior_min) / (prior_max - prior_min)
    else:
        density_prior_norm = np.zeros_like(density_prior, dtype=np.float32)

    return density_prior_norm.astype(np.float32), density_resized_norm.astype(np.float32)


def build_rgb_colorbar(density_max: float = 100.0) -> Tuple[ListedColormap, BoundaryNorm, np.ndarray, list[str]]:
    """Build a discrete colorbar for the 15-level RGB rendering."""
    color_list = np.vstack(
        [
            np.array([[255, 255, 255]], dtype=np.uint8),
            ADS_COLORMAP_15,
        ]
    ) / 255.0

    cmap = ListedColormap(color_list)
    bounds = np.arange(-0.5, 16.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    tick_pos = np.arange(16)
    edges = np.linspace(1, density_max, 16)

    tick_labels = ["0"]
    for i in range(15):
        tick_labels.append(f"{edges[i]:.1f}-{edges[i + 1]:.1f}")

    return cmap, norm, tick_pos, tick_labels


def set_plot_style() -> None:
    """Set figure fonts for publication-style visualization."""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"


def plot_rendered_rgb_with_colorbar(
    rgb_img_resized: Image.Image,
    density_max: float,
    title: str = "Rendered RGB Eye Diagram",
):
    """Create a compact figure of the rendered RGB eye diagram with its colorbar."""
    cmap, norm, tick_pos, tick_labels = build_rgb_colorbar(density_max=density_max)

    fig = plt.figure(figsize=(7.0, 4.2))
    ax_img = fig.add_axes([0.06, 0.12, 0.66, 0.76])
    ax_cb = fig.add_axes([0.65, 0.12, 0.035, 0.76])

    ax_img.imshow(rgb_img_resized)
    ax_img.set_title(title, fontsize=14)
    ax_img.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cb = plt.colorbar(sm, cax=ax_cb, boundaries=np.arange(-0.5, 16.5, 1.0), ticks=tick_pos)
    cb.set_label("Density value", fontsize=12)
    cb.ax.set_yticklabels(tick_labels)
    cb.ax.tick_params(labelsize=10)

    return fig


def plot_density_prior(
    density_prior: np.ndarray,
    cmap: str = "YlOrBr",
):
    """Visualize the patch-aligned density prior."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))

    im = ax.imshow(
        density_prior,
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="equal",
    )

    ax.set_title(
        f"Patch-Aligned Density Prior ({density_prior.shape[0]}×{density_prior.shape[1]})",
        fontsize=14,
    )
    ax.set_xlabel("Patch column", fontsize=12)
    ax.set_ylabel("Patch row", fontsize=12)

    ax.set_xticks(np.arange(density_prior.shape[1]))
    ax.set_yticks(np.arange(density_prior.shape[0]))
    ax.tick_params(labelsize=8)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Normalized density", fontsize=12)
    cb.ax.tick_params(labelsize=10)

    fig.tight_layout()
    return fig


def save_outputs(
    out_dir: str | Path,
    sample_id: str,
    rgb_img_resized: Image.Image,
    density_prior: np.ndarray,
    density_resized_norm: np.ndarray,
    fig_rgb,
    fig_prior,
    image_size: int,
    patch_size: int,
) -> None:
    """Save images, figures, and CSV outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_grid = image_size // patch_size

    paths = {
        "resized_rgb": out_dir / f"sample_{sample_id}_rendered_rgb_{image_size}x{image_size}.png",
        "rgb_figure": out_dir / f"sample_{sample_id}_rendered_rgb_with_colorbar.png",
        "density_prior_csv": out_dir / f"sample_{sample_id}_density_prior_{patch_grid}x{patch_grid}.csv",
        "density_prior_figure": out_dir / f"sample_{sample_id}_density_prior_visualization.png",
        "density_resized_csv": out_dir / f"sample_{sample_id}_density_resized_norm_{image_size}x{image_size}.csv",
    }

    rgb_img_resized.save(paths["resized_rgb"])

    fig_rgb.savefig(paths["rgb_figure"], dpi=600, bbox_inches="tight")
    fig_prior.savefig(paths["density_prior_figure"], dpi=600, bbox_inches="tight")

    pd.DataFrame(density_prior).to_csv(paths["density_prior_csv"], header=False, index=False)
    pd.DataFrame(density_resized_norm).to_csv(
        paths["density_resized_csv"], header=False, index=False
    )

    print("Saved files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def preprocess_eye_matrix(
    input_csv: str | Path,
    out_dir: str | Path,
    config: PreprocessConfig,
) -> None:
    """Run the full preprocessing pipeline for one eyeMatrix.csv file."""
    set_plot_style()

    matrix = read_density_matrix(input_csv)
    sample_id = extract_sample_id(input_csv)

    rgb_img = matrix_to_colormap_rgb(
        matrix,
        density_min=config.density_min,
        density_max=config.density_max,
    )
    rgb_img_resized = resize_rgb_image(rgb_img, image_size=config.image_size)

    density_prior, density_resized_norm = extract_patch_density_prior(
        matrix,
        image_size=config.image_size,
        patch_size=config.patch_size,
        density_min=config.density_min,
        density_max=config.density_max,
    )

    print("Input CSV:", input_csv)
    print("Sample ID:", sample_id)
    print("Raw density matrix shape:", matrix.shape)
    print("Resized RGB input size:", rgb_img_resized.size)
    print("Patch-aligned density prior shape:", density_prior.shape)

    fig_rgb = plot_rendered_rgb_with_colorbar(
        rgb_img_resized,
        density_max=config.density_max,
    )
    fig_prior = plot_density_prior(density_prior)

    if config.save_outputs:
        save_outputs(
            out_dir=out_dir,
            sample_id=sample_id,
            rgb_img_resized=rgb_img_resized,
            density_prior=density_prior,
            density_resized_norm=density_resized_norm,
            fig_rgb=fig_rgb,
            fig_prior=fig_prior,
            image_size=config.image_size,
            patch_size=config.patch_size,
        )

    if config.show_figures:
        plt.show()
    else:
        plt.close(fig_rgb)
        plt.close(fig_prior)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess an eyeMatrix.csv file for DSAA-ViT."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to eyeMatrix.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory for saving preprocessing outputs.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="RGB input image size. Default: 224.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size for extracting density prior. Default: 16.",
    )
    parser.add_argument(
        "--density-min",
        type=float,
        default=0.0,
        help="Minimum density value. Default: 0.",
    )
    parser.add_argument(
        "--density-max",
        type=float,
        default=100.0,
        help="Maximum density value. Default: 100.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output files.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()

    config = PreprocessConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        density_min=args.density_min,
        density_max=args.density_max,
        save_outputs=not args.no_save,
        show_figures=args.show,
    )

    preprocess_eye_matrix(
        input_csv=args.input,
        out_dir=args.out_dir,
        config=config,
    )


if __name__ == "__main__":
    main()