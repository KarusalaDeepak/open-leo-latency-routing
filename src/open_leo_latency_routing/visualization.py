"""Small, deterministic helpers for publication figure export."""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# IEEEtran journal mode uses a 43-pica text block, which is 7.1399 physical
# inches after converting TeX points.  The rounded value also maps to an exact
# 714-pixel canvas at Matplotlib's deterministic 100-dpi layout resolution.
IEEE_TEXT_WIDTH_IN = 7.14
IEEE_MIN_EFFECTIVE_TEXT_PT = 8.0
IEEE_PREFERRED_TEXT_PT = 9.0
OPAQUE_GRID_COLOR = "#d9d9d9"

_SANS_SERIF_FALLBACKS = (
    "Arial",
    "Helvetica",
    "Liberation Sans",
    "DejaVu Sans",
)
_PDF_ALPHA_PATTERN = re.compile(rb"/(?:CA|ca)\s+([0-9]*\.?[0-9]+)")


def configure_ieee_figure_style(*, base_font_size: float = 9.0) -> str:
    """Apply an opaque, IEEE-sized plotting style and return the chosen font.

    Arial and Helvetica are preferred for production compatibility.  The
    explicit fallbacks keep headless Linux rebuilds deterministic when those
    proprietary/system fonts are unavailable.
    """

    import matplotlib as mpl
    from matplotlib import font_manager

    selected_font = _SANS_SERIF_FALLBACKS[-1]
    for candidate in _SANS_SERIF_FALLBACKS:
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        selected_font = candidate
        break

    mpl.rcParams.update(
        {
            "font.family": selected_font,
            "font.size": base_font_size,
            "axes.titlesize": max(9.0, base_font_size),
            "axes.labelsize": base_font_size,
            "xtick.labelsize": base_font_size,
            "ytick.labelsize": base_font_size,
            "legend.fontsize": base_font_size,
            "legend.title_fontsize": base_font_size,
            "figure.dpi": 100,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "grid.color": OPAQUE_GRID_COLOR,
            "grid.alpha": 1.0,
            "grid.linewidth": 0.65,
            "axes.axisbelow": True,
            "legend.framealpha": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return selected_font


def _flatten_png_to_rgb(path: Path, *, dpi: int) -> None:
    """Remove an unnecessary PNG alpha channel against an opaque white page."""

    from PIL import Image

    with Image.open(path) as image:
        resolution = image.info.get("dpi", (dpi, dpi))
        if "A" in image.getbands():
            white = Image.new("RGB", image.size, "white")
            white.paste(image.convert("RGB"), mask=image.getchannel("A"))
            flattened = white
        else:
            flattened = image.convert("RGB")
        flattened.save(path, dpi=resolution)


def _remove_backend_only_pdf_transparency(path: Path) -> None:
    """Replace Matplotlib's invisible-stroke alpha state with opaque state.

    The PDF backend emits ``/CA 0`` when an artist has no stroke, even though
    that artist also has zero stroke width.  Replacing the single digit is
    rendering-neutral and preserves every byte offset in the PDF xref table.
    Any other non-opaque state is treated as a generator regression.
    """

    original = path.read_bytes()
    opaque = original.replace(b"/CA 0 /ca 1", b"/CA 1 /ca 1")
    non_opaque = [
        value
        for value in _PDF_ALPHA_PATTERN.findall(opaque)
        if abs(float(value) - 1.0) > 1e-12
    ]
    if non_opaque:
        values = ", ".join(sorted({value.decode("ascii") for value in non_opaque}))
        raise ValueError(f"non-opaque PDF graphics state(s) remain in {path}: {values}")
    if opaque != original:
        path.write_bytes(opaque)


def save_png_pdf_pair(
    figure: "Figure",
    output: str | Path,
    *,
    dpi: int = 300,
    bbox_inches: str | None = None,
) -> tuple[Path, Path]:
    """Save one figure canvas as a high-resolution PNG and vector PDF.

    ``output`` may be a suffixless path or end in ``.png``/``.pdf``.  The
    returned paths always share the same stem, which prevents the raster and
    vector manuscript assets from silently diverging.
    """

    requested = Path(output)
    stem = (
        requested.with_suffix("")
        if requested.suffix.lower() in {".png", ".pdf"}
        else requested
    )
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.patch.set_facecolor("white")
    figure.savefig(
        png_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    _flatten_png_to_rgb(png_path, dpi=dpi)
    figure.savefig(
        pdf_path,
        bbox_inches=bbox_inches,
        facecolor="white",
        edgecolor="white",
        transparent=False,
        metadata={"CreationDate": None, "ModDate": None},
    )
    _remove_backend_only_pdf_transparency(pdf_path)
    return png_path, pdf_path
