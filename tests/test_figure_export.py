"""Regression tests for paired raster/vector publication exports."""

from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from open_leo_latency_routing.visualization import (
    IEEE_TEXT_WIDTH_IN,
    configure_ieee_figure_style,
    save_png_pdf_pair,
)


class FigureExportTests(unittest.TestCase):
    def test_one_canvas_is_exported_to_matching_png_and_pdf_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "publication_figure.png"
            figure, axis = plt.subplots()
            axis.plot([0, 1], [0, 1], marker="o", linestyle="--")
            try:
                png_path, pdf_path = save_png_pdf_pair(figure, output)
            finally:
                plt.close(figure)

            self.assertEqual(png_path, output)
            self.assertEqual(pdf_path, output.with_suffix(".pdf"))
            self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF-"))
            self.assertGreater(png_path.stat().st_size, 1_000)
            self.assertGreater(pdf_path.stat().st_size, 1_000)

    def test_publication_export_has_fixed_canvas_rgb_png_and_opaque_pdf(self) -> None:
        configure_ieee_figure_style(base_font_size=9.0)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "ieee_figure"
            figure, axis = plt.subplots(figsize=(IEEE_TEXT_WIDTH_IN, 2.0))
            axis.plot([0, 1], [0, 1], marker="s", linestyle="--")
            axis.grid(color="#d9d9d9")
            try:
                png_path, pdf_path = save_png_pdf_pair(
                    figure,
                    output,
                    dpi=100,
                )
            finally:
                plt.close(figure)

            with Image.open(png_path) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (int(IEEE_TEXT_WIDTH_IN * 100), 200))

            pdf_bytes = pdf_path.read_bytes()
            media_box = re.search(
                rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
                pdf_bytes,
            )
            self.assertIsNotNone(media_box)
            assert media_box is not None
            self.assertAlmostEqual(float(media_box.group(1)), IEEE_TEXT_WIDTH_IN * 72, places=3)
            self.assertAlmostEqual(float(media_box.group(2)), 144.0, places=3)
            alpha_values = re.findall(rb"/(?:CA|ca)\s+([0-9]*\.?[0-9]+)", pdf_bytes)
            self.assertTrue(alpha_values)
            self.assertTrue(all(abs(float(value) - 1.0) < 1e-12 for value in alpha_values))

    def test_publication_export_rejects_partial_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "transparent_figure"
            figure, axis = plt.subplots()
            axis.grid(alpha=0.5)
            try:
                with self.assertRaisesRegex(ValueError, "non-opaque PDF"):
                    save_png_pdf_pair(figure, output)
            finally:
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
