"""
Motor imagery PDF report generator

The public behavior is intentionally kept close to the original class:
instantiating MotorImageryPdfReport builds and saves the report by default
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from PIL import Image
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg


XAnchor = Literal["left", "center", "right"]
YAnchor = Literal["bottom", "center", "top"]
TextAlign = Literal["left", "center", "right"]
PathLike = Union[str, Path]


MONTAGE_LABELS = {
    "DSI_24": "DSI 24 chs",
    "GTEC_32": "g.Nautilus 32 chs",
    "EGI_64": "HydroCel GNS 64 chs",
    "EGI_128": "HydroCel GNS 128 chs",
}


class MotorImageryPdfReport:
    """Create a PDF report from saved analysis plots and summary metadata"""

    def __init__(
        self,
        plot_folder: PathLike,
        helper_folder: PathLike,
        date_test: str = "N/A",
        montage_name: str = "N/A",
        resolution: int = 1,
        age_at_test: str = "N/A",
        save_folder: Optional[PathLike] = None,
        verbose: bool = True,
        auto_generate: bool = True,
        report_title: str = "Command Following Report",
        report_name: Optional[str] = None,
    ) -> None:
        """Initialize report paths and metadata used to assemble the PDF"""
        self.plot_folder = Path(plot_folder).expanduser()
        self.helper_folder = Path(helper_folder).expanduser()
        self.save_folder = Path(save_folder).expanduser() if save_folder else self.plot_folder
        self.verbose = verbose
        self.report_title = report_title

        self.show_all_lines = False
        self.show_all_points = False

        self.formatted_date = date.today().strftime("%Y-%m-%d")
        self.version = "N/A"
        self.base_name = report_name or self._infer_report_name(self.plot_folder, self.save_folder)
        self.sub_name, self.ses_name = self._parse_subject_session(self.base_name)

        self.date_test = date_test
        self.montage_name = MONTAGE_LABELS.get(montage_name, "N/A")
        self.resolution = resolution
        self.age_at_test = age_at_test
        self.output_name = f"mcf_report_{self.base_name}.pdf"
        self.output_path = self.save_folder / self.output_name

        self.spatial_filter = "Surface Laplacian"
        self.condition = self._infer_condition(self.sub_name)

        self.canvas_dim: Dict[str, float] = {}
        self.dict_alph: Dict[str, Tuple[float, float]] = {}
        self.missing_files: List[Path] = []
        self.drawn_files: List[Path] = []

        if auto_generate:
            self.generate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self) -> Path:
        """Build and save the PDF report"""
        self._validate_paths()
        self._create_canvas()
        self._get_canvas_dims()

        self._log(f"Creating report: {self.output_path}")

        self._log("Page 1/3: timeline, topoplots, band effects")
        self._start_page()
        self._plot_timeline()
        self._plot_topoplots()
        self._plot_band_effects()
        self._next_page()

        self._log("Page 2/3: brain schematic and PSDs")
        self._start_page()
        self._plot_brain()
        self._plot_psds()
        self._next_page()

        self._log("Page 3/3: statistical distributions and bridged candidates")
        self._start_page()
        self._plot_stat_dist()
        self._plot_bridged_candidates()

        self._export_report_generation_metadata()
        self.c.save()
        self._log(f"Saved report: {self.output_path}")
        self._log(f"Figures drawn: {len(self.drawn_files)}")

        if self.missing_files:
            missing_names = ", ".join(path.name for path in self.missing_files)
            self._log(f"Optional figures skipped: {missing_names}")

        return self.output_path

    # ------------------------------------------------------------------
    # Metadata and validation
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_report_name(plot_folder: Path, save_folder: Path) -> str:
        """Infer the subject/run folder name for legacy callers."""
        if plot_folder.name.lower() == "images":
            if save_folder.name.lower() == "one_over_f_subtracted" and save_folder.parent.name:
                return save_folder.parent.name
            if save_folder.name:
                return save_folder.name
            if plot_folder.parent.name:
                return plot_folder.parent.name
        return plot_folder.name

    @staticmethod
    def _parse_subject_session(base_name: str) -> Tuple[str, str]:
        """Extract subject and session from a BIDS-like folder name"""
        subject = "N/A"
        session = "N/A"

        subject_match = re.search(r"sub-([^_]+)", base_name)
        session_match = re.search(r"ses-([^_]+)", base_name)

        if subject_match:
            subject = subject_match.group(1)

        if session_match:
            session = session_match.group(1)

        return subject, session

    @staticmethod
    def _infer_condition(subject: str) -> str:
        """Infer clinical condition from the subject identifier"""
        subject_upper = subject.upper()

        if "HC" in subject_upper:
            return "HC"

        if "TBI" in subject_upper:
            return "TBI"

        return "N/A"

    def _validate_paths(self) -> None:
        """Validate required folders and create the output folder"""
        if not self.plot_folder.exists():
            raise FileNotFoundError(f"Plot folder does not exist: {self.plot_folder}")

        if not self.plot_folder.is_dir():
            raise NotADirectoryError(f"Plot folder is not a directory: {self.plot_folder}")

        if not self.helper_folder.exists():
            self._log(f"Helper folder not found, helper images will be skipped: {self.helper_folder}")

        self.save_folder.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        """Print terminal feedback when verbose mode is enabled"""
        if self.verbose:
            print(f"[PdfReport] {message}")

    def _export_report_generation_metadata(self) -> None:
        """Write metadata about the PDF generation process"""
        payload = {
            "output_pdf": str(self.output_path),
            "report_title": self.report_title,
            "report_name": self.base_name,
            "subject": self.sub_name,
            "session": self.ses_name,
            "date_test": self.date_test,
            "montage": self.montage_name,
            "age_at_test": self.age_at_test,
            "condition": self.condition,
            "source_plot_folder": str(self.plot_folder),
            "figures_drawn": [path.name for path in self.drawn_files],
            "missing_optional_files": [path.name for path in self.missing_files],
            "generated_on": self.formatted_date,
        }
        path = self.save_folder / "pdf_report_generation_metadata.json"
        try:
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            self._log(f"Could not write PDF metadata JSON: {exc}")

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def move_right_by(x: float, dx: float) -> float:
        """Move an x coordinate to the right by dx"""
        return x + dx

    @staticmethod
    def move_left_by(x: float, dx: float) -> float:
        """Move an x coordinate to the left by dx"""
        return x - dx

    @staticmethod
    def move_up_by(y: float, dy: float) -> float:
        """Move a y coordinate upward by dy"""
        return y + dy

    @staticmethod
    def move_down_by(y: float, dy: float) -> float:
        """Move a y coordinate downward by dy"""
        return y - dy

    def show_key_point(self, name: str, x: float, y: float, show: bool = True) -> None:
        """Create an anchor point at given coordinates"""
        self.dict_alph[name] = (x, y)

        if show:
            self.c.setFont("Helvetica", 16)
            self.c.drawString(x, y, name)

    def draw_hline(
        self,
        x1: float,
        x2: float,
        y: float,
        color=colors.black,
        line_width: float = 1.0,
        return_coord: bool = True,
    ) -> Optional[Dict[str, float]]:
        """Draw a horizontal line between x1 and x2 at a fixed y"""
        self.c.setStrokeColor(color)
        self.c.setLineWidth(line_width)
        self.c.line(x1, y, x2, y)

        if return_coord:
            return {"x1": x1, "x2": x2, "y": y}

        return None

    def draw_vline(
        self,
        x: float,
        y1: float,
        y2: float,
        color=colors.black,
        line_width: float = 1.0,
        return_coord: bool = True,
    ) -> Optional[Dict[str, float]]:
        """Draw a vertical line between y1 and y2 at a fixed x"""
        self.c.setStrokeColor(color)
        self.c.setLineWidth(line_width)
        self.c.line(x, y1, x, y2)

        if return_coord:
            return {"x": x, "y1": y1, "y2": y2}

        return None

    def draw_dline(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color=colors.black,
        line_width: float = 1.0,
        return_coord: bool = True,
    ) -> Optional[Dict[str, float]]:
        """Draw a diagonal line from x1, y1 to x2, y2"""
        self.c.setStrokeColor(color)
        self.c.setLineWidth(line_width)
        self.c.line(x1, y1, x2, y2)

        if return_coord:
            return {"x1": x1, "x2": x2, "y1": y1, "y2": y2}

        return None

    def _create_canvas(self) -> None:
        """Create a ReportLab canvas"""
        self.c = canvas.Canvas(str(self.output_path), pagesize=letter)

    def _next_page(self) -> None:
        """Force the document to create a new page"""
        self.c.showPage()

    def _get_canvas_dims(self, margin: int = 15) -> None:
        """Create a dictionary of canvas dimensions"""
        width, height = letter
        xmin = 0
        ymin = 0
        xmax = width
        ymax = height

        self.canvas_dim = {
            "width": width,
            "height": height,
            "xleft": xmin,
            "xright": xmax,
            "xmid": (xmax - xmin) / 2,
            "ytop": ymax,
            "ybottom": ymin,
            "ymid": abs(ymax - ymin) / 2,
            "margin": margin,
        }

    def _create_corner_points(self) -> None:
        """Create corner points with appropriate coordinates"""
        margin = self.canvas_dim["margin"]

        self.top_left = (
            self.move_right_by(self.canvas_dim["xleft"], margin),
            self.move_down_by(self.canvas_dim["ytop"], margin),
        )
        self.top_right = (
            self.move_left_by(self.canvas_dim["xright"], margin),
            self.move_down_by(self.canvas_dim["ytop"], margin),
        )
        self.bottom_right = (
            self.move_left_by(self.canvas_dim["xright"], margin),
            self.move_up_by(self.canvas_dim["ybottom"], margin),
        )
        self.bottom_left = (
            self.move_right_by(self.canvas_dim["xleft"], margin),
            self.move_up_by(self.canvas_dim["ybottom"], margin),
        )

        self.page_width = self.top_right[0] - self.top_left[0]
        self.page_height = self.top_right[1] - self.bottom_right[1]

    def _create_page_sections(self) -> None:
        """Create points and optional debug lines to split the page into sections"""
        page_unit = self.page_height / 20
        line_below_header = 2 * page_unit
        line_below_upper_section = 11 * page_unit
        xmid_with_margin = self.move_right_by(self.page_width / 2, self.top_left[0])

        self.dict_alph = {}
        self.show_key_point("A", self.top_left[0], self.top_left[1], self.show_all_points)
        self.show_key_point("B", self.top_right[0], self.top_right[1], self.show_all_points)
        self.show_key_point("C", self.bottom_right[0], self.bottom_right[1], self.show_all_points)
        self.show_key_point("D", self.bottom_left[0], self.bottom_left[1], self.show_all_points)

        self.show_key_point(
            "E",
            self.top_left[0],
            self.move_down_by(self.top_left[1], line_below_header),
            self.show_all_points,
        )
        self.show_key_point(
            "F",
            self.top_right[0],
            self.move_down_by(self.top_left[1], line_below_header),
            self.show_all_points,
        )
        self.show_key_point("I", xmid_with_margin, self.top_right[1], self.show_all_points)
        self.show_key_point(
            "J",
            xmid_with_margin,
            self.move_down_by(self.top_left[1], line_below_header),
            self.show_all_points,
        )

        self.show_key_point(
            "G",
            self.top_left[0],
            self.move_down_by(self.top_left[1], line_below_upper_section),
            self.show_all_points,
        )
        self.show_key_point(
            "H",
            self.top_right[0],
            self.move_down_by(self.top_left[1], line_below_upper_section),
            self.show_all_points,
        )
        self.show_key_point(
            "K",
            xmid_with_margin,
            self.move_down_by(self.top_left[1], line_below_upper_section),
            self.show_all_points,
        )
        self.show_key_point("L", xmid_with_margin, self.bottom_right[1], self.show_all_points)

        if self.show_all_lines:
            self._draw_page_debug_lines(line_below_header, line_below_upper_section)

    def _draw_page_debug_lines(
        self,
        line_below_header: float,
        line_below_upper_section: float,
    ) -> None:
        """Draw optional page layout guide lines"""
        self.draw_hline(self.top_left[0], self.top_right[0], self.top_left[1], colors.black)
        self.draw_vline(self.top_right[0], self.top_right[1], self.bottom_right[1], colors.red)
        self.draw_hline(self.bottom_left[0], self.bottom_right[0], self.bottom_left[1], colors.blue)
        self.draw_vline(self.bottom_left[0], self.top_left[1], self.bottom_left[1], colors.green)
        self.draw_vline(self.canvas_dim["xmid"], self.top_right[1], self.bottom_right[1], colors.red)
        self.draw_hline(
            self.top_left[0],
            self.top_right[0],
            self.move_down_by(self.top_left[1], line_below_upper_section),
            colors.grey,
        )
        self.draw_hline(
            self.top_left[0],
            self.top_right[0],
            self.move_down_by(self.top_left[1], line_below_header),
            colors.black,
        )

    def _start_page(self) -> None:
        """Initialize the repeated page structure"""
        self._create_corner_points()
        self._create_page_sections()
        self._generate_header()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _write_text(
        self,
        text_list: List[str],
        x: float,
        y: float,
        font_size: int = 12,
        bold_flags: Optional[List[bool]] = None,
        align: TextAlign = "center",
    ) -> None:
        """Write a list of adjacent text fragments at a specified position"""
        if not text_list:
            return

        if bold_flags is None:
            bold_flags = [False] * len(text_list)

        if len(bold_flags) != len(text_list):
            raise ValueError("bold_flags must have the same length as text_list")

        if align not in {"center", "left", "right"}:
            raise ValueError(f"Invalid text alignment: {align}")

        fragments = []
        total_width = 0.0

        for text, is_bold in zip(text_list, bold_flags):
            font = "Helvetica-Bold" if is_bold else "Helvetica"
            text_value = str(text)
            width = self.c.stringWidth(text_value, font, font_size)
            fragments.append((text_value, width, font))
            total_width += width

        if align == "center":
            current_x = x - total_width / 2
        elif align == "right":
            current_x = x - total_width
        else:
            current_x = x

        for text, width, font in fragments:
            self.c.setFont(font, font_size)
            self.c.drawString(current_x, y, text)
            current_x += width

    @staticmethod
    def _is_svg(path: PathLike) -> bool:
        """Return True if the file looks like an SVG"""
        return Path(path).suffix.lower() == ".svg"

    @staticmethod
    def _get_raster_size(path: PathLike) -> Tuple[float, float]:
        """Return raster image size in pixels"""
        with Image.open(path) as img:
            width, height = img.size

        return float(width), float(height)

    @staticmethod
    def _resolve_size_preserve_ratio(
        orig_w: float,
        orig_h: float,
        width: Optional[float],
        height: Optional[float],
    ) -> Tuple[float, float]:
        """Resolve final width and height while preserving aspect ratio"""
        if orig_w <= 0 or orig_h <= 0:
            raise ValueError("Invalid intrinsic size for image")

        if width is None and height is None:
            return orig_w, orig_h

        if width is not None and height is None:
            scale = width / orig_w
            return width, orig_h * scale

        if height is not None and width is None:
            scale = height / orig_h
            return orig_w * scale, height

        scale = min(width / orig_w, height / orig_h)
        return orig_w * scale, orig_h * scale

    @staticmethod
    def _anchor_to_bottom_left(
        x: float,
        y: float,
        w: float,
        h: float,
        x_anchor: XAnchor,
        y_anchor: YAnchor,
    ) -> Tuple[float, float]:
        """Convert an anchor point to bottom-left coordinates"""
        if x_anchor == "left":
            x_left = x
        elif x_anchor == "center":
            x_left = x - w / 2
        elif x_anchor == "right":
            x_left = x - w
        else:
            raise ValueError(f"Invalid x_anchor: {x_anchor}")

        if y_anchor == "bottom":
            y_bottom = y
        elif y_anchor == "center":
            y_bottom = y - h / 2
        elif y_anchor == "top":
            y_bottom = y - h
        else:
            raise ValueError(f"Invalid y_anchor: {y_anchor}")

        return x_left, y_bottom

    def draw_figure_at(
        self,
        image_path: PathLike,
        *,
        x: float,
        y: float,
        x_anchor: XAnchor = "left",
        y_anchor: YAnchor = "bottom",
        width: Optional[float] = None,
        height: Optional[float] = None,
    ) -> Dict[str, float]:
        """Draw a figure whose position is defined by an anchor point"""
        path = Path(image_path)
        drawing = None

        if self._is_svg(path):
            drawing = svg2rlg(str(path))

            if drawing is None:
                raise ValueError(f"Could not parse SVG file: {path}")

            orig_w = float(drawing.width)
            orig_h = float(drawing.height)
        else:
            orig_w, orig_h = self._get_raster_size(path)

        final_w, final_h = self._resolve_size_preserve_ratio(orig_w, orig_h, width, height)
        x_left, y_bottom = self._anchor_to_bottom_left(
            x=x,
            y=y,
            w=final_w,
            h=final_h,
            x_anchor=x_anchor,
            y_anchor=y_anchor,
        )

        if drawing is not None:
            scale = final_w / float(drawing.width)
            drawing.scale(scale, scale)
            renderPDF.draw(drawing, self.c, x_left, y_bottom)
        else:
            self.c.drawImage(str(path), x_left, y_bottom, width=final_w, height=final_h)

        return {
            "xleft": x_left,
            "y_bottom": y_bottom,
            "width": final_w,
            "height": final_h,
        }

    def _draw_if_exists(
        self,
        image_path: PathLike,
        *,
        x: float,
        y: float,
        x_anchor: XAnchor = "left",
        y_anchor: YAnchor = "bottom",
        width: Optional[float] = None,
        height: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """Draw an image if it exists and record missing optional inputs"""
        path = Path(image_path)

        if not path.exists():
            self.missing_files.append(path)
            return None

        placement = self.draw_figure_at(
            path,
            x=x,
            y=y,
            x_anchor=x_anchor,
            y_anchor=y_anchor,
            width=width,
            height=height,
        )
        self.drawn_files.append(path)
        return placement

    def _draw_plot(self, filename: str, **kwargs) -> Optional[Dict[str, float]]:
        """Draw a plot from the plot folder if it exists"""
        return self._draw_if_exists(self.plot_folder / filename, **kwargs)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _generate_header(self) -> None:
        """Populate header with subject and measurement information"""
        y0 = 15
        dx = 10

        self._write_text(
            [self.report_title],
            self.canvas_dim["xmid"],
            self.move_down_by(self.dict_alph["I"][1], y0),
            font_size=17,
            bold_flags=[True],
            align="center",
        )
        self._write_text(
            [f"Generated: {self.formatted_date}"],
            self.top_right[0],
            self.move_down_by(self.dict_alph["I"][1], y0),
            font_size=11,
            bold_flags=[False],
            align="right",
        )
        self._write_text(
            [f"Version: {self.version}"],
            self.top_right[0],
            self.move_down_by(self.move_down_by(self.dict_alph["I"][1], y0), y0),
            font_size=10,
            bold_flags=[False],
            align="right",
        )

        font_size = 11
        header_height = self.move_down_by(self.dict_alph["I"][1], 15) - self.move_up_by(
            self.dict_alph["J"][1], 5
        )
        dy = header_height / 3

        self._write_text(
            ["Subject: ", self.sub_name],
            self.canvas_dim["xmid"] * 2 / 3 - dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="right",
        )
        self._write_text(
            ["Session: ", self.ses_name],
            self.canvas_dim["xmid"] * 2 / 3 - dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + 2 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="right",
        )
        self._write_text(
            ["Date of assessment: ", self.date_test],
            self.canvas_dim["xmid"] * 2 / 3 - dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + 3 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="right",
        )

        self._write_text(
            ["Age at test: ", f"{self.age_at_test} "],
            self.canvas_dim["xmid"],
            self.move_down_by(self.dict_alph["I"][1], y0 + dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="center",
        )
        self._write_text(
            ["Condition: ", f"{self.condition} "],
            self.canvas_dim["xmid"],
            self.move_down_by(self.dict_alph["I"][1], y0 + 2 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="center",
        )
        self._write_text(
            ["Montage: ", self.montage_name],
            self.canvas_dim["xmid"],
            self.move_down_by(self.dict_alph["I"][1], y0 + 3 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="center",
        )

        right_column_x = self.canvas_dim["xmid"] * 2 * 2 / 3 + dx
        self._write_text(
            ["Resolution: ", f"{self.resolution} Hz"],
            right_column_x,
            self.move_down_by(self.dict_alph["I"][1], y0 + dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="left",
        )
        self._write_text(
            ["Spatial filter: ", self.spatial_filter],
            right_column_x,
            self.move_down_by(self.dict_alph["I"][1], y0 + 2 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="left",
        )
        self._write_text(
            ["Filter: ", "1-40 Hz"],
            right_column_x,
            self.move_down_by(self.dict_alph["I"][1], y0 + 3 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="left",
        )

    # ------------------------------------------------------------------
    # Page 1
    # ------------------------------------------------------------------
    def _plot_timeline(self) -> None:
        """Plot paradigm timeline"""
        self._draw_plot(
            "paradigm_timeline.svg",
            x=self.canvas_dim["xmid"],
            y=self.move_down_by(self.dict_alph["J"][1], 5),
            x_anchor="center",
            y_anchor="top",
            width=500,
        )

    def _plot_topoplots(self) -> None:
        """Plot band topoplots and p-value maps"""
        dy = 130
        height = 300

        self._draw_plot(
            "topoplot.svg",
            x=self.canvas_dim["xmid"],
            y=self.move_down_by(self.dict_alph["J"][1], dy),
            x_anchor="center",
            y_anchor="top",
            height=height,
        )
        self._draw_plot(
            "pvalue_left.svg",
            x=self.move_right_by(self.canvas_dim["xleft"], 55),
            y=self.move_down_by(self.dict_alph["J"][1], dy + 30),
            x_anchor="left",
            y_anchor="top",
            height=int(height * 0.95),
        )
        self._draw_plot(
            "pvalue_right.svg",
            x=self.move_left_by(self.canvas_dim["xright"], 55),
            y=self.move_down_by(self.dict_alph["J"][1], dy + 30),
            x_anchor="right",
            y_anchor="top",
            height=int(height * 0.93),
        )

    def _plot_band_effects(self) -> None:
        """Plot bootstrap CI per band"""
        self._draw_plot(
            "band_effect.svg",
            x=self.canvas_dim["xmid"],
            y=self.move_up_by(self.dict_alph["L"][1], 5),
            x_anchor="center",
            y_anchor="bottom",
            width=450,
        )

    # ------------------------------------------------------------------
    # Page 2
    # ------------------------------------------------------------------
    def _plot_brain(self) -> None:
        """Plot brain icon with guide lines"""
        y_mid = abs(self.dict_alph["J"][1] - self.dict_alph["L"][1]) / 2
        self._draw_if_exists(
            self.helper_folder / "brain_c3c4p3p4.svg",
            x=self.canvas_dim["xmid"],
            y=self.move_up_by(y_mid, 10),
            x_anchor="center",
            y_anchor="center",
            width=85,
        )

        coord = self.draw_vline(
            x=self.move_right_by(self.canvas_dim["xmid"], 16),
            y1=self.move_up_by(y_mid, 34),
            y2=self.move_up_by(y_mid, 68),
            color=colors.blue,
        )
        self.draw_dline(
            x1=coord["x"],
            x2=self.move_left_by(coord["x"], 90),
            y1=coord["y2"],
            y2=self.move_up_by(coord["y2"], 70),
            color=colors.blue,
            return_coord=False,
        )

        coord = self.draw_vline(
            x=self.move_left_by(self.canvas_dim["xmid"], 13),
            y1=self.move_up_by(y_mid, 34),
            y2=self.move_up_by(y_mid, 68),
            color=colors.red,
        )
        self.draw_dline(
            x1=coord["x"],
            x2=self.move_right_by(coord["x"], 90),
            y1=coord["y2"],
            y2=self.move_up_by(coord["y2"], 70),
            color=colors.red,
            return_coord=False,
        )

        coord = self.draw_dline(
            x1=self.move_right_by(self.canvas_dim["xmid"], 20),
            x2=self.move_right_by(self.canvas_dim["xmid"], 12.5),
            y1=self.move_up_by(y_mid, 7.5),
            y2=y_mid,
            color=colors.blue,
        )
        self.draw_hline(
            x1=coord["x2"],
            x2=self.move_left_by(coord["x2"], 100),
            y=coord["y2"],
            color=colors.blue,
            return_coord=False,
        )

        coord = self.draw_dline(
            x1=self.move_left_by(self.canvas_dim["xmid"], 17),
            x2=self.move_left_by(self.canvas_dim["xmid"], 9.5),
            y1=self.move_up_by(y_mid, 12),
            y2=self.move_up_by(y_mid, 19.5),
            color=colors.red,
        )
        self.draw_hline(
            x1=coord["x2"],
            x2=self.move_right_by(coord["x2"], 100),
            y=coord["y2"],
            color=colors.red,
            return_coord=False,
        )

        coord = self.draw_vline(
            x=self.move_right_by(self.canvas_dim["xmid"], 16),
            y1=self.move_down_by(y_mid, 11),
            y2=self.move_down_by(y_mid, 45),
            color=colors.blue,
        )
        self.draw_dline(
            x1=coord["x"],
            x2=self.move_left_by(coord["x"], 90),
            y1=coord["y2"],
            y2=self.move_down_by(coord["y2"], 70),
            color=colors.blue,
            return_coord=False,
        )

        coord = self.draw_vline(
            x=self.move_left_by(self.canvas_dim["xmid"], 13),
            y1=self.move_down_by(y_mid, 11),
            y2=self.move_down_by(y_mid, 45),
            color=colors.red,
        )
        self.draw_dline(
            x1=coord["x"],
            x2=self.move_right_by(coord["x"], 90),
            y1=coord["y2"],
            y2=self.move_down_by(coord["y2"], 70),
            color=colors.red,
            return_coord=False,
        )

    def _plot_psds(self) -> None:
        """Plot PSDs of selected channels"""
        y_mid = abs(self.dict_alph["J"][1] - self.dict_alph["L"][1]) / 2
        width = 200
        narrow_width = int(width * 0.85)

        self._draw_plot(
            "psd_fc3.svg",
            x=self.move_right_by(self.canvas_dim["xmid"], 30),
            y=self.move_down_by(self.dict_alph["J"][1], 5),
            x_anchor="left",
            y_anchor="top",
            width=narrow_width,
        )
        self._draw_plot(
            "psd_c3.svg",
            x=self.move_left_by(self.canvas_dim["xright"], 20),
            y=self.move_up_by(y_mid, 10),
            x_anchor="right",
            y_anchor="center",
            width=width,
        )
        self._draw_first_existing_plot(
            [
                (
                    "psd_p3.svg",
                    {
                        "x": self.move_right_by(self.canvas_dim["xmid"], 30),
                        "y": self.move_up_by(self.canvas_dim["ybottom"], 5),
                        "x_anchor": "left",
                        "y_anchor": "bottom",
                        "width": narrow_width,
                    },
                ),
                (
                    "psd_cp3.svg",
                    {
                        "x": self.move_left_by(self.canvas_dim["xmid"], 30),
                        "y": self.move_up_by(self.canvas_dim["ybottom"], 5),
                        "x_anchor": "right",
                        "y_anchor": "bottom",
                        "width": narrow_width,
                    },
                ),
            ]
        )

        self._draw_plot(
            "psd_fc4.svg",
            x=self.move_left_by(self.canvas_dim["xmid"], 30),
            y=self.move_down_by(self.dict_alph["J"][1], 5),
            x_anchor="right",
            y_anchor="top",
            width=narrow_width,
        )
        self._draw_plot(
            "psd_c4.svg",
            x=self.move_right_by(self.canvas_dim["xleft"], 20),
            y=self.move_up_by(y_mid, 10),
            x_anchor="left",
            y_anchor="center",
            width=width,
        )
        self._draw_first_existing_plot(
            [
                (
                    "psd_p4.svg",
                    {
                        "x": self.move_left_by(self.canvas_dim["xmid"], 30),
                        "y": self.move_up_by(self.canvas_dim["ybottom"], 5),
                        "x_anchor": "right",
                        "y_anchor": "bottom",
                        "width": narrow_width,
                    },
                ),
                (
                    "psd_cp4.svg",
                    {
                        "x": self.move_left_by(self.canvas_dim["xmid"], 30),
                        "y": self.move_up_by(self.canvas_dim["ybottom"], 5),
                        "x_anchor": "right",
                        "y_anchor": "bottom",
                        "width": narrow_width,
                    },
                ),
            ]
        )

    def _draw_first_existing_plot(self, candidates: List[Tuple[str, Dict]]) -> None:
        """Draw the first existing plot from a candidate list"""
        missing_candidates = []

        for filename, kwargs in candidates:
            path = self.plot_folder / filename

            if path.exists():
                self._draw_if_exists(path, **kwargs)
                return

            missing_candidates.append(path)

        self.missing_files.extend(missing_candidates)

    # ------------------------------------------------------------------
    # Page 3
    # ------------------------------------------------------------------
    def _plot_stat_dist(self) -> None:
        """Plot permutation and bootstrap distribution results"""
        dy = 5
        height = 200

        self._draw_plot(
            "stat_distribution_left.svg",
            x=self.canvas_dim["xmid"],
            y=self.move_down_by(self.dict_alph["J"][1], dy),
            x_anchor="center",
            y_anchor="top",
            height=height,
        )
        self._draw_plot(
            "stat_distribution_right.svg",
            x=self.canvas_dim["xmid"],
            y=self.move_down_by(self.dict_alph["J"][1], height + dy * 3),
            x_anchor="center",
            y_anchor="top",
            height=height,
        )

    def _plot_bridged_candidates(self) -> None:
        """Plot bridged channel candidates"""
        self._draw_plot(
            "bridged_candidates.svg",
            x=self.canvas_dim["xmid"],
            y=self.move_up_by(self.dict_alph["L"][1], 5),
            x_anchor="center",
            y_anchor="bottom",
            height=250,
        )
