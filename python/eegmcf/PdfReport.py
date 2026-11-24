"""
Motor imagery PDF report generator

The public behavior is intentionally kept close to the original class:
instantiating MotorImageryPdfReport builds and saves the report by default
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from PIL import Image
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg


if __package__:
    from .RecordingMetadata import PACKAGE_VERSION, clean, known, identity_from_name, parse_recording_time
else:
    from RecordingMetadata import PACKAGE_VERSION, clean, known, identity_from_name, parse_recording_time


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
        resolution: Optional[float] = None,
        age_at_test: str = "N/A",
        save_folder: Optional[PathLike] = None,
        verbose: bool = True,
        auto_generate: bool = True,
        report_title: str = "Command Following Report",
        report_name: Optional[str] = None,
        recording_metadata: Optional[Dict] = None,
        analysis_parameters: Optional[Dict] = None,
    ) -> None:
        """Initialize report paths and metadata used to assemble the PDF"""
        self.plot_folder = Path(plot_folder).expanduser()
        self.helper_folder = Path(helper_folder).expanduser()
        self.save_folder = Path(save_folder).expanduser() if save_folder else self.plot_folder
        self.verbose = verbose
        self.report_title = report_title

        self.show_all_lines = False
        self.show_all_points = False

        self.version = PACKAGE_VERSION
        self.base_name = report_name or self._infer_report_name(self.plot_folder, self.save_folder)
        self.recording_metadata = self._read_json(self.plot_folder.parent / "recording_metadata.json")
        self.recording_metadata.update(recording_metadata or {})
        analysis_metadata = self._read_json(self.plot_folder.parent / "report_metadata.json")
        self.analysis_parameters = dict(analysis_metadata.get("parameters") or {})
        self.analysis_parameters.update(analysis_parameters or {})
        self.analysis_warnings = analysis_metadata.get("warnings") or []
        identity = identity_from_name(self.base_name)
        for key, value in identity.items():
            if not known(self.recording_metadata.get(key)):
                self.recording_metadata[key] = value
        self.sub_name = clean(self.recording_metadata.get("subject_id"))
        self.ses_name = clean(self.recording_metadata.get("session_id"))
        self.run_name = clean(self.recording_metadata.get("run_id"))
        self.recording_id = clean(self.recording_metadata.get("recording_id"))
        self.date_test = "N/A"
        for value in (date_test, self.recording_metadata.get("recording_date"), self.recording_metadata.get("date_test"), self.recording_metadata.get("recording_timestamp")):
            day, _, _ = parse_recording_time(value)
            if known(day):
                self.date_test = day
                break
        self.recording_timestamp = clean(self.recording_metadata.get("recording_timestamp"))
        montage = montage_name if known(montage_name) else clean(self.recording_metadata.get("montage_type"))
        self.montage_name = MONTAGE_LABELS.get(montage, montage)
        self.resolution = self.analysis_parameters.get("psd_resolution_hz", resolution)
        self.age_at_test = clean(age_at_test if known(age_at_test) else self.recording_metadata.get("age_at_test"))
        self.condition = clean(self.recording_metadata.get("condition"))
        self.spatial_filter = clean(self.recording_metadata.get("spatial_filter"))
        self.output_name = f"mcf_report_{self.base_name}.pdf"
        self.output_path = self.save_folder / self.output_name
        self.psd_processing = clean(self.analysis_parameters.get("psd_processing"))
        if self.psd_processing == "one_over_f_subtracted" and self.report_title == "Command Following Report":
            self.report_title = "Command Following Report (1/f-subtracted PSD)"
        if self.psd_processing == "N/A" and "1/f-subtracted" in report_title:
            self.psd_processing = "one_over_f_subtracted"

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
        self.generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.formatted_date = self.generated_at[:10]
        self.report_id = str(uuid4())
        self.missing_files.clear()
        self.drawn_files.clear()
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

        self.c.save()
        self._export_report_generation_metadata()
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
    def _read_json(path: Path) -> Dict:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError):
            return {}

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
            "report_id": self.report_id,
            "package_version": self.version,
            "recording_id": self.recording_id,
            "subject_id": self.sub_name,
            "session_id": self.ses_name,
            "run_id": self.run_name,
            "recording_date": self.date_test,
            "recording_timestamp": self.recording_timestamp,
            "recording_date_source": clean(self.recording_metadata.get("recording_date_source")),
            "generated_at": self.generated_at,
            "spatial_filter": self.spatial_filter,
            "psd_resolution_hz": self.resolution,
            "analysis_parameters": self.analysis_parameters,
            "recording_metadata": self.recording_metadata,
            "analysis_warnings": self.analysis_warnings,
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
                json.dump(payload, fh, indent=2, default=str)
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
        self.c.setTitle(self.report_title)
        self.c.setSubject(f"Recording: {self.recording_id}; report ID: {self.report_id}")

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
    def _fit_text(self, value: str, x: float, y: float, width: float, size: float = 9, bold: bool = False) -> None:
        """Fit text inside its allocated column without clipping adjacent fields"""
        font = "Helvetica-Bold" if bold else "Helvetica"
        text = str(value).replace("\n", " ").replace("\r", " ")
        measured = self.c.stringWidth(text, font, size)
        if measured > width:
            size *= width / measured
        self.c.setFont(font, size)
        self.c.drawString(x, y, text)

    @staticmethod
    def _format_number(value, suffix: str = "") -> str:
        try:
            return f"{float(value):g}{suffix}"
        except (ValueError, TypeError):
            return "N/A"

    def _generate_header(self) -> None:
        """Show recording identity and actual analysis settings on every page"""
        left, top = self.top_left
        right = self.top_right[0]
        width = right - left
        col = width / 3
        self.c.setFillColor(colors.black)
        self._fit_text(self.report_title, left, top - 12, width, 15, True)
        self._fit_text(f"Recording ID: {self.recording_id}", left, top - 24, width, 9, True)

        hp = self.analysis_parameters.get("highpass", self.recording_metadata.get("highpass"))
        lp = self.analysis_parameters.get("lowpass", self.recording_metadata.get("lowpass"))
        filter_range = f"{self._format_number(hp)}-{self._format_number(lp)} Hz" if known(hp) and known(lp) else "N/A"
        sfreq = self.analysis_parameters.get("sfreq", self.recording_metadata.get("sfreq"))
        generated = self.formatted_date + " UTC"
        rows = [
            (f"Subject ID: {self.sub_name}", f"Session: {self.ses_name}   Run: {self.run_name}", f"Generated: {generated}"),
            (f"Recording date: {self.date_test}", f"Age at test: {self.age_at_test}   Condition: {self.condition}", f"Montage: {self.montage_name}"),
            (f"Sampling: {self._format_number(sfreq, ' Hz')}   Filter: {filter_range}", f"Spatial filter: {self.spatial_filter}", f"PSD resolution: {self._format_number(self.resolution, ' Hz/bin')}"),
        ]
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                self._fit_text(value, left + j * col, top - 35 - 10 * i, col - 8, 8.5)
        self.c.setFont("Helvetica", 7)
        self.c.drawRightString(right, top - 65, f"Page {self.c.getPageNumber()} / 3")
        self.draw_hline(left, right, self.dict_alph["J"][1], colors.lightgrey, line_width=0.5, return_coord=False)

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

        # Fit the spectra to three non-overlapping rows without stretching the plots
        # Keep the center-row inner edges aligned with the original guide lines
        outer_height = 218
        middle_height = 230

        self._draw_plot(
            "psd_fc3.svg",
            x=self.move_right_by(self.canvas_dim["xmid"], 30),
            y=self.move_down_by(self.dict_alph["J"][1], 5),
            x_anchor="left",
            y_anchor="top",
            width=narrow_width,
            height=outer_height,
        )
        self._draw_plot(
            "psd_c3.svg",
            x=self.move_left_by(self.canvas_dim["xright"], 20 + width),
            y=self.move_up_by(y_mid, 10),
            x_anchor="left",
            y_anchor="center",
            width=width,
            height=middle_height,
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
                        "height": outer_height,
                    },
                ),
                (
                    "psd_cp3.svg",
                    {
                        "x": self.move_right_by(self.canvas_dim["xmid"], 30),
                        "y": self.move_up_by(self.canvas_dim["ybottom"], 5),
                        "x_anchor": "left",
                        "y_anchor": "bottom",
                        "width": narrow_width,
                        "height": outer_height,
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
            height=outer_height,
        )
        self._draw_plot(
            "psd_c4.svg",
            x=self.move_right_by(self.canvas_dim["xleft"], 20 + width),
            y=self.move_up_by(y_mid, 10),
            x_anchor="right",
            y_anchor="center",
            width=width,
            height=middle_height,
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
                        "height": outer_height,
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
                        "height": outer_height,
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
