"""
MotorImageryPdfReport: class

PDF report generator for motor imagery analysis results
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib import colors
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
import xml.etree.ElementTree as ET

import os
import datetime
from datetime import date
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Literal, Optional, Tuple


XAnchor = Literal["left",     "center", "right"]
YAnchor = Literal["bottom",   "center", "top"  ]
FitMode = Literal["preserve", "stretch"        ]


class MotorImageryPdfReport:
    """Create a PDF report from saved analysis plots and summary metadata"""
    
    def __init__(
        self,
        plot_folder: str,
        helper_folder: str,
        date_test: str = "N/A",
        montage_name: str = "N/A",
        resolution: int = 1,
        age_at_test: str = "N/A",
        save_folder: Optional[str] = None,
    ) -> None:
        """Initialize report paths and metadata used to assemble the PDF"""
        
        self.plot_folder   = plot_folder
        self.helper_folder = helper_folder
        
        self.show_all_lines  = False
        self.show_all_points = False
        
        today = date.today()
        self.formatted_date = today.strftime("%Y-%m-%d")
        
        self.version        = 'N/A'
        
        self.base_name      = self.plot_folder.split("/")[-1]
        self.sub_name       = self.base_name.split("sub-")[1].split("_ses")[0]
        self.ses_name       = self.base_name.split("ses-")[1].split("_")[0]
        
        self.date_test      = date_test
        
        if montage_name   == "DSI_24":
            self.montage_name = "DSI 24 chs"
        elif montage_name == "GTEC_32":
            self.montage_name = "g.Nautilus 32 chs"
        elif montage_name == "EGI_64":
            self.montage_name = "HydroCel GNS 64 chs"
        elif montage_name == "EGI_128":
            self.montage_name = "HydroCel GNS 128 chs"
        else:
            self.montage_name = "N/A"

        self.resolution  = resolution
        self.age_at_test = age_at_test
        self.save_folder = save_folder
        self.output_name = f"mcf_report_{self.base_name}.pdf"
        
        self.spatial_filter = 'Surface Laplacian'
        
        if "HC" in self.sub_name:
            self.condition = "HC"
        elif "TBI" in self.sub_name:
            self.condition = "TBI"
        else:
            self.condition = 'N/A'
        
        
        self._create_canvas()
        
        # Page 1
        self._get_canvas_dims()
        self._create_corner_points()
        self._create_page_sections()
        self._generate_header()
        
        self._plot_timeline()
        self._plot_topoplots()
        self._plot_band_effects()
        self._next_page() # ---------------------
        
        # Page 2
        self._create_corner_points()
        self._create_page_sections()
        self._generate_header()
        
        self._plot_brain()
        self._plot_psds()
        self._next_page() # ---------------------
        
        # Page 3
        self._create_corner_points()
        self._create_page_sections()
        self._generate_header()
        
        self._plot_stat_dist()
        self._plot_bridged_candidates()
        
        
        # Save the canvas
        self.c.save()
        

    # -------------------------------------------------------------------------
    # METHODS FOR STRUCTURAL BUILDING
    # -------------------------------------------------------------------------
    # Functions to find specific coordinates using anchor points and distances (dx or dy) from them
    # Useful for navigating a PDF file, assuming the bottom left corner to be the point (0,0)
    @ staticmethod
    def move_right_by(x: float, dx: float) -> float:
        """Move a point with coordinate x to the right by dx"""
        return x + dx

    @ staticmethod
    def move_left_by(x: float, dx: float) -> float:
        """Move a point with coordinate x to the left by dx"""
        return x - dx

    @ staticmethod
    def move_up_by(y: float, dy: float) -> float:
        """Move a point with coordinate y upward by dy"""
        return y + dy

    @ staticmethod
    def move_down_by(y: float, dy: float) -> float:
        """Move a point with coordinate y downward by dy"""
        return y - dy

    def show_key_point(self, name: str, x: float, y: float, show: bool = True) -> None:
        """Generate anchor point at given coordinates on the PDF"""
        self.dict_alph[name] = [x, y]
        if show:
            self.c.setFont("Helvetica", 16) # Set the font for text display
            self.c.drawString(x, y, name)   # Draw the name at the specified coordinates

    def draw_hline(self, x1: float, x2: float, y: float, color: Any = colors.black, line_width: float = 1.0, return_coord: bool = True) -> Optional[Dict[str, float]]:
        """Draws a horizontal line between x1 and x2 at a fixed y"""
        self.c.setStrokeColor(color)    # Set the line color
        self.c.setLineWidth(line_width) # Set the line width
        self.c.line(x1, y, x2, y)       # Draws a line
        if return_coord:
            return {'x1': x1, 'x2': x2, 'y': y}

    def draw_vline(self, x: float, y1: float, y2: float, color: Any = colors.black, line_width: float = 1.0, return_coord: bool = True) -> Optional[Dict[str, float]]:
        """Draws a vertical line between y1 and y2 at a fixed x"""
        self.c.setStrokeColor(color)    # Set the line color
        self.c.setLineWidth(line_width) # Set the line width
        self.c.line(x, y1, x, y2)       # Draws a line
        if return_coord:
            return {'x': x, 'y1': y1, 'y2': y2}

    def draw_dline(self, x1: float, y1: float, x2: float, y2: float, color: Any = colors.black, line_width: float = 1.0, return_coord: bool = True) -> Optional[Dict[str, float]]:
        """Draws a diagonal line from (x1, y1) to (x2, y2)"""
        self.c.setStrokeColor(color)    # Set the line color
        self.c.setLineWidth(line_width) # Set the line width
        self.c.line(x1, y1, x2, y2)     # Draw the line
        if return_coord:
            return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}

    def _create_canvas(self) -> None:
        """Create a new canvas"""
        # Set up the canvas
        self.c = canvas.Canvas(
            os.path.join(self.save_folder, self.output_name),
            pagesize=letter,
        )
        
    def _next_page(self) -> None:
        """Force document into creating a new page"""
        self.c.showPage()
        
    def _get_canvas_dims(self, margin: int = 15) -> None:
        """Create a dictionary of canvas dimensions"""
        # Canvas dimensions
        width, height = letter  # Get the dimensions of the page
        xmin = 0
        xmax = width  # 612
        xmid = (xmax - xmin) / 2
        ymax = height  # 792
        ymin = 0
        ymid = abs(ymax - ymin) / 2
        
        self.canvas_dim            = {}
        self.canvas_dim['width']   = width
        self.canvas_dim['height']  = height
        self.canvas_dim['xleft']   = xmin
        self.canvas_dim['xright']  = xmax
        self.canvas_dim['xmid']    = xmid
        self.canvas_dim['ytop']    = ymax
        self.canvas_dim['ybottom'] = ymin
        self.canvas_dim['ymid']    = ymid
        self.canvas_dim['margin']  = margin

    def _create_corner_points(self) -> None:
        """Create corner points with appropriate coordinates"""
        # Coordinates of the corners
        self.top_left = (
            self.move_right_by(self.canvas_dim['xleft'], self.canvas_dim['margin']),
            self.move_down_by( self.canvas_dim['ytop'], self.canvas_dim['margin']),
        )
        self.top_right = (
            self.move_left_by( self.canvas_dim['xright'], self.canvas_dim['margin']),
            self.move_down_by( self.canvas_dim['ytop'], self.canvas_dim['margin']),
        )
        self.bottom_right = (
            self.move_left_by( self.canvas_dim['xright'], self.canvas_dim['margin']),
            self.move_up_by(   self.canvas_dim['ybottom'], self.canvas_dim['margin']),
        )
        self.bottom_left = (
            self.move_right_by(self.canvas_dim['xleft'], self.canvas_dim['margin']),
            self.move_up_by(   self.canvas_dim['ybottom'], self.canvas_dim['margin']),
        )

        # Canvas dimensions after taking into account the distance from the edges (delta)
        self.page_width  = self.top_right[0] - self.top_left[0]
        self.page_height = self.top_right[1] - self.bottom_right[1]

    def _create_page_sections(self) -> None:
        """Create points and lines to split a page into needed sections"""
        # Break down the page into 20 units
        h_ = self.page_height / 20
        # Define a space 2 units from top of page (splits upper section from header)
        line_below_header = 2 * h_
        # Define a space 11 units from top of page (splits lower section from upper section)
        line_below_upper_section = 11 * h_

        # Create dictionary for anchor points
        self.dict_alph = {}
        # Create anchor points at the corners, clockwise starting from top left
        self.show_key_point(
            "A", x=self.top_left[0],  y=self.top_left[1],  show=self.show_all_points
        )  # Top left (A)
        self.show_key_point(
            "B", x=self.top_right[0], y=self.top_right[1], show=self.show_all_points
        )  # Top right (B)
        self.show_key_point(
            "C",
            x=self.bottom_right[0],
            y=self.bottom_right[1],
            show=self.show_all_points,
        )  # Bottom right (C)
        self.show_key_point(
            "D", x=self.bottom_left[0], y=self.bottom_left[1], show=self.show_all_points
        )  # Bottom left (D)

        # Create anchor points between HEADER and UPPER SECTION
        self.show_key_point(
            "E",
            x=self.top_left[0],
            y=self.move_down_by(self.top_left[1], line_below_header),
            show=self.show_all_points,
        )
        self.show_key_point(
            "F",
            x=self.top_right[0],
            y=self.move_down_by(self.top_left[1], line_below_header),
            show=self.show_all_points,
        )
        self.show_key_point(
            "I",
            x=self.move_right_by(self.page_width / 2, self.top_left[0]),
            y=self.top_right[1],
            show=self.show_all_points,
        )
        self.show_key_point(
            "J",
            x=self.move_right_by(self.page_width / 2, self.top_left[0]),
            y=self.move_down_by(self.top_left[1], line_below_header),
            show=self.show_all_points,
        )
        
        # Create anchor points between UPPER SECTION and LOWER SECTION
        self.show_key_point(
            "G",
            x=self.top_left[0],
            y=self.move_down_by(self.top_left[1], line_below_upper_section),
            show=self.show_all_points,
        )
        self.show_key_point(
            "H",
            x=self.top_right[0],
            y=self.move_down_by(self.top_left[1], line_below_upper_section),
            show=self.show_all_points,
        )
        self.show_key_point(
            "K",
            x=self.move_right_by(self.page_width / 2, self.top_left[0]),
            y=self.move_down_by(self.top_left[1], line_below_upper_section),
            show=self.show_all_points,
        )
        self.show_key_point(
            "L",
            x=self.move_right_by(self.page_width / 2, self.top_left[0]),
            y=self.bottom_right[1],
            show=self.show_all_points,
        )

        # Draw lines connecting the corners
        if self.show_all_lines:
            # AB (top horizontal)
            self.draw_hline(
                x1=self.top_left[0],
                x2=self.top_right[0],
                y=self.top_left[1],
                color=colors.black,
            )
            # BC (right vertical)
            self.draw_vline(
                x=self.top_right[0],
                y1=self.top_right[1],
                y2=self.bottom_right[1],
                color=colors.red,
            )
            # CD (bottom horizontal)
            self.draw_hline(
                x1=self.bottom_left[0],
                x2=self.bottom_right[0],
                y=self.bottom_left[1],
                color=colors.blue,
            )
            # DA (left vertical)
            self.draw_vline(
                x=self.bottom_left[0],
                y1=self.top_left[1],
                y2=self.bottom_left[1],
                color=colors.green,
            )
            # Split page vertically into two regions
            # V line
            self.draw_vline(
                x=self.canvas_dim['xmid'],
                y1=self.top_right[1],
                y2=self.bottom_right[1],
                color=colors.red,
            )
            # Create the boundary between UPPER SECTION and LOWER SECTION, 11 unit from the top
            # H line
            self.draw_hline(
                x1=self.top_left[0],
                x2=self.top_right[0],
                y=self.move_down_by(self.top_left[1], line_below_upper_section),
                color=colors.grey,
            )
            # Create the boundary between HEADER and UPPER SECTION, 2 unit from the top
            # H line
            self.draw_hline(
                x1=self.top_left[0],
                x2=self.top_right[0],
                y=self.move_down_by(self.top_left[1], line_below_header),
                color=colors.black,
            )


    # -------------------------------------------------------------------------
    # METHODS FOR PLOTTING
    # -------------------------------------------------------------------------
    def _write_text(self, text_list: List[str], x: float, y: float, font_size: int = 12, bold_flags: Optional[List[bool]] = None, align: str = "center") -> None:
        """Writes a list of texts at a specified position with optional bold and alignment

        Args:
            text_list (list): List of texts to be displayed
            x (float): X-coordinate for alignment, bottom left corner
            y (float): Y-coordinate where texts start, bottom left corner
            font_size (int): Font size of the text
            bold_flags (list, optional): List of boolean values indicating if corresponding text is bold
            align (str): Alignment of text; options are 'center', 'left', 'right'
        """
        if bold_flags is None:
            bold_flags = [False] * len(text_list)

        total_width = 0
        widths = []
        fonts = []

        for text, is_bold in zip(text_list, bold_flags):
            font = "Helvetica-Bold" if is_bold else "Helvetica"
            text_width = self.c.stringWidth(text, font, font_size)
            widths.append(text_width)
            fonts.append(font)
            total_width += text_width

        if align == "center":
            current_x = x - total_width / 2
        elif align == "left":
            current_x = x
        elif align == "right":
            current_x = x - total_width

        for text, width, font in zip(text_list, widths, fonts):
            self.c.setFont(font, font_size)
            self.c.drawString(current_x, y, text)
            current_x += width

    def _is_svg(self, path: str) -> bool:
        """Return True if the file looks like an SVG"""
        return os.path.splitext(path.lower())[1] == ".svg"

    def _get_raster_size(self, path: str) -> Tuple[float, float]:
        """
        Return (width, height) in *pixels* from PIL
        Note: ReportLab uses points, but for aspect-ratio math, units cancel out,
        so pixels are fine as an intrinsic size reference
        """
        with Image.open(path) as img:
            w, h = img.size
        return float(w), float(h)

    def _get_svg_size(self, path: str) -> Tuple[float, float]:
        """Return (width, height) in ReportLab drawing units (points-ish)"""
        drawing = svg2rlg(path)
        return float(drawing.width), float(drawing.height)

    def _resolve_size_preserve_ratio(
        self,
        orig_w: float,
        orig_h: float,
        width: Optional[float],
        height: Optional[float],
    ) -> Tuple[float, float]:
        """
        Resolve final (w, h) preserving aspect ratio
        Rules:
        - width only  -> compute height
        - height only -> compute width
        - both        -> fit inside (width, height) preserving ratio
        - neither     -> use original
        """
        if orig_w <= 0 or orig_h <= 0:
            raise ValueError("Invalid intrinsic size for image (non-positive width/height).")

        if width is None and height is None:
            return orig_w, orig_h

        if width is not None and height is None:
            scale = width / orig_w
            return width, orig_h * scale

        if height is not None and width is None:
            scale = height / orig_h
            return orig_w * scale, height

        # Both provided: fit within bounding box
        scale = min(width / orig_w, height / orig_h)
        return orig_w * scale, orig_h * scale

    def _anchor_to_bottom_left(
        self,
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
        image_path: str,
        *,
        x: float,
        y: float,
        x_anchor: XAnchor = "left",
        y_anchor: YAnchor = "bottom",
        width: Optional[float] = None,
        height: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Draw a figure whose position is defined by an anchor point (x, y)

        Examples:
        - x_anchor='center' means the *center* of the figure touches x
        - y_anchor='top'    means the *top edge* of the figure touches y

        If only one of width/height is provided, the other is inferred by ratio
        If both are provided, the figure is fit inside that box preserving ratio

        Returns a dict with final placement info (useful for debugging/layout)
        """
        if self._is_svg(image_path):
            orig_w, orig_h = self._get_svg_size(image_path)
        else:
            orig_w, orig_h = self._get_raster_size(image_path)

        final_w, final_h = self._resolve_size_preserve_ratio(
            orig_w=orig_w,
            orig_h=orig_h,
            width=width,
            height=height,
        )

        x_left, y_bottom = self._anchor_to_bottom_left(
            x=x, y=y, w=final_w, h=final_h, x_anchor=x_anchor, y_anchor=y_anchor
        )

        # Actually draw
        if self._is_svg(image_path):
            drawing = svg2rlg(image_path)
            scale = final_w / float(drawing.width)  # same as final_h/drawing.height (ratio preserved)
            drawing.scale(scale, scale)
            renderPDF.draw(drawing, self.c, x_left, y_bottom)
        else:
            self.c.drawImage(image_path, x_left, y_bottom, width=final_w, height=final_h)

        return {
            "xleft": x_left,
            "y_bottom": y_bottom,
            "width": final_w,
            "height": final_h,
        }


    # -------------------------------------------------------------------------
    # METHODS FOR BUILDING PDF CONTENT
    # -------------------------------------------------------------------------
    # Page 1
    def _generate_header(self) -> None:
        """Populate header with text"""
        # Write the HEADER with subject and measurement information
        y0 = 15
        dx = 10
        dy = 15
        # Title
        self._write_text(
            ["Command Following Report"],
            self.canvas_dim["xmid"],
            self.move_down_by(self.dict_alph["I"][1], y0),
            font_size=17,
            bold_flags=[True],
            align="center",
        )
        # Upper right corner
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

        # Left section
        font_size = 11
        height = self.move_down_by(self.dict_alph["I"][1], 15) - self.move_up_by(
            self.dict_alph["J"][1], 5
        )
        dy = height / 3
        # Left column
        self._write_text(
            ["Subject: ", f"{self.sub_name}"],
            self.canvas_dim['xmid'] * 2 / 3 - dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="right",
        )
        self._write_text(
            ["Session: ", f"{self.ses_name}"],
            self.canvas_dim['xmid'] * 2 / 3 - dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + 2 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="right",
        )
        self._write_text(
            ["Date of assessment: ", f"{self.date_test}"],
            self.canvas_dim['xmid'] * 2 / 3 - dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + 3 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="right",
        )

        # Center column
        self._write_text(
            ["Age at test: ", f"{self.age_at_test} "],
            self.canvas_dim['xmid'],
            self.move_down_by(self.dict_alph["I"][1], y0 + dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="center",
        )
        self._write_text(
            ["Condition: ", f"{self.condition} "],
            self.canvas_dim['xmid'],
            self.move_down_by(self.dict_alph["I"][1], y0 + 2 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="center",
        )
        self._write_text(
            ["Montage: ", f"{self.montage_name}"],
            self.canvas_dim['xmid'],
            self.move_down_by(self.dict_alph["I"][1], y0 + 3 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="center",
        )

        # Right column
        self._write_text(
            ["Resolution: ", f"{self.resolution} Hz"],
            self.canvas_dim['xmid'] * 2 * 2 / 3 + dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="left",
        )
        self._write_text(
            ["Spatial filter: ", f"{self.spatial_filter}"],
            self.canvas_dim['xmid'] * 2 * 2 / 3 + dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + 2 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="left",
        )
        self._write_text(
            ["Filter: ", f"1-40 Hz"],
            self.canvas_dim['xmid'] * 2 * 2 / 3 + dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + 3 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="left",
        )

    def _plot_timeline(self) -> None:
        """Plot paradigm timeline"""
        dy         = 5
        width      = 500
        image_path = os.path.join(self.plot_folder, "paradigm_timeline.svg")
        if Path(image_path).exists():
            self.draw_figure_at(
                image_path,
                x=self.canvas_dim["xmid"],
                y=self.move_down_by(self.dict_alph["J"][1], dy),
                x_anchor="center",
                y_anchor="top",
                width=width,
            )

    def _plot_topoplots(self) -> None:
        """Plot band topoplots with signed r^2 coefficients"""
        dy     = 130
        height = 300
        
        # Import r2 topoplots 
        topoplot = os.path.join(self.plot_folder, "topoplot.svg")
        if Path(topoplot).exists():
            self.draw_figure_at(
                topoplot,
                x=self.canvas_dim["xmid"],
                y=self.move_down_by(self.dict_alph["J"][1], dy),
                x_anchor="center",
                y_anchor="top",
                height=height,
            )

        # Left p-value
        p_left = os.path.join(self.plot_folder, "pvalue_left.svg")
        if Path(p_left).exists():
            self.draw_figure_at(
                p_left,
                x=self.move_right_by(self.canvas_dim["xleft"], 55),
                y=self.move_down_by(self.dict_alph["J"][1], dy + 30),
                x_anchor="left",
                y_anchor="top",
                height=int(height * 0.95),
            )

        # Right p-value
        p_right = os.path.join(self.plot_folder, "pvalue_right.svg")
        if Path(p_right).exists():
            self.draw_figure_at(
                p_right,
                x=self.move_left_by(self.canvas_dim["xright"], 55),
                y=self.move_down_by(self.dict_alph["J"][1], dy + 30),
                x_anchor="right",
                y_anchor="top",
                height=int(height * 0.93),
            )

    def _plot_band_effects(self) -> None:
        """Plot bootstrap CI per band"""
        dy         = 5
        width      = 450
        image_path = os.path.join(self.plot_folder, "band_effect.svg")
        if Path(image_path).exists():
            self.draw_figure_at(
                image_path,
                x=self.canvas_dim["xmid"],
                y=self.move_up_by(self.dict_alph["L"][1], dy),
                x_anchor="center",
                y_anchor="bottom",
                width=width,
            )

    # Page 2 - psds
    def _plot_brain(self) -> None:
        """Plot brain icon with arrows"""
        
        y_mid = abs(self.dict_alph['J'][1] - self.dict_alph['L'][1]) / 2
        width      = 85
        image_path = os.path.join(self.helper_folder, "brain_c3c4p3p4.svg")
        if Path(image_path).exists():
            self.draw_figure_at(
                image_path,
                x=self.canvas_dim["xmid"],
                y=self.move_up_by(y_mid, 10),
                x_anchor="center",
                y_anchor="center",
                width=width,
            )
        
        # Arrow to top right
        coord = self.draw_vline(
            x     = self.move_right_by(self.canvas_dim['xmid'], 16),
            y1    = self.move_up_by(y_mid, 34),
            y2    = self.move_up_by(y_mid, 68),
            color = colors.blue,
        )
        self.draw_dline(
            x1    = coord['x'],
            x2    = self.move_left_by(coord['x'], 90),
            y1    = coord['y2'],
            y2    = self.move_up_by(coord['y2'], 70),
            color = colors.blue,
            return_coord=False,
        )
        # Arrow to top left
        coord = self.draw_vline(
            x     = self.move_left_by(self.canvas_dim['xmid'], 13),
            y1    = self.move_up_by(y_mid, 34),
            y2    = self.move_up_by(y_mid, 68),
            color = colors.red,
        )
        self.draw_dline(
            x1    = coord['x'],
            x2    = self.move_right_by(coord['x'], 90),
            y1    = coord['y2'],
            y2    = self.move_up_by(coord['y2'], 70),
            color = colors.red,
            return_coord=False,
        )
        
        # Arrow to mid right
        coord = self.draw_dline(
            x1    = self.move_right_by(self.canvas_dim['xmid'], 20),
            x2    = self.move_right_by(self.canvas_dim['xmid'], 12.5),
            y1    = self.move_up_by(y_mid, 7.5),
            y2    = y_mid,
            color = colors.blue,
        )
        self.draw_hline(
            x1    = coord['x2'],
            x2    = self.move_left_by(coord['x2'], 100),
            y     = coord['y2'],
            color = colors.blue,
            return_coord=False,
        )
        # Arrow to mid left
        coord = self.draw_dline(
            x1    = self.move_left_by(self.canvas_dim['xmid'], 17),
            x2    = self.move_left_by(self.canvas_dim['xmid'], 9.5),
            y1    = self.move_up_by(y_mid, 12),
            y2    = self.move_up_by(y_mid, 19.5),
            color = colors.red,
        )
        self.draw_hline(
            x1    = coord['x2'],
            x2    = self.move_right_by(coord['x2'], 100),
            y     = coord['y2'],
            color = colors.red,
            return_coord=False,
        )
        
        # Arrow to bottom right
        coord = self.draw_vline(
            x     = self.move_right_by(self.canvas_dim['xmid'], 16),
            y1    = self.move_down_by(y_mid, 11),
            y2    = self.move_down_by(y_mid, 45),
            color = colors.blue,
        )
        self.draw_dline(
            x1    = coord['x'],
            x2    = self.move_left_by(coord['x'], 90),
            y1    = coord['y2'],
            y2    = self.move_down_by(coord['y2'], 70),
            color = colors.blue,
            return_coord=False,
        )
        # Arrow to bottom left
        coord = self.draw_vline(
            x     = self.move_left_by(self.canvas_dim['xmid'], 13),
            y1    = self.move_down_by(y_mid, 11),
            y2    = self.move_down_by(y_mid, 45),
            color = colors.red,
        )
        self.draw_dline(
            x1    = coord['x'],
            x2    = self.move_right_by(coord['x'], 90),
            y1    = coord['y2'],
            y2    = self.move_down_by(coord['y2'], 70),
            color = colors.red,
            return_coord=False,
        )

    def _plot_psds(self) -> None:
        """Plot PSDs of selected channels"""  
        
        y_mid = abs(self.dict_alph['J'][1] - self.dict_alph['L'][1]) / 2
        width = 200
        
        # Right side
        image_path = os.path.join(self.plot_folder, "psd_fc3.svg")
        if Path(image_path).exists():
            self.draw_figure_at(
                image_path,
                x=self.move_right_by(self.canvas_dim["xmid"], 30),
                y=self.move_down_by(self.dict_alph['J'][1], 5),
                x_anchor="left",
                y_anchor="top",
                width=int(width * 0.85),
            )
        
        image_path = os.path.join(self.plot_folder, "psd_c3.svg")
        if Path(image_path).exists():
            self.draw_figure_at(
                image_path,
                x=self.move_left_by(self.canvas_dim["xright"], 20),
                y=self.move_up_by(y_mid, 10),
                x_anchor="right",
                y_anchor="center",
                width=width,
            )
        
        image_path = os.path.join(self.plot_folder, "psd_p3.svg")
        if Path(image_path).exists():
            self.draw_figure_at(
                image_path,
                x=self.move_right_by(self.canvas_dim["xmid"], 30),
                y=self.move_up_by(self.canvas_dim['ybottom'], 5),
                x_anchor="left",
                y_anchor="bottom",
                width=int(width * 0.85),
            )
        else:
            image_path = os.path.join(self.plot_folder, "psd_cp3.svg")
            if Path(image_path).exists():
                self.draw_figure_at(
                    image_path,
                    x=self.move_left_by(self.canvas_dim["xmid"], 30),
                    y=self.move_up_by(self.canvas_dim['ybottom'], 5),
                    x_anchor="right",
                    y_anchor="bottom",
                    width=int(width * 0.85),
                )
        
        # Left side
        image_path = os.path.join(self.plot_folder, "psd_fc4.svg")
        if Path(image_path).exists():
            self.draw_figure_at(
                image_path,
                x=self.move_left_by(self.canvas_dim["xmid"], 30),
                y=self.move_down_by(self.dict_alph['J'][1], 5),
                x_anchor="right",
                y_anchor="top",
                width=int(width * 0.85),
            )
        
        image_path = os.path.join(self.plot_folder, "psd_c4.svg")
        if Path(image_path).exists():
            self.draw_figure_at(
                image_path,
                x=self.move_right_by(self.canvas_dim["xleft"], 20),
                y=self.move_up_by(y_mid, 10),
                x_anchor="left",
                y_anchor="center",
                width=width,
            )
        
        image_path = os.path.join(self.plot_folder, "psd_p4.svg")
        if Path(image_path).exists():
            self.draw_figure_at(
                image_path,
                x=self.move_left_by(self.canvas_dim["xmid"], 30),
                y=self.move_up_by(self.canvas_dim['ybottom'], 5),
                x_anchor="right",
                y_anchor="bottom",
                width=int(width * 0.85),
            )
        else:
            image_path = os.path.join(self.plot_folder, "psd_cp4.svg")
            if Path(image_path).exists():
                self.draw_figure_at(
                    image_path,
                    x=self.move_left_by(self.canvas_dim["xmid"], 30),
                    y=self.move_up_by(self.canvas_dim['ybottom'], 5),
                    x_anchor="right",
                    y_anchor="bottom",
                    width=int(width * 0.85),
                )
            
    # Page 3 - stats
    def _plot_stat_dist(self) -> None:
        """Plot permutation and bootstrap distribution results"""
        
        dy     = 5
        height = 200
        
        # Import left vs rest
        stat_left = os.path.join(self.plot_folder, "stat_distribution_left.svg")
        if Path(stat_left).exists():
            self.draw_figure_at(
                stat_left,
                x=self.canvas_dim["xmid"],
                y=self.move_down_by(self.dict_alph["J"][1], dy),
                x_anchor="center",
                y_anchor="top",
                height=height,
            )
        
        # Import right vs rest
        stat_right = os.path.join(self.plot_folder, "stat_distribution_right.svg")
        if Path(stat_right).exists():
            self.draw_figure_at(
                stat_right,
                x=self.canvas_dim["xmid"],
                y=self.move_down_by(self.dict_alph["J"][1], height + dy * 3),
                x_anchor="center",
                y_anchor="top",
                height=height,
            )
    
    def _plot_bridged_candidates(self) -> None:
        """Plot bridged candidates"""
        
        dy     = 5
        height = 250
        
        bridged = os.path.join(self.plot_folder, "bridged_candidates.svg")
        if Path(bridged).exists():
            self.draw_figure_at(
                bridged,
                x=self.canvas_dim["xmid"],
                y=self.move_up_by(self.dict_alph["L"][1], dy),
                x_anchor="center",
                y_anchor="bottom",
                height=height,
            )
        

