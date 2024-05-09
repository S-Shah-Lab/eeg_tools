from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
import xml.etree.ElementTree as ET

import datetime
from datetime import date
from PIL import Image


class generate_pdf:
    def __init__(self, plot_folder, montage_name, resolution):
        # Import folder and set up information regarding the subject
        self.plot_folder = plot_folder
        self.montage_name = montage_name
        self.resolution = resolution

        self.base_name = self.plot_folder.split("/")[-2]
        # Subject's name
        self.sub_name = self.base_name.split("sub-")[1].split("_ses")[0]
        # Subject's session
        self.ses_name = self.base_name.split("ses-")[1].split("_")[0]

        # Subject's condition
        if "HC" in self.sub_name:
            self.condition = "HC"
        elif "BI" in self.sub_name:
            self.condition = "TBI"
        else:
            self.condition = "N/A"

        # Subject's age
        self.sub_age = self.prompt_for_age()

        # Get today's date
        today = date.today()
        self.formatted_date = today.strftime("%Y-%m-%d")

        # Measurement date
        self.date_test = self.prompt_for_date()

        # Montage (they are now provided during class initialization)
        # self.montage_name = str(input("Enter montage name [DSI 24, GTEC 32, EGI 128]: "))

        # Resolution (they are now provided during class initialization)
        # self.resolution = str(input("Enter the PSD resolution (e.g. 1): "))

        # Set up the canvas
        self.c = canvas.Canvas(
            f"{self.plot_folder}MI_report_sub-{self.sub_name}_ses-{self.ses_name}.pdf",
            pagesize=letter,
        )
        # Canvas dimensions
        width, height = letter  # Get the dimensions of the page
        xmin = 0
        xmax = width  # 612
        self.xmid = (xmax - xmin) / 2
        ymin = height  # 792
        ymax = 0
        # Distance from the canvas edges
        delta = 15

        # Coordinates of the corners
        self.top_left = (
            self.move_right_by(xmin, delta),
            self.move_down_by(ymin, delta),
        )
        self.top_right = (
            self.move_left_by(xmax, delta),
            self.move_down_by(ymin, delta),
        )
        self.bottom_right = (
            self.move_left_by(xmax, delta),
            self.move_up_by(ymax, delta),
        )
        self.bottom_left = (
            self.move_right_by(xmin, delta),
            self.move_up_by(ymax, delta),
        )

        # Canvas dimensions after taking into account the distance from the edges (delta)
        self.page_width = self.top_right[0] - self.top_left[0]
        self.page_height = self.top_right[1] - self.bottom_right[1]

        # Create dictionary for anchor points
        self.dict_alph = {}

        # Create anchor points at the corners, clockwise starting from top left
        self.show_key_point(
            "A", x=self.top_left[0], y=self.top_left[1], show=False
        )  # Top left (A)
        self.show_key_point(
            "B", x=self.top_right[0], y=self.top_right[1], show=False
        )  # Top right (B)
        self.show_key_point(
            "C", x=self.bottom_right[0], y=self.bottom_right[1], show=False
        )  # Bottom right (C)
        self.show_key_point(
            "D", x=self.bottom_left[0], y=self.bottom_left[1], show=False
        )  # Bottom left (D)

        # Draw lines connecting the corners
        # AB
        # self.draw_hline(x1=self.top_left[0], x2=self.top_right[0], y=self.top_left[1], color=colors.black)

        # BC
        # self.draw_vline(x=self.top_right[0], y1=self.top_right[1], y2=self.bottom_right[1], color=colors.red)

        # CD
        # self.draw_hline(x1=self.bottom_left[0], x2=self.bottom_right[0], y=self.bottom_left[1], color=colors.blue)

        # DA
        # self.draw_vline(x=self.bottom_left[0], y1=self.top_left[1], y2=self.bottom_left[1], color=colors.green)

        # Break down the page into 20 units
        h_ = self.page_height / 20

        # Create the boundary between HEADER and UPPER SECTION, 2 unit from the top
        # H line
        gap_from_top = 2 * h_
        # self.draw_hline(x1=self.top_left[0], x2=self.top_right[0], y=self.move_down_by(self.top_left[1], gap_from_top), color=colors.grey)
        self.draw_hline(
            x1=self.top_left[0],
            x2=self.top_right[0],
            y=self.move_down_by(self.top_left[1], gap_from_top),
            color=colors.black,
        )
        # Create additional anchor points
        self.show_key_point(
            "E",
            x=self.top_left[0],
            y=self.move_down_by(self.top_left[1], gap_from_top),
            show=False,
        )
        self.show_key_point(
            "F",
            x=self.top_right[0],
            y=self.move_down_by(self.top_left[1], gap_from_top),
            show=False,
        )
        self.show_key_point(
            "I",
            x=self.move_right_by(self.page_width / 2, self.top_left[0]),
            y=self.top_right[1],
            show=False,
        )
        self.show_key_point(
            "J",
            x=self.move_right_by(self.page_width / 2, self.top_left[0]),
            y=self.move_down_by(self.top_left[1], gap_from_top),
            show=False,
        )

        # Create the boundary between UPPER SECTION and LOWER SECTION, 11 unit from the top
        # H line
        gap_from_top = 11 * h_
        # self.draw_hline(x1=self.top_left[0], x2=self.top_right[0], y=self.move_down_by(top_left[1], gap_from_top), color=colors.grey)
        self.show_key_point(
            "G",
            x=self.top_left[0],
            y=self.move_down_by(self.top_left[1], gap_from_top),
            show=False,
        )
        self.show_key_point(
            "H",
            x=self.top_right[0],
            y=self.move_down_by(self.top_left[1], gap_from_top),
            show=False,
        )
        self.show_key_point(
            "K",
            x=self.move_right_by(self.page_width / 2, self.top_left[0]),
            y=self.move_down_by(self.top_left[1], gap_from_top),
            show=False,
        )
        self.show_key_point(
            "L",
            x=self.move_right_by(self.page_width / 2, self.top_left[0]),
            y=self.bottom_right[1],
            show=False,
        )

        # Mid line
        # V line
        # self.draw_vline(self.c, x=self.move_right_by(self.xmid, self.top_left[0]), y1=self.top_right[1], y2=self.bottom_right[1], color=colors.red)

        # Write the HEADER with subject and measurement information
        y0 = 15
        dx = 10
        dy = 15
        self.write_text(
            ["Motor Imagery Report"],
            self.xmid,
            self.move_down_by(self.dict_alph["I"][1], y0),
            font_size=17,
            bold_flags=[True],
            align="center",
        )
        self.write_text(
            [f"Report: {self.formatted_date}"],
            self.top_right[0],
            self.move_down_by(self.dict_alph["I"][1], y0),
            font_size=11,
            bold_flags=[False],
            align="right",
        )

        font_size = 11
        height = self.move_down_by(self.dict_alph["I"][1], 15) - self.move_up_by(
            self.dict_alph["J"][1], 5
        )
        dy = height / 3
        # Left side
        self.write_text(
            ["Subject: ", f"{self.sub_name}"],
            self.xmid - dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="right",
        )
        self.write_text(
            ["Session: ", f"{self.ses_name}"],
            self.xmid - dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + 2 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="right",
        )
        self.write_text(
            ["Date of Assessment: ", f"{self.date_test}"],
            self.xmid - dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + 3 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="right",
        )
        # Right side
        self.write_text(
            ["Age: ", f"{self.sub_age} "],
            self.xmid + dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="left",
        )
        self.write_text(
            ["Condition: ", f"{self.condition} "],
            self.xmid + dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + 2 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="left",
        )
        self.write_text(
            ["Montage (Resolution): ", f"{self.montage_name} ({self.resolution} Hz)"],
            self.xmid + dx,
            self.move_down_by(self.dict_alph["I"][1], y0 + 3 * dy),
            font_size=font_size,
            bold_flags=[False, True],
            align="left",
        )

        # Import r2 topoplots to the UPPER SECTION
        dy = 5
        height = self.move_down_by(self.dict_alph["E"][1], dy) - self.dict_alph["G"][1]
        width = self.image_ratio(
            f"{self.plot_folder}topoR2_{self.resolution}_Hz.png", new_height=height
        )
        x_left = self.page_width / 2 - width / 2 + self.top_left[0]
        x_right = self.page_width / 2 + width / 2 + self.top_left[0]
        image_path = f"{self.plot_folder}topoR2_{self.resolution}_Hz.svg"
        self.draw_svg_image(
            image_path,
            x=x_left,
            y=self.move_up_by(self.dict_alph["G"][1], 20),
            width=None,
            height=height,
        )

        # Import p-val for left hand to the UPPER SECTION
        width = self.image_ratio(
            f"{plot_folder}pVal_left_{self.resolution}_Hz.png", new_height=height
        )
        image_path = f"{plot_folder}pVal_left_1_Hz.svg"
        self.draw_svg_image(
            image_path,
            x=x_left - width,
            y=self.dict_alph["K"][1],
            width=width,
            height=None,
        )

        # Import p-val for right hand to the UPPER SECTION
        width = self.image_ratio(
            f"{plot_folder}pVal_right_{self.resolution}_Hz.png", new_height=height
        )
        image_path = f"{plot_folder}pVal_right_{self.resolution}_Hz.svg"
        self.draw_svg_image(
            image_path, x=x_right, y=self.dict_alph["K"][1], width=width, height=None
        )

        # Import c3 psds to the LOWER SECTION
        image_path = f"{plot_folder}c3_PSDs_{self.resolution}_Hz.svg"
        height = self.dict_alph["G"][1] - self.dict_alph["D"][1]
        self.draw_svg_image(
            image_path,
            x=self.move_right_by(self.dict_alph["D"][0], 15),
            y=self.move_up_by(self.dict_alph["L"][1], 5),
            width=None,
            height=height,
        )

        # Import c4 psds to the LOWER SECTION
        image_path = f"{plot_folder}c4_PSDs_{self.resolution}_Hz.svg"
        height = self.dict_alph["G"][1] - self.dict_alph["D"][1]
        self.draw_svg_image(
            image_path,
            x=self.move_right_by(self.dict_alph["L"][0], 15),
            y=self.move_up_by(self.dict_alph["L"][1], 5),
            width=None,
            height=height,
        )

        # Save the canvas
        self.c.save()

    def prompt_for_date(
        self,
    ):
        """Asks the user a date in the format yyyy-mm-dd

        Args:
                None

        Returns:
                str: Date in the format yyyy-mm-dd
        """
        while True:
            user_input = input("Enter date of the test [yyyy-mm-dd] (0 if not known): ")
            try:
                if user_input == "0":
                    print("Date not known.")
                    return "N/A"
                else:
                    # Attempt to parse the date string into a datetime object.
                    parsed_date = datetime.datetime.strptime(user_input, "%Y-%m-%d")
                    return parsed_date.strftime(
                        "%Y-%m-%d"
                    )  # Returns the date in yyyy-mm-dd format
            except ValueError:
                # If there's an error in parsing, it means the format is incorrect.
                print("Invalid date format. Enter date of the test [yyyy-mm-dd]: ")

    def prompt_for_age(
        self,
    ):
        """Asks the user the subject's age

        Args:
                None

        Returns:
                int: Age of the subject
        """
        while True:
            user_input = input("Enter the subject's age (0 if not known): ")
            try:
                # Attempt to convert the input to an integer.
                age = int(user_input)
                # Additional check: ensure age is within a reasonable range
                if age == 0:
                    print("Age not known.")
                    return "N/A"
                elif age < 0 or age > 120:
                    print("Invalid age entered. Please enter the subject's age.")
                else:
                    return age  # Returns the age as an integer
            except ValueError:
                # If conversion to integer fails, it means the input was not a valid integer.
                print("Invalid input. Please enter the subject's age.")

    # Functions to find specific coordinates using anchor points and distances (dx or dy) from them.
    # Useful for navigating a PDF file, assuming the bottom left corner to be the point (0,0).
    def move_right_by(self, x, dx):
        """Move a point to the right by a given distance.

        Args:
            x (float): The current x-coordinate.
            dx (float): The distance to move to the right.

        Returns:
            float: The new x-coordinate after moving right.
        """
        return x + dx

    def move_left_by(self, x, dx):
        """Move a point to the left by a given distance.

        Args:
            x (float): The current x-coordinate.
            dx (float): The distance to move to the left.

        Returns:
            float: The new x-coordinate after moving left.
        """
        return x - dx

    def move_up_by(self, y, dy):
        """Move a point upward by a given distance.

        Args:
            y (float): The current y-coordinate.
            dy (float): The distance to move upward.

        Returns:
            float: The new y-coordinate after moving up.
        """
        return y + dy

    def move_down_by(self, y, dy):
        """Move a point downward by a given distance.

        Args:
            y (float): The current y-coordinate.
            dy (float): The distance to move downward.

        Returns:
            float: The new y-coordinate after moving down.
        """
        return y - dy

    def show_key_point(self, name, x, y, show=True):
        """Generate anchor point with its name at given coordinates on the PDF. The point can be displayed as well.

        Args:
            name (str): The name of the key point.
            x (float): x-coordinate of the key point.
            y (float): y-coordinate of the key point.
            show (bool): Flag to either show or not show the point visually on the PDF.

        Modifies:
            dict_alph (dict): Dictionary to store key point names and coordinates.
        """
        self.dict_alph[name] = [x, y]
        if show:
            self.c.setFont("Helvetica", 16)  # Set the font for text display.
            self.c.drawString(x, y, name)  # Draw the name at the specified coordinates.

    def draw_hline(self, x1, x2, y, color=colors.black):
        """Draws a horizontal line between two x-coordinates at a fixed y-coordinate.

        Args:
            x1 (float): The starting x-coordinate of the line.
            x2 (float): The ending x-coordinate of the line.
            y (float): The y-coordinate at which the line is drawn.
            color: The color of the line.

        Modifies:
            c (canvas): The PDF canvas.
        """
        self.c.setStrokeColor(color)  # Set the line color.
        self.c.line(x1, y, x2, y)  # Draws a line from (x1, y) to (x2, y).

    def draw_vline(self, x, y1, y2, color=colors.black):
        """Draws a vertical line between two y-coordinates at a fixed x-coordinate.

        Args:
            x (float): The x-coordinate at which the line is drawn.
            y1 (float): The starting y-coordinate of the line.
            y2 (float): The ending y-coordinate of the line.
            color: The color of the line.

        Modifies:
            c (canvas): The PDF canvas.
        """
        self.c.setStrokeColor(color)  # Set the line color.
        self.c.line(x, y1, x, y2)  # Draws a line from (x, y1) to (x, y2).

    def image_ratio(self, image_path, new_width=None, new_height=None):
        """Calculate new dimensions for an image to maintain aspect ratio.

        Args:
            image_path (str): Path to the image file.
            new_width (int, optional): Desired new width of the image.
            new_height (int, optional): Desired new height of the image.

        Returns:
            float: The new dimension not provided by the user (either new width or height).

        Raises:
            ValueError: If neither new_width nor new_height is provided.
        """
        img = Image.open(image_path)
        orig_width, orig_height = img.size

        if new_width:
            scale_factor = new_width / orig_width
            new_height = orig_height * scale_factor
            return new_height

        elif new_height:
            scale_factor = new_height / orig_height
            new_width = orig_width * scale_factor
            return new_width

        else:
            raise ValueError("Either new_width or new_height must be provided.")

    def draw_image(self, img_path, x_left, y_bottom, width=None, height=None):
        """Draws an image on a canvas with the specified dimensions. The image must be non-vector (e.g. .png, .jpeg)
               Doesn't work with .pdf or .svg

        Args:
            img_path (str): Path to the image file.
            x_left (float): Left x-coordinate where the image starts, bottom left corner.
            y_bottom (float): Bottom y-coordinate where the image starts, bottom left corner.
            width (int, optional): Width to which the image should be resized.
            height (int, optional): Height to which the image should be resized.

        Modifies:
            c (canvas): The PDF canvas.
        """
        if width:
            height = self.image_ratio(img_path, new_width=width)
        elif height:
            width = self.image_ratio(img_path, new_height=height)

        self.c.drawImage(img_path, x_left, y_bottom, width=width, height=height)

    def write_text(
        self, text_list, x, y, font_size=12, bold_flags=None, align="center"
    ):
        """Writes a list of texts at a specified position with optional bold and alignment.

        Args:
            text_list (list): List of texts to be displayed.
            x (float): X-coordinate for alignment, bottom left corner.
            y (float): Y-coordinate where texts start, bottom left corner.
            font_size (int): Font size of the text.
            bold_flags (list, optional): List of boolean values indicating if corresponding text is bold.
            align (str): Alignment of text; options are 'center', 'left', 'right'.

            Modifies:
            c (canvas): The PDF canvas.
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

    def get_svg_dimensions(self, svg_file):
        """Retrieve the dimensions of an SVG file.

        Args:
            svg_file (str): Path to the SVG file.

        Returns:
            tuple: A tuple containing the 'width' and 'height' of the SVG as strings.
        """
        tree = ET.parse(svg_file)
        root = tree.getroot()
        width = root.get("width")
        height = root.get("height")

        return width, height

    def scale(self, drawing, scaling_factor):
        """Scales a drawing object uniformly along both axes.

        Args:
            drawing (Drawing): A Drawing object from reportlab.graphics.shapes.
            scaling_factor (float): The factor by which to scale the drawing.

        Returns:
            Drawing: The scaled Drawing object.
        """
        scaling_x = scaling_y = scaling_factor
        drawing.scale(scaling_x, scaling_y)
        return drawing

    def add_image(self, image_path, scaling_factor, x, y):
        """Adds a scaled image to a reportlab canvas at specified coordinates.
           Works for .svg images

        Args:
            image_path (str): Path to the SVG image file.
            scaling_factor (float): The factor by which to scale the image.
            x (float): The x-coordinate where the image will be placed, bottom left corner.
            y (float): The y-coordinate where the image will be placed, bottom left corner.

            Modifies:
            c (canvas): The PDF canvas.
        """
        drawing = svg2rlg(image_path)
        # print(f"Minimum width before scaling: {drawing.minWidth()}")
        # print(f"Width before scaling: {drawing.width}")
        # print(f"Height before scaling: {drawing.height}")
        scaled_drawing = self.scale(drawing, scaling_factor=scaling_factor)
        renderPDF.draw(scaled_drawing, self.c, x, y)

    def draw_svg_image(self, image_path, x, y, width=None, height=None):
        """Draw an SVG image on a canvas at a scaled size based on specified width or height.

        Args:
            image_path (str): Path to the SVG file.
            x (float): The x-coordinate where the image will be placed, bottom left corner.
            y (float): The y-coordinate where the image will be placed, bottom left corner.
            width (float, optional): The desired width for scaling the SVG image.
            height (float, optional): The desired height for scaling the SVG image.

        Modifies:
            c (canvas): The PDF canvas.
        """
        svg_dimensions = self.get_svg_dimensions(image_path)
        svg_width_pt, svg_height_pt = float(svg_dimensions[0].split("pt")[0]), float(
            svg_dimensions[1].split("pt")[0]
        )
        svg_width_px = svg_width_pt * 4 / 3  # Convert points to pixels for width
        svg_height_px = svg_height_pt * 4 / 3  # Convert points to pixels for height

        if height:
            scaling_factor = height / svg_height_px
        elif width:
            scaling_factor = width / svg_width_px
        else:
            raise ValueError(
                "Either width or height must be provided to scale the SVG."
            )

        # Adjust scaling factor slightly larger than calculated
        scaling_factor_adjusted = scaling_factor

        # Call the function to add the image to the canvas
        self.add_image(image_path, scaling_factor=scaling_factor_adjusted, x=x, y=y)
