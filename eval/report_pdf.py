# ======================================================
# eval/report_pdf.py
# PDF report class and image-layout utilities
# ======================================================

from fpdf import FPDF


class ReportPDF(FPDF):
    # -- Header printed on every page ----------------
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Synthetic Data Validation Report", ln=True, align="C")
        self.ln(3)

    # -- Section title: always starts a NEW page ------
    def section(self, title: str):
        self.add_page()
        self.set_font("Arial", "B", 13)
        self.set_fill_color(220, 230, 245)
        self.cell(0, 11, title, ln=True, fill=True)
        #self.ln(4)

    # -- Metrics table --------------------------------
    def add_metrics_table(self, metrics_dict: dict, title: str):
        self.set_font("Arial", "B", 11)
        self.cell(0, 8, title, ln=True)
        self.set_font("Arial", "", 10)
        for k, v in metrics_dict.items():
            text = f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}"
            self.multi_cell(0, 6, text)
        self.ln(4)

    # -- Italic note block ----------------------------
    def add_note(self, text: str):
        self.set_font("Arial", "I", 9)
        self.multi_cell(0, 5, text)
        self.ln(3)


# -- Layout helpers -----------------------------------

def add_images_grid(pdf: ReportPDF, image_paths: list[str],
                    per_row: int = 3, per_page: int = 9,
                    width: float | None = None):
    """
    Place images in a grid layout.
    per_row  : images per row (2 or 3)
    per_page : images per page (e.g. 9 for 3x3, 4 for 2x2, 6 for 3x2)
    width    : image width in mm; auto-computed from per_row if None
    """
    if not image_paths:
        return

    page_w   = 190          # usable page width (mm)
    gap      = 4            # gap between images (mm)
    if width is None:
        width = (page_w - gap * (per_row - 1)) / per_row

    count = 0
    for img in image_paths:
        if count % per_page == 0:
            pdf.add_page()
        # x position based on column index
        col = count % per_row
        x   = pdf.l_margin + col * (width + gap)
        pdf.image(img, x=x, w=width)
        if col == per_row - 1:
            pdf.ln(2)
        count += 1
    pdf.ln(4)


def add_images_full_width(pdf: ReportPDF, image_paths: list[str], width: float = 190):
    """One image per page, full width (for large plots like correlation matrices)."""
    for img in image_paths:
        pdf.add_page()
        pdf.image(img, w=width)
        pdf.ln(4)


def add_images_two_per_page(pdf: ReportPDF, image_paths: list[str], width: float = 190):
    """Two images per page, stacked vertically."""
    for i, img in enumerate(image_paths):
        if i % 2 == 0:
            pdf.add_page()
        pdf.image(img, w=width)
        pdf.ln(5)


def add_images_four_per_page(pdf: ReportPDF, image_paths: list[str], width: float = 93):
    """
    Four images per page in a 2x2 grid (good for temporal trajectories).
    """
    page_w = 190
    gap    = 4
    per_row = 2
    width   = (page_w - gap) / per_row

    count = 0
    for img in image_paths:
        if count % 4 == 0:
            pdf.add_page()
        col = count % per_row
        x   = pdf.l_margin + col * (width + gap)
        pdf.image(img, x=x, w=width)
        if col == 1:
            pdf.ln(2)
        count += 1
    pdf.ln(4)