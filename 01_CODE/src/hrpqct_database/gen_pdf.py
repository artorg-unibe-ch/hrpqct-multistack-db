import os
from datetime import date
from pathlib import Path

from fpdf import FPDF

# Margin
MARGIN = 10
SUBSPACE = 5
# Page width: Width of A4 is 210mm
pw = 210 - 2 * MARGIN
# Cell height
ch = 6

# flake8: noqa: E501


class PDF(FPDF):
    def __init__(
        self,
        dataset_patient,
        gender,
        study_name,
        site=str(""),
        pat_no=int(0),
        Radius_measurement_number=int(0),
        Tibia_measurement_number=int(0),
        born=int(),
        num_slices=int(0),
        save_dir=Path,
        images_3d_path=Path,
        images_section_path=Path,
        t_scores=None,
        Radius_height_mm=int(0),
        Tibia_height_mm=int(0),
        Measurement_date=None,
    ):
        super().__init__()
        self.font = "Arial"
        self.today = date.today()
        self.date = self.today.strftime("%d/%m/%Y")
        self.site = str(site)
        self.pat_no = str(pat_no)
        self.born = str(born)
        self.gender = str(gender)
        self.study_name = str(study_name.iloc[0])
        self.num_slices = str(num_slices)
        self.epw = pw - 2 * SUBSPACE
        self.eph = 297 - 2 * SUBSPACE - 2 * MARGIN
        self.save_dir = save_dir
        self.dataset_patient = dataset_patient
        self.images_3d_path = images_3d_path
        self.images_section_path = images_section_path
        self.Radius_measurement_number = str(Radius_measurement_number)
        self.Tibia_measurement_number = str(Tibia_measurement_number)
        self.Radius_height_mm = str(Radius_height_mm)
        self.Tibia_height_mm = str(Tibia_height_mm)
        self.path_logo_left = (
            Path(__file__).parent
            / "utils"
            / "pdf-export-utils"
            / "1000px-Logo_Universität_Bern.png"
        )
        self.path_logo_right = (
            Path(__file__).parent
            / "utils"
            / "pdf-export-utils"
            / "1000px-Inselspital.png"
        )

        self.Measurement_date = str(Measurement_date)

        # If t_scores is provided, map them according to your parameter order:
        # Parameter order for t_scores:
        # [0]: Ct vBMD, [1]: Rel. cortical thickness, [2]: Tb BV/TV,
        # [3]: Ct/Tb (area ratio), [4]: Ultimate stress, [5]: Tot vBMD

        if t_scores is not None:
            # Mapping for Tibia (convert each element to float)
            self.ResDens_t = [
                float(t_scores["Tibia"][5]),
                float(t_scores["Tibia"][0]),
                float(t_scores["Tibia"][2]),
            ]
            self.ResStruct_t = [
                float(t_scores["Tibia"][1]),
                float(t_scores["Tibia"][3]),
            ]
            self.ResMeca_t = [float(t_scores["Tibia"][4])]

            # Mapping for Radius:
            self.ResDens_r = [
                float(t_scores["Radius"][5]),
                float(t_scores["Radius"][0]),
                float(t_scores["Radius"][2]),
            ]
            self.ResStruct_r = [
                float(t_scores["Radius"][1]),
                float(t_scores["Radius"][3]),
            ]
            self.ResMeca_r = [float(t_scores["Radius"][4])]

        else:
            # Fallback defaults if t_scores is not provided.
            self.ResDens_t = [0, 0, 0]
            self.ResStruct_t = [0, 0]
            self.ResMeca_t = [0]
            self.ResDens_r = [0, 0, 0]
            self.ResStruct_r = [0, 0]
            self.ResMeca_r = [0]

        self.Density = ["Tot.vBMD", "Ct.vBMD", "Tb. BV/TV"]
        self.Structure = ["Rel. Ct. Thickness", "Ct./Tb. ratio"]
        self.Meca = ["Ultimate stress"]

    def header(self):
        self.set_font(self.font, "", 10)
        self.cell(0, 8, "HR-pQCT Double- and Triple-stack Evaluation Sheet", 0, 1, "C")
        self.image(
            str(self.path_logo_right.resolve()),
            pw - 33 + 10,
            8,
            33,
        )
        self.image(
            str(self.path_logo_left.resolve()),
            10,
            3,
            20,
        )
        self.cell(0, 8, " ", 0, 1, "C")

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font, "", 10)
        self.cell(0, 8, f"Page {self.page_no()}", 0, 0, "C")

    def gen_pdf(self):

        # Page 1: Tibia

        self.add_page()
        self.set_font(self.font, "B", 20)
        self.cell(
            w=0,
            h=15,
            txt="  Tibia -3D Density and Structure Analysis",
            ln=1,
            border=1,
            align="C",
        )

        self.set_font(self.font, "", 12)
        self.cell(w=(pw / 4), h=ch, txt=" ", ln=1)
        self.cell(w=(pw / 4), h=ch, txt="Pat-No: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.pat_no, ln=0)
        self.cell(w=(pw / 4), h=ch, txt="Measurement Date: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.Measurement_date, ln=1)

        self.cell(w=(pw / 4), h=ch, txt="Age: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.born, ln=0)
        self.cell(w=(pw / 4), h=ch, txt="Evaluation Date: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.date, ln=1)

        self.cell(w=(pw / 4), h=ch, txt="Gender: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.gender, ln=0)
        self.cell(w=(pw / 4), h=ch, txt="Tibia height [mm]: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.Tibia_height_mm, ln=1)

        self.cell(w=(pw / 4), h=ch, txt="Study Name: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.study_name, ln=0)
        self.cell(w=(pw / 4), h=ch, txt="Measurement No: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.Tibia_measurement_number, ln=1)

        # Tibia Image
        tibia_img_path = [
            (Path(self.save_dir).absolute() / f"{self.pat_no}_Tibia_radar.png", 1.0),
            (Path(self.images_3d_path) / f"{self.Tibia_measurement_number}.png", 0.8),
        ]

        # Image Grid (2x1)
        img_w, img_h = 90, 90  # Image width and height
        spacing = 10
        total_grid_width = (2 * img_w) + spacing  # Total width of 2 images with spacing
        start_x, start_y = (
            pw - total_grid_width
        ) / 2 + 15, self.get_y()  # Center images

        for i, (tibia_img_path, scale) in enumerate(tibia_img_path):
            if tibia_img_path.is_file():
                y_pos = start_y
                if i % 2 == 1:
                    y_pos += 10  # Apply offset for second image only

                x_pos = start_x + i * (img_w + spacing)  # Horizontal placement
                self.image(
                    str(tibia_img_path.resolve()), x=x_pos, y=y_pos, w=img_w * scale
                )
            else:
                print(f"Error: {tibia_img_path} is not a valid PNG file.")

        self.set_y(160)  # adjust y position

        # Tibia Section Images
        tibia_section_path = [
            (
                Path(self.images_section_path)
                / f"{self.Tibia_measurement_number}_L.png",
                0.8,
            ),
            (
                Path(self.images_section_path)
                / f"{self.Tibia_measurement_number}_M.png",
                0.8,
            ),
            (
                Path(self.images_section_path)
                / f"{self.Tibia_measurement_number}_R.png",
                0.8,
            ),
        ]

        # Image Grid (3x1)
        img_w, img_h = 60, 60  # Image width and height
        spacing = 5
        total_grid_width = (3 * img_w) + spacing  # Total width of 2 images with spacing
        start_x, start_y = (
            pw - total_grid_width
        ) / 2 + 15, self.get_y()  # Center images

        for i, (tibia_section_path, scale) in enumerate(tibia_section_path):
            if tibia_section_path.is_file():
                x_pos = start_x + i * (img_w + spacing)  # Horizontal placement only
                self.image(
                    str(tibia_section_path.resolve()),
                    x=x_pos,
                    y=start_y,
                    w=img_w * scale,
                )
            else:
                print(f"Error: {tibia_section_path} is not a valid PNG file.")

        self.set_y(215)  # adjust y position
        # Table Header
        self.set_font(self.font, "B", 12)
        self.set_fill_color(128, 128, 128)
        self.cell(pw / 2, ch, "Parameter", 1, 0, "C", fill=1)
        self.cell(pw / 2, ch, "T-Score [SD]", 1, 1, "C", fill=1)

        # Density section

        self.set_font(self.font, "B", 12)
        self.set_fill_color(200, 200, 200)
        self.cell(pw, ch, "Density", 1, 1, "C", fill=1)

        self.set_font(self.font, "", 12)
        for i in range(0, len(self.Density)):
            self.cell(pw / 2, ch, self.Density[i], 1, 0, "L")
            self.cell(pw / 2, ch, "{:.3f}".format(self.ResDens_t[i]), 1, 1, "C")

        # Structure section
        self.set_font(self.font, "B", 12)
        self.set_fill_color(200, 200, 200)
        self.cell(pw, ch, "Structure", 1, 1, "C", fill=1)

        self.set_font(self.font, "", 12)
        for i in range(0, len(self.Structure)):
            self.cell(pw / 2, ch, self.Structure[i], 1, 0, "L")
            self.cell(pw / 2, ch, "{:.3f}".format(self.ResStruct_t[i]), 1, 1, "C")

        # Meca section
        self.set_font(self.font, "B", 12)
        self.set_fill_color(200, 200, 200)
        self.cell(pw, ch, "Mechanical Property", 1, 1, "C", fill=1)

        self.set_font(self.font, "", 12)
        for i in range(0, len(self.Meca)):
            self.cell(pw / 2, ch, self.Meca[i], 1, 0, "L")
            self.cell(pw / 2, ch, "{:.3f}".format(self.ResMeca_t[i]), 1, 1, "C")
        # ___________________________________________________________________________________________________
        # Page 2: Radius
        self.add_page()
        self.set_font(self.font, "B", 20)
        self.cell(
            w=0,
            h=15,
            txt="  Radius - 3D Density and Structure Analysis",
            ln=1,
            border=1,
            align="C",
        )

        self.set_font(self.font, "", 12)
        self.cell(w=(pw / 4), h=ch, txt=" ", ln=1)
        self.cell(w=(pw / 4), h=ch, txt="Pat-No: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.pat_no, ln=0)
        self.cell(w=(pw / 4), h=ch, txt="Measurement Date: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.Measurement_date, ln=1)

        self.cell(w=(pw / 4), h=ch, txt="Age: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.born, ln=0)

        self.cell(w=(pw / 4), h=ch, txt="Evaluation Date: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.date, ln=1)

        self.cell(w=(pw / 4), h=ch, txt="Gender: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.gender, ln=0)
        self.cell(w=(pw / 4), h=ch, txt="Radius height [mm]: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.Radius_height_mm, ln=1)

        self.cell(w=(pw / 4), h=ch, txt="Study Name: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.study_name, ln=0)
        self.cell(w=(pw / 4), h=ch, txt="Measurement No: ", ln=0)
        self.cell(w=(pw / 4), h=ch, txt=self.Radius_measurement_number, ln=1)

        self.ln(5)

        # Radius Image
        radius_img_path = [
            (Path(self.save_dir).absolute() / f"{self.pat_no}_Radius_radar.png", 1.0),
            (Path(self.images_3d_path) / f"{self.Radius_measurement_number}.png", 0.8),
        ]

        # Image Grid (2x1)
        img_w, img_h = 90, 90  # Image width and height
        spacing = 10
        total_grid_width = (2 * img_w) + spacing  # Total width of 2 images with spacing
        start_x, start_y = (
            pw - total_grid_width
        ) / 2 + 15, self.get_y()  # Center images

        for i, (radius_img_path, scale) in enumerate(radius_img_path):
            if radius_img_path.is_file():
                y_pos = start_y + (i // 2) * img_h
                if i % 2 == 1:
                    y_pos += 10  # Shift second image downward
                x_pos = start_x + i * (img_w + spacing)  # Horizontal placement only
                self.image(
                    str(radius_img_path.resolve()), x=x_pos, y=y_pos, w=img_w * scale
                )
            else:
                print(f"Error: {radius_img_path} is not a valid PNG file.")

        self.set_y(160)  # adjust y position

        # Tibia Section Images
        radius_section_path = [
            (
                Path(self.images_section_path)
                / f"{self.Radius_measurement_number}_L.png",
                0.8,
            ),
            (
                Path(self.images_section_path)
                / f"{self.Radius_measurement_number}_M.png",
                0.8,
            ),
            (
                Path(self.images_section_path)
                / f"{self.Radius_measurement_number}_R.png",
                0.8,
            ),
        ]

        # Image Grid (3x1)
        img_w, img_h = 60, 60  # Image width and height
        spacing = 5
        total_grid_width = (3 * img_w) + spacing  # Total width of 2 images with spacing
        start_x, start_y = (
            pw - total_grid_width
        ) / 2 + 15, self.get_y()  # Center images

        for i, (radius_section_path, scale) in enumerate(radius_section_path):
            if radius_section_path.is_file():
                x_pos = start_x + i * (img_w + spacing)  # Horizontal placement only
                self.image(
                    str(radius_section_path.resolve()),
                    x=x_pos,
                    y=start_y,
                    w=img_w * scale,
                )
            else:
                print(f"Error: {radius_section_path} is not a valid PNG file.")

        self.set_y(215)  # adjust y position
        # Table Header
        self.set_font(self.font, "B", 12)
        self.set_fill_color(128, 128, 128)
        self.cell(pw / 2, ch, "Parameter", 1, 0, "C", fill=1)
        self.cell(pw / 2, ch, "T-Score [SD]", 1, 1, "C", fill=1)

        # Density section

        self.set_font(self.font, "B", 12)
        self.set_fill_color(200, 200, 200)
        self.cell(pw, ch, "Density", 1, 1, "C", fill=1)

        self.set_font(self.font, "", 12)
        for i in range(0, len(self.Density)):
            self.cell(pw / 2, ch, self.Density[i], 1, 0, "L")
            self.cell(pw / 2, ch, "{:.3f}".format(self.ResDens_r[i]), 1, 1, "C")

        # Structure section
        self.set_font(self.font, "B", 12)
        self.set_fill_color(200, 200, 200)
        self.cell(pw, ch, "Structure", 1, 1, "C", fill=1)

        self.set_font(self.font, "", 12)
        for i in range(0, len(self.Structure)):
            self.cell(pw / 2, ch, self.Structure[i], 1, 0, "L")
            self.cell(pw / 2, ch, "{:.3f}".format(self.ResStruct_r[i]), 1, 1, "C")

        # Meca section
        self.set_font(self.font, "B", 12)
        self.set_fill_color(200, 200, 200)
        self.cell(pw, ch, "Mechanical Property", 1, 1, "C", fill=1)

        self.set_font(self.font, "", 12)
        for i in range(0, len(self.Meca)):
            self.cell(pw / 2, ch, self.Meca[i], 1, 0, "L")
            self.cell(pw / 2, ch, "{:.3f}".format(self.ResMeca_r[i]), 1, 1, "C")
        # ____________________________________________________________________________________
        # Save the PDF
        Path(("02_OUTPUT/PDFs")).mkdir(parents=True, exist_ok=True)
        outdir = Path(f"02_OUTPUT") / "PDFs" / f"{self.save_dir.name}.pdf"
        print("Saving PDF to:", outdir)
        self.output(outdir, "F")
