import pytesseract
import io
import cv2
import numpy as np
import fitz  # PyMuPDF
from docx import Document
from PIL import Image  # Pulls in Pillow image handlers
import pytesseract    # The OCR connection engine
from fastapi import FastAPI, UploadFile, File, HTTPException, status
import olefile
import pandas as pd
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"
app = FastAPI(
    title="Document Ingestion Service",
    description="Decoupled parsing service for PDFs, DOCX,  TXT and image files."
)


class DocumentParserService:
    @staticmethod
    def parse_doc_legacy(file_bytes: bytes) -> str:
        """
        Parses legacy binary .doc files by stripping out the text streams
        from the OLE structure directly in memory.
        """
        try:
            ole = olefile.OleFileIO(io.BytesIO(file_bytes))
            if ole.exists('WordDocument'):
                # Extract the core text stream out of the binary file layout
                stream = ole.openstream('WordDocument').read()
                # Decode using latin-1 or utf-16 depending on document headers
                # We filter out non-printable binary characters safely
                raw_str = stream.decode('latin-1', errors='ignore')

                # Clean up binary headers and non-text artifacts using a quick regex filter
                text_blocks = "".join(c for c in raw_str if c.isprintable() or c in '\n\r\t')

                # Strip out common residual system metadata headers from old Word files
                clean_text = text_blocks.strip()
                return clean_text
            else:
                raise ValueError("Not a valid OLE Word Document structure.")
        except Exception as e:


         raise ValueError(f"Legacy .doc parser failed: {str(e)}")

    @staticmethod
    def parse_pdf(file_bytes: bytes) -> str:
        """Extracts text from a raw PDF binary byte stream."""
        text_content = []
        # Open the PDF directly from memory using a stream buffer
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text_content.append(page.get_text())

        extracted_text = "\n".join(text_content).strip()
        if not extracted_text:
            raise ValueError("The uploaded PDF appears to be empty or scan-only (contains no selectable text layers).")
        return extracted_text

    @staticmethod
    def parse_docx(file_bytes: bytes) -> str:
        """Extracts text from a raw DOCX binary byte stream."""
        # Wrap raw bytes inside an in-memory file-like object
        file_stream = io.BytesIO(file_bytes)
        try:
            doc = Document(file_stream)
        except Exception:
            raise ValueError("Failed to structuralize file. The DOCX file might be corrupted.")

        text_content = []
        # Extract main body paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)

        # # Extract nested table contents
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text_content.append(cell.text)

        extracted_text = "\n".join(text_content).strip()
        if not extracted_text:
            raise ValueError("The uploaded DOCX file contains no readable text paragraphs.")
        return extracted_text

    @staticmethod
    def parse_txt(file_bytes: bytes) -> str:
        """Decodes raw text files with fallback encoding protections."""
        try:
            return file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            # Fallback handling if a user saves text with local encoding sheets
            try:
                return file_bytes.decode("latin-1").strip()
            except Exception:
                raise ValueError("Could not decode plain text file. Please ensure it uses standard UTF-8 encoding.")

    @staticmethod
    def parse_csv(file_bytes: bytes) -> str:
        """
        Parses CSV files dynamically and formats them into a clean Markdown table string.
        """
        try:
            # 1. Wrap the raw memory bytes in a stream buffer
            byte_stream = io.BytesIO(file_bytes)
            # 2. Convert binary bytes stream to text stream, explicitly dropping corrupt symbols safely
            text_stream = io.TextIOWrapper(byte_stream, encoding='utf-8', errors='ignore')

            # 3. Read the cleaned text matrix into pandas
            df = pd.read_csv(text_stream)
            # Clean empty rows/columns completely to save chatbot tokens
            df = df.dropna(how='all')
            # Convert the table into a markdown string (perfect layout for LLMs to read)
            if df.empty:
                return "--- Empty CSV Data Matrix Detected ---"

                # Convert the table into a markdown string layout
            return df.to_markdown(index=False)

        except Exception as e:
            raise ValueError(f"CSV Parser Engine failed: {str(e)}")

    @staticmethod
    def parse_excel(file_bytes: bytes) -> str:
        """
        Parses Excel files (.xlsx, .xls) across all containing worksheets
        and joins them into a unified readable text block.
        """
        try:
            # Read all sheets at once by setting sheet_name=None
            excel_dict = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, engine='openpyxl')

            combined_text_blocks = []

            for sheet_name, df in excel_dict.items():
                df = df.dropna(how='all')  # Clear noise
                if not df.empty:
                    # Mark sheet barriers clearly so the chatbot knows which table it's looking at
                    sheet_header = f"--- SHEET NAME: {sheet_name} ---\n"
                    markdown_table = df.to_markdown(index=False)
                    combined_text_blocks.append(sheet_header + markdown_table)

            return "\n\n".join(combined_text_blocks)
        except Exception as e:
            raise ValueError(f"Excel Parser Engine failed: {str(e)}")
    @staticmethod
    def parse_image(file_bytes: bytes) -> str:
        """Extracts textual layout strings from scanned images using local Tesseract OCR on D Drive."""
        try:
            # 1. Convert raw byte stream into an OpenCV pixel matrix
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Could not decode the uploaded image format.")

                # 2. Resize image if it's too small (Tesseract needs letters to be at least 20-30 pixels high)
                # We upscale the image slightly to make small PAN card fonts legible to the engine
            height, width = img.shape[:2]
            if width < 1500:
                img = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            # 3. IMAGE PRE-PROCESSING MAGIC:
            # Convert to Grayscale to remove background colors (blue/green card gradients)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 4. HIGH-ACCURACY FILTERING:
            # Bilateral filter removes background noise/textures (holograms) while keeping text edges sharp
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            # 5. OTSU'S BINARIZATION:
            # Automatically calculates the absolute best threshold value for the whole card
            _, processed_img = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


            # 6. Convert back to PIL format so Tesseract can read it
            final_image = Image.fromarray(processed_img)

            # 7. Optional: Optimize Tesseract Configuration for ID Cards
            # --psm 4 assumes a single column of text of variable sizes (great for IDs)
            custom_config = r'--psm 11 -l eng'
            extracted_text = pytesseract.image_to_string(final_image, config=custom_config)

            cleaned_text = extracted_text.strip()

            if not cleaned_text:
                raise ValueError("OCR completed successfully, but found no readable alphanumeric text in this image.")
            return cleaned_text

        except Exception as e:
            raise ValueError(f"OCR Engine failed to process image pixels: {str(e)}")

