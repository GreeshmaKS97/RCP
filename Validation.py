# validation_service.py
import re
import io
import pytesseract
from PIL import Image


class ContentValidationService:
    @staticmethod
    def validate_any_content(parsed_text: str, file_bytes: bytes = None, file_extension: str = "") -> dict:
        """
        Input-Agnostic Validation Gateway.
        Catches scattered OCR character noise by checking token density and word structures.
        """
        raw_text = parsed_text.strip()

        # 1. Base Token Splitter
        # Breaks text into individual strings to evaluate if they look like human vocabulary
        tokens = [t.strip() for t in re.split(r'[\s\n\r]+', raw_text) if t.strip()]

        # ──────────────────────────────────────────────────────────────────
        # GATE 1: EMPTINESS & WORD VOLUME CHECK
        # ──────────────────────────────────────────────────────────────────
        if not raw_text or len(tokens) < 3:
            return {
                "is_valid": False,
                "readability_score": 0.0,
                "reason": "Validation Failed: No meaningful text strings could be extracted."
            }

        # ──────────────────────────────────────────────────────────────────
        # GATE 2: SCATTERED CHARACTER NOISE SHIELD (The fix for your current issue)
        # ──────────────────────────────────────────────────────────────────
        # Real documents have structural words (average length > 3 characters).
        # Noise artifacts present as an explosion of single letters ("e", "x", "¢", "\").
        meaningful_words = [t for t in tokens if len(t) >= 3]

        # Calculate what percentage of extracted strings are actually sustainable words
        word_structure_ratio = (len(meaningful_words) / len(tokens)) * 100 if tokens else 0

        # If more than 65% of your document consists of isolated 1- or 2-character fragments,
        # it is a photographic artifact, not a readable text layout.
        if word_structure_ratio < 35.0:
            return {
                "is_valid": False,
                "readability_score": round(word_structure_ratio, 2),
                "reason": f"Rejected: Extracted text consists primarily of random character fragments ({round(100 - word_structure_ratio, 2)}% structural noise). Please upload a clear document."
            }

        # ──────────────────────────────────────────────────────────────────
        # GATE 3: IMAGE CONTENT SPATIAL SHIELD
        # ──────────────────────────────────────────────────────────────────
        text_area_percentage = 100.0
        if file_bytes and file_extension in {"png", "jpg", "jpeg"}:
            try:
                with Image.open(io.BytesIO(file_bytes)) as img:
                    img_width, img_height = img.size
                    total_image_area = img_width * img_height

                ocr_data = pytesseract.image_to_data(Image.open(io.BytesIO(file_bytes)),
                                                     output_type=pytesseract.Output.DICT)

                total_text_area = 0
                n_boxes = len(ocr_data['text'])

                for i in range(n_boxes):
                    word = str(ocr_data['text'][i]).strip()
                    if word and any(char.isalnum() for char in word):
                        w = ocr_data['width'][i]
                        h = ocr_data['height'][i]
                        total_text_area += (w * h)

                if total_image_area > 0:
                    text_area_percentage = (total_text_area / total_image_area) * 100

            except Exception:
                text_area_percentage = 100.0

            if text_area_percentage < 1.5:  # Catches tiny stray text inside gigantic images
                return {
                    "is_valid": False,
                    "readability_score": 0.0,
                    "reason": "Rejected: Text area density is too sparse to qualify as a document resource."
                }

        # ──────────────────────────────────────────────────────────────────
        # GATE 4: STANDARD CHARACTER CHARACTER HEALTH
        # ──────────────────────────────────────────────────────────────────
        alphanumeric_and_spaces = len(re.findall(r'[a-zA-Z0-9\s.,!?()\-:]', raw_text))
        total_characters = len(raw_text)
        character_health_score = (alphanumeric_and_spaces / total_characters) * 100 if total_characters > 0 else 0

        if character_health_score < 70.0:
            return {
                "is_valid": False,
                "readability_score": round(character_health_score, 2),
                "reason": "Text contains too many unrecognizable symbols or corrupted font styles."
            }

        return {
            "is_valid": True,
            "readability_score": round(character_health_score, 2),
            "word_cohesion_score": round(word_structure_ratio, 2),
            "reason": "Text is stable, cohesive, and legible for processing."
        }