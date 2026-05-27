print("\n" + "="*50)
print("!!! COPIED MASTER PIPELINE ACTIVATED SUCCESSFULLY !!!")
print("="*50 + "\n")
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, status
import re
# CRUCIAL CROSS-IMPORTS: Points directly to your single file extraction and validation services
from data_ingestion import DocumentParserService
from Validation import ContentValidationService  # <--- Points to your validation.py file

app = FastAPI(
    title="Central Chatbot Controller Platform",
    description="Main integration layer orchestrating single document pipelines."
)


@app.post("/upload", status_code=status.HTTP_200_OK)
async def upload_and_extract_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    file_extension = filename.split(".")[-1].lower() if "." in filename else ""

    supported_extensions = {"pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "csv", "xlsx"}
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '.{file_extension}'."
        )

    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="File is empty.")

        # Parse text using your data ingestion tool
        if file_extension == "pdf":
            raw_text = DocumentParserService.parse_pdf(file_bytes)
        elif file_extension == "docx":
            raw_text = DocumentParserService.parse_docx(file_bytes)
        elif file_extension == "doc":
            raw_text = DocumentParserService.parse_doc_legacy(file_bytes)
        elif file_extension == "txt":
            raw_text = DocumentParserService.parse_txt(file_bytes)
        elif file_extension == "csv":
            raw_text = DocumentParserService.parse_csv(file_bytes)
        elif file_extension == "xlsx":
            raw_text = DocumentParserService.parse_excel(file_bytes)
        else:
            raw_text = DocumentParserService.parse_image(file_bytes)

        # EMBEDDED SHIELD LOGIC
        clean_text = raw_text.strip()
        tokens = [t.strip() for t in re.split(r'[\s\n\r]+', clean_text) if t.strip()]

        # Filter for sustainable structural words
        meaningful_words = [t for t in tokens if len(t) >= 3]
        word_structure_ratio = (len(meaningful_words) / len(tokens)) * 100 if tokens else 0

        # CRITICAL REJECTION GUARD
        if word_structure_ratio < 45.0 and file_extension in {"png", "jpg", "jpeg"}:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Rejected: This file contains mostly non-text visual elements ({round(100 - word_structure_ratio, 2)}% character noise)."
            )

        return {
            "status": "success",
            "filename": filename,
            "file_type": file_extension,
            "cohesion_metrics": f"{round(word_structure_ratio, 2)}%",
            "raw_text": raw_text
        }

    except HTTPException as http_err:
        raise http_err
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    finally:
        await file.close()


if __name__ == "__main__":
    uvicorn.run("main1:app", host="127.0.0.1", port=8005, reload=True)