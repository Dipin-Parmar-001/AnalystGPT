import os
import csv
from fastapi import UploadFile, HTTPException

UPLOAD_DIR = "C:/Users/Hp/Desktop/Dipen Parmar/AnalystGPT/uploads"
def parse_and_save_csv(file: UploadFile) -> str:
    """
    Parses an uploaded file, validates that it is a non-empty, valid CSV file,
    and saves it to the upload/ folder.
    
    Args:
        file (UploadFile): The uploaded file.
        
    Returns:
        str: The path to the saved file.
    """
    # 1. Check file extension
    filename = file.filename or ""
    if not filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .csv files are allowed.")
    
    # Read the file content
    try:
        content = file.file.read()
        file.file.seek(0)  # Reset pointer
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # 2. Check if the file is empty
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    
    # 3. Check if it is actually a CSV
    try:
        decoded_content = content.decode('utf-8', errors='ignore')
        # Try to parse the first few lines as CSV
        sample = "\n".join(decoded_content.splitlines()[:5])
        if not sample:
            raise ValueError("No content found")
        # Check if the dialect can be detected or if we can read it using csv.reader
        csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', '|'])
    except Exception:
        # Fallback verification: see if it has simple tabular structure
        try:
            reader = csv.reader(decoded_content.splitlines())
            # Check if we can read at least the first row and it has columns
            first_row = next(reader)
            if not first_row or len(first_row) < 1:
                raise HTTPException(status_code=400, detail="File is not a valid CSV.")
        except Exception:
            raise HTTPException(status_code=400, detail="File is not a valid CSV.")
    
    # 4. Save to the upload/ folder
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    return file_path
