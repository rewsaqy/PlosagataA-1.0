"""OCR poster hoaks -> teks. Coba pytesseract, fallback ke pesan jelas."""
from pathlib import Path


def ocr_poster(image_path: str) -> dict:
    p = Path(image_path)
    if not p.exists():
        return {"ok": False, "text": "", "error": f"File tidak ada: {image_path}"}
    try:
        from PIL import Image
        import pytesseract

        img = Image.open(p).convert("RGB")
        # Bahasa Indonesia + Inggris kalau tessdata tersedia
        try:
            text = pytesseract.image_to_string(img, lang="ind+eng")
        except Exception:
            text = pytesseract.image_to_string(img)
        text = text.strip()
        return {"ok": True, "text": text, "error": ""}
    except ImportError:
        return {
            "ok": False,
            "text": "",
            "error": "PIL/pytesseract belum install. Jalankan: pip install pillow pytesseract + install tesseract-ocr. Atau ketik manual isi poster ke form web.",
        }
    except Exception as e:
        return {"ok": False, "text": "", "error": f"OCR gagal: {e}"}
