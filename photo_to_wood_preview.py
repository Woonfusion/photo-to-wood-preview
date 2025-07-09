from flask import Flask, request, jsonify
import os
import tempfile
import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# ▼▼ Map hier de houtstructuur‑namen naar texture‑URL’s ▼▼
WOOD_TEXTURES = {
    "Eiken":    "https://example.com/textures/eiken.jpg",
    "Berken":   "https://example.com/textures/berken.jpg",
    "Vuren":    "https://example.com/textures/vuren.jpg",
    "Populier": "https://example.com/textures/populier.jpg",
    "MDF":      "https://example.com/textures/mdf.jpg",
}

def enhance_contrast_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def create_engraving_lines(gray):
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    lines = cv2.dilate(edges, kernel, iterations=1)
    return cv2.bitwise_not(lines)

def blend_with_wood(inv_lines, wood_bgr):
    engraving = cv2.cvtColor(inv_lines, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    wood = wood_bgr.astype(float) / 255.0
    return (engraving * wood * 255).clip(0, 255).astype(np.uint8)

def add_text(img_bgr, watermark, caption):
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except IOError:
        font = ImageFont.load_default()
    w, h = img.size
    draw.text((10, h - 40), watermark, font=font, fill=(255, 255, 255, 200))
    cw, ch = draw.textsize(caption, font=font)
    draw.text(((w - cw) // 2, h - ch - 10), caption, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def enforce_orientation(img, orientation):
    h, w = img.shape[:2]
    if orientation.lower().startswith("staand") and w > h:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation.lower().startswith("liggend") and h > w:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def download_image(url, suffix):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return path

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    file_url = data.get("file_url")
    wood_type = data.get("wood_type", "Eiken")
    size_label = data.get("size_label", "20×30 cm")
    orientation = data.get("orientation", "staand")

    if not file_url:
        return jsonify({"error": "file_url is required"}), 400

    wood_url = WOOD_TEXTURES.get(wood_type, WOOD_TEXTURES["Eiken"])
    try:
        img_path = download_image(file_url, ".jpg")
        wood_path = download_image(wood_url, ".jpg")

        img = cv2.imread(img_path)
        if img is None:
            return jsonify({"error": f"Afbeelding kan niet worden geladen van URL: {file_url}"}), 422

        wood = cv2.imread(wood_path)
        if wood is None:
            return jsonify({"error": f"Houtstructuur kan niet worden geladen van URL: {wood_url}"}), 422

        img = enforce_orientation(img, orientation)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_enh = enhance_contrast_gray(gray)
        inv_lines = create_engraving_lines(gray_enh)

        wood_resized = cv2.resize(wood, (inv_lines.shape[1], inv_lines.shape[0]))
        blended = blend_with_wood(inv_lines, wood_resized)

        caption = f"{wood_type} – {size_label}"
        final_bgr = add_text(blended, "Woonfusion.nl", caption)

        out_name = f"preview_{wood_type.lower()}_{size_label.replace('×','x').replace(' ','').replace('cm','cm')}.jpg"
        out_path = os.path.join("static", out_name)
        os.makedirs("static", exist_ok=True)
        cv2.imwrite(out_path, final_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        public_url = request.host_url + f"static/{out_name}"
        return jsonify({"preview_url": public_url})

    except Exception as e:
        return jsonify({"error": f"Verwerkingsfout: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
