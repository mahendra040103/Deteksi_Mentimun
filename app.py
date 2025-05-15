from flask import Flask, render_template, request, redirect, url_for
from model.detector import DiseaseDetector
from PIL import Image
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
detector = DiseaseDetector()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_url = None

    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)

        file = request.files["image"]
        if file.filename != "":
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)

            image = Image.open(img_path).convert("RGB")
            result_image, results = detector.predict(image)

            result_image.save(img_path)
            img_url = url_for("static", filename="uploads/" + file.filename)
            result = results

    return render_template("index.html", image_url=img_url, result=result)


if __name__ == "__main__":
    # Untuk hosting seperti Render.com
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
