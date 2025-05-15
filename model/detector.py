import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import gdown  # pastikan 'gdown' ditambahkan ke requirements.txt

LABELS = {
    1: "Anthracnose",
    2: "Bacterial Wilt",
    3: "Downy Mildew",
    4: "Fresh Leaf",
    5: "Gummy Stem Blight",
}
NUM_CLASSES = 6

MODEL_PATH = "model/ModelSkenarioSatu.pth"
MODEL_URL = (
    "https://drive.google.com/file/d/1fO9fVxA62M7Oof7_4jgoOhtJ8wFcrlyD/view?usp=sharing"
)


class DiseaseDetector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.ensure_model_exists()
        self.model = self.load_model()

    def ensure_model_exists(self):
        """Cek dan unduh model dari Google Drive jika belum tersedia"""
        if not os.path.exists(MODEL_PATH):
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            print("Mengunduh model dari Google Drive...")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            print("Model berhasil diunduh.")

    def load_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model

    def predict(self, image: Image.Image):
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(img_tensor)[0]
        return self.draw_boxes(image.copy(), outputs)

    def draw_boxes(self, image, outputs):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        results = []
        for box, label, score in zip(
            outputs["boxes"], outputs["labels"], outputs["scores"]
        ):
            if score >= self.threshold:
                label_id = label.item()
                label_name = LABELS.get(label_id, "Unknown")
                box = box.cpu().numpy()
                draw.rectangle(box, outline="red", width=3)
                draw.text(
                    (box[0], box[1]),
                    f"{label_name} ({score:.2f})",
                    fill="red",
                    font=font,
                )
                results.append((label_name, score.item()))
        return image, results
