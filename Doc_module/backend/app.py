from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import base64
import io
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
import tempfile

import numpy as np
import pydicom
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "modal_weights"
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
GRADCAM_DIR = STORAGE_DIR / "gradcam"
DB_PATH = STORAGE_DIR / "doc_module.db"
DOCTOR_USERNAME = os.getenv("DOC_MODULE_DOCTOR_USERNAME", "doctor")
DOCTOR_PASSWORD = os.getenv("DOC_MODULE_DOCTOR_PASSWORD", "doctor123")

for directory in [STORAGE_DIR, UPLOAD_DIR, GRADCAM_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)


class CNNTransformerFeatureExtractor(nn.Module):
    def __init__(self, num_channels=3, cnn_out_channels=64, num_heads=8, num_layers=2, pretrained=False):
        super().__init__()
        self.custom_cnn = nn.Sequential(
            nn.Conv2d(num_channels, cnn_out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_out_channels, cnn_out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cnn_out_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_out_channels * 2, cnn_out_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cnn_out_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.custom_out_channels = cnn_out_channels * 4
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.pretrained_cnn = nn.Sequential(*list(backbone.features), nn.AdaptiveAvgPool2d((7, 7)))
        self.pretrained_out_channels = 1280
        self.total_channels = self.custom_out_channels + self.pretrained_out_channels
        self.pos_embedding = nn.Parameter(torch.randn(1, 49, self.total_channels))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.total_channels,
            nhead=num_heads,
            batch_first=True,
            dropout=0.1,
            dim_feedforward=2048,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        custom_feat = self.custom_cnn(x)
        pretrained_feat = self.pretrained_cnn(x)
        combined_feat = torch.cat([custom_feat, pretrained_feat], dim=1)
        batch_size, channels, height, width = combined_feat.shape
        transformer_input = combined_feat.view(batch_size, channels, height * width).permute(0, 2, 1)
        transformer_input = transformer_input + self.pos_embedding[:, :transformer_input.size(1)]
        transformer_output = self.transformer_encoder(transformer_input)
        return transformer_output.mean(dim=1)


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return self.fc3(x)


class MultimodalClassificationModel(nn.Module):
    def __init__(self, mri_extractor, pet_extractor, classifier_head):
        super().__init__()
        self.mri_extractor = mri_extractor
        self.pet_extractor = pet_extractor
        self.classifier_head = classifier_head

    def forward(self, mri_tensor, pet_tensor):
        mri_features = self.mri_extractor(mri_tensor)
        pet_features = self.pet_extractor(pet_tensor)
        combined = torch.cat([mri_features, pet_features], dim=1)
        return self.classifier_head(combined)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, target_score, output_size):
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)
        heatmap = F.interpolate(heatmap, size=output_size, mode="bilinear", align_corners=False)
        heatmap = heatmap.squeeze().detach().cpu().numpy()
        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def init_db():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            dob TEXT NOT NULL,
            diagnosis TEXT NOT NULL,
            diagnostic_band TEXT NOT NULL,
            certainty_label TEXT NOT NULL,
            certainty_note TEXT NOT NULL,
            clinician_summary TEXT NOT NULL,
            patient_summary TEXT NOT NULL,
            recommended_next_step TEXT NOT NULL,
            raw_cluster INTEGER NOT NULL,
            raw_cluster_label TEXT NOT NULL,
            diagnosis_scores_json TEXT NOT NULL,
            raw_cluster_scores_json TEXT NOT NULL,
            mri_image_path TEXT NOT NULL,
            pet_image_path TEXT NOT NULL,
            mri_gradcam_path TEXT NOT NULL,
            pet_gradcam_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()


def get_db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def load_metadata():
    metadata_path = WEIGHTS_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("metadata.json not found in modal_weights")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_models():
    metadata = load_metadata()
    extractor_cfg = metadata["feature_extractor_config"]
    classifier_cfg = metadata["classification_head_config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mri_extractor = CNNTransformerFeatureExtractor(
        num_channels=extractor_cfg["num_channels"],
        cnn_out_channels=extractor_cfg["cnn_out_channels"],
        num_heads=extractor_cfg["num_heads"],
        num_layers=extractor_cfg["num_layers"],
        pretrained=False,
    ).to(device)
    pet_extractor = CNNTransformerFeatureExtractor(
        num_channels=extractor_cfg["num_channels"],
        cnn_out_channels=extractor_cfg["cnn_out_channels"],
        num_heads=extractor_cfg["num_heads"],
        num_layers=extractor_cfg["num_layers"],
        pretrained=False,
    ).to(device)

    extractor_state = torch.load(WEIGHTS_DIR / metadata["paths"]["feature_extractor_weights"], map_location=device)
    mri_extractor.load_state_dict(extractor_state)
    pet_extractor.load_state_dict(extractor_state)

    classifier_head = ClassificationHead(
        input_dim=classifier_cfg["input_dim"],
        num_classes=classifier_cfg["num_classes"],
    ).to(device)
    classifier_head.load_state_dict(
        torch.load(WEIGHTS_DIR / metadata["paths"]["classification_head_weights"], map_location=device)
    )

    mri_extractor.eval()
    pet_extractor.eval()
    classifier_head.eval()
    return metadata, device, mri_extractor, pet_extractor, classifier_head


def dicom_to_pil(file_storage):
    dataset = pydicom.dcmread(file_storage.stream, force=True)
    pixel_array = dataset.pixel_array.astype(np.float32)

    if pixel_array.ndim > 2:
        pixel_array = pixel_array[0]

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    pixel_array -= pixel_array.min()
    max_value = pixel_array.max()
    if max_value > 0:
        pixel_array = pixel_array / max_value

    pixel_array = (pixel_array * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixel_array).convert("RGB")


def preprocess_image(file_storage):
    filename = (file_storage.filename or "").lower()
    file_storage.stream.seek(0)

    if filename.endswith(".dcm"):
        image = dicom_to_pil(file_storage)
    else:
        image = Image.open(file_storage.stream).convert("RGB")

    image = image.resize((256, 256))
    image_array = np.array(image)
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).float().unsqueeze(0)
    return image, image_array, tensor


def save_pil_image(image, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def encode_image_path(path):
    image = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_overlay(image_array, heatmap, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = image_array.astype(np.float32) / 255.0
    heat = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    heat_rgb = np.stack([heat, np.zeros_like(heat), 1.0 - heat], axis=-1)
    overlay = np.clip((0.58 * image + 0.42 * heat_rgb) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(destination)


def clinical_grouping(num_classes):
    if num_classes == 5:
        return {"CN": [0, 1], "MCI": [2], "AD": [3, 4]}
    if num_classes == 3:
        return {"CN": [0], "MCI": [1], "AD": [2]}

    groups = {"CN": [], "MCI": [], "AD": []}
    for index in range(num_classes):
        if index <= max(0, num_classes // 3 - 1):
            groups["CN"].append(index)
        elif index >= max(0, (2 * num_classes) // 3):
            groups["AD"].append(index)
        else:
            groups["MCI"].append(index)
    return groups


def summarize_for_patient(diagnosis, confidence):
    if confidence >= 0.8:
        certainty_label = "High pattern match"
        certainty_note = "The model found a strong imaging pattern match. A specialist should still confirm the result."
    elif confidence >= 0.6:
        certainty_label = "Moderate pattern match"
        certainty_note = "The model found a moderate pattern match. Clinical review is important before drawing conclusions."
    else:
        certainty_label = "Preliminary pattern match"
        certainty_note = "The model found only a limited pattern match. This should be treated as an early screening signal."

    summaries = {
        "CN": {
            "patient_summary": "Your scan pattern is currently closer to the cognitively normal range.",
            "clinician_summary": "Model grouped the study into the cognitively normal bucket after aggregating cluster scores.",
            "next_step": "Continue routine follow-up and review symptoms clinically if there are ongoing concerns.",
            "band": "Lower concern pattern",
        },
        "MCI": {
            "patient_summary": "Your scan pattern is closer to mild cognitive impairment, which may need further clinical assessment.",
            "clinician_summary": "Model grouped the study into the mild cognitive impairment bucket based on aggregated multimodal evidence.",
            "next_step": "Arrange memory testing, neurological review, and longitudinal follow-up to confirm the significance.",
            "band": "Intermediate concern pattern",
        },
        "AD": {
            "patient_summary": "Your scan pattern is closer to the Alzheimer's disease range and should be reviewed by a specialist promptly.",
            "clinician_summary": "Model grouped the study into the Alzheimer's-pattern bucket after aggregating the higher-severity clusters.",
            "next_step": "Recommend urgent specialist review, cognitive workup, and treatment planning discussion.",
            "band": "Higher concern pattern",
        },
    }
    selected = summaries[diagnosis]
    return {
        "certainty_label": certainty_label,
        "certainty_note": certainty_note,
        "patient_summary": selected["patient_summary"],
        "clinician_summary": selected["clinician_summary"],
        "recommended_next_step": selected["next_step"],
        "diagnostic_band": selected["band"],
    }


def normalize_scores(score_dict):
    total = sum(score_dict.values()) or 1.0
    return {
        key: {
            "value": round(float((value / total) * 100), 1),
            "label": "high" if value >= 0.7 else ("moderate" if value >= 0.4 else "low"),
        }
        for key, value in sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
    }


def fetch_prediction_by_id(prediction_id, include_clinician_view=False):
    connection = get_db_connection()
    row = connection.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    connection.close()
    if row is None:
        return None

    diagnosis_scores = json.loads(row["diagnosis_scores_json"])
    raw_cluster_scores = json.loads(row["raw_cluster_scores_json"])

    response = {
        "prediction_id": row["id"],
        "patient_id": row["patient_id"],
        "dob": row["dob"],
        "diagnosis": row["diagnosis"],
        "diagnostic_band": row["diagnostic_band"],
        "certainty_label": row["certainty_label"],
        "certainty_note": row["certainty_note"],
        "patient_summary": row["patient_summary"],
        "recommended_next_step": row["recommended_next_step"],
        "created_at": row["created_at"],
        "patient_scale": {
            "headline": row["diagnostic_band"],
            "certainty": row["certainty_label"],
            "explanation": row["certainty_note"],
        },
        "mri_gradcam_image": encode_image_path(row["mri_gradcam_path"]),
        "pet_gradcam_image": encode_image_path(row["pet_gradcam_path"]),
    }

    if include_clinician_view:
        response.update(
            {
                "clinician_summary": row["clinician_summary"],
                "raw_cluster": row["raw_cluster"],
                "raw_cluster_label": row["raw_cluster_label"],
                "diagnosis_scores": normalize_scores(diagnosis_scores),
                "raw_cluster_scores": normalize_scores(raw_cluster_scores),
                "mri_image": encode_image_path(row["mri_image_path"]),
                "pet_image": encode_image_path(row["pet_image_path"]),
            }
        )

    return response


def build_report_text(row, include_clinician_view=False):
    diagnosis_scores = json.loads(row["diagnosis_scores_json"])
    raw_cluster_scores = json.loads(row["raw_cluster_scores_json"])
    lines = [
        "AD Screening Report",
        "",
        f"Patient ID: {row['patient_id']}",
        f"Date of Birth: {row['dob']}",
        f"Created At: {row['created_at']}",
        "",
        f"Clinical Category: {row['diagnosis']}",
        f"Concern Band: {row['diagnostic_band']}",
        f"Certainty: {row['certainty_label']}",
        "",
        "Patient Explanation:",
        row["patient_summary"],
        "",
        "Next Recommended Step:",
        row["recommended_next_step"],
        "",
        "Certainty Note:",
        row["certainty_note"],
    ]

    if include_clinician_view:
        lines.extend(
            [
                "",
                "Clinician Summary:",
                row["clinician_summary"],
                "",
                f"Raw Cluster: {row['raw_cluster_label']}",
                "",
                "Three-category scores:",
            ]
        )
        for key, value in diagnosis_scores.items():
            lines.append(f"- {key}: {round(value * 100, 1)}%")
        lines.extend(["", "Raw cluster scores:"])
        for key, value in raw_cluster_scores.items():
            lines.append(f"- {key}: {round(value * 100, 1)}%")

    return "\n".join(lines)


def ordered_diagnosis_scores(score_dict):
    ordered = {}
    for label in ["CN", "MCI", "AD"]:
        ordered[label] = float(score_dict.get(label, 0.0))
    return ordered


def wrap_text_lines(pdf, text, max_width, font_name="Helvetica", font_size=10):
    pdf.setFont(font_name, font_size)
    words = text.split()
    current_line = []
    lines = []
    for word in words:
        tentative = " ".join(current_line + [word])
        if pdf.stringWidth(tentative, font_name, font_size) <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines or [""]


def draw_wrapped_text(pdf, text, x, y, max_width, line_height=14, font_name="Helvetica", font_size=10, color=colors.HexColor("#3F3A35")):
    lines = wrap_text_lines(pdf, text, max_width, font_name=font_name, font_size=font_size)

    pdf.setFont(font_name, font_size)
    pdf.setFillColor(color)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= line_height
    return y


def draw_score_table(pdf, title, items, x, y, width):
    items = list(items.items())
    box_height = max(92, 44 + (len(items) * 22))
    pdf.setFillColor(colors.HexColor("#F8F5F0"))
    pdf.roundRect(x, y - box_height, width, box_height, 10, stroke=0, fill=1)
    pdf.setStrokeColor(colors.HexColor("#DCCFBE"))
    pdf.roundRect(x, y - box_height, width, box_height, 10, stroke=1, fill=0)
    pdf.setFillColor(colors.HexColor("#6F6357"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(x + 12, y - 18, title)
    row_y = y - 38

    for label, value in items:
        percent = round(float(value) * 100, 1)
        pdf.setFont("Helvetica", 9)
        pdf.setFillColor(colors.HexColor("#433E39"))
        pdf.drawString(x + 12, row_y, label)
        pdf.drawRightString(x + width - 12, row_y, f"{percent}%")
        pdf.setFillColor(colors.HexColor("#ECE4D9"))
        pdf.roundRect(x + 12, row_y - 11, width - 24, 7, 3, stroke=0, fill=1)
        pdf.setFillColor(colors.HexColor("#8F89FF"))
        pdf.roundRect(x + 12, row_y - 11, (width - 24) * max(0.0, min(1.0, float(value))), 7, 3, stroke=0, fill=1)
        row_y -= 22


def draw_appendix_table(pdf, x, y, width, rows):
    header_height = 30
    note_width = width * 0.24
    value_width = width * 0.26
    section_width = width - note_width - value_width - 28

    measured_rows = []
    table_height = header_height
    for section, value, note in rows:
        note_lines = wrap_text_lines(pdf, note, note_width - 12, font_name="Helvetica", font_size=8)
        section_lines = wrap_text_lines(pdf, section, section_width - 12, font_name="Helvetica-Bold", font_size=9)
        value_lines = wrap_text_lines(pdf, value, value_width - 12, font_name="Helvetica", font_size=9)
        content_lines = max(len(note_lines), len(section_lines), len(value_lines))
        row_height = max(30, 12 + (content_lines * 10))
        measured_rows.append((section, value, note, note_lines, section_lines, value_lines, row_height))
        table_height += row_height

    pdf.setFillColor(colors.white)
    pdf.roundRect(x, y - table_height, width, table_height, 10, stroke=1, fill=1)
    pdf.setFillColor(colors.HexColor("#F3EEE7"))
    pdf.roundRect(x, y - header_height, width, header_height, 10, stroke=0, fill=1)

    col1 = x + 14
    col2 = col1 + section_width
    col3 = col2 + value_width

    pdf.setFillColor(colors.HexColor("#6F6357"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(col1, y - 19, "Section")
    pdf.drawString(col2, y - 19, "Value")
    pdf.drawString(col3, y - 19, "Clinical Note")

    current_y = y - header_height
    pdf.setStrokeColor(colors.HexColor("#E2D7C9"))
    for section, value, note, note_lines, section_lines, value_lines, row_height in measured_rows:
        pdf.line(x + 10, current_y, x + width - 10, current_y)
        section_y = current_y - 18
        value_y = current_y - 18
        note_y = current_y - 14

        pdf.setFont("Helvetica-Bold", 9)
        pdf.setFillColor(colors.HexColor("#3F3A35"))
        for line in section_lines:
            pdf.drawString(col1, section_y, line)
            section_y -= 10

        pdf.setFont("Helvetica", 9)
        for line in value_lines:
            pdf.drawString(col2, value_y, line)
            value_y -= 10

        pdf.setFont("Helvetica", 8)
        pdf.setFillColor(colors.HexColor("#5E554C"))
        for line in note_lines:
            pdf.drawString(col3, note_y, line)
            note_y -= 10

        current_y -= row_height

    return y - table_height


def draw_image_panel(pdf, image_path, title, x, y, width, height):
    pdf.setFillColor(colors.white)
    pdf.roundRect(x, y - height, width, height, 10, stroke=1, fill=1)
    pdf.setFillColor(colors.HexColor("#6F6357"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(x + 10, y - 16, title)

    image = Image.open(image_path).convert("RGB")
    image_reader = ImageReader(image)
    image_width, image_height = image.size
    available_width = width - 20
    available_height = height - 34
    scale = min(available_width / image_width, available_height / image_height)
    render_width = image_width * scale
    render_height = image_height * scale
    render_x = x + (width - render_width) / 2
    render_y = y - height + 10 + (available_height - render_height) / 2
    pdf.drawImage(image_reader, render_x, render_y, render_width, render_height, preserveAspectRatio=True, mask="auto")


def build_pdf_report(row, include_clinician_view=False):
    diagnosis_scores = ordered_diagnosis_scores(json.loads(row["diagnosis_scores_json"]))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.close()

    pdf = canvas.Canvas(temp_file.name, pagesize=A4)
    width, height = A4
    margin = 42

    pdf.setTitle(f"AD Screening Report {row['patient_id']}")
    pdf.setFillColor(colors.HexColor("#F8F4EE"))
    pdf.rect(0, 0, width, height, stroke=0, fill=1)

    pdf.setFillColor(colors.HexColor("#FFFFFF"))
    pdf.roundRect(margin, height - 118, width - (margin * 2), 82, 16, stroke=0, fill=1)
    pdf.setStrokeColor(colors.HexColor("#DCCFBE"))
    pdf.roundRect(margin, height - 118, width - (margin * 2), 82, 16, stroke=1, fill=0)

    pdf.setFillColor(colors.HexColor("#7B73FF"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(margin + 20, height - 58, "NEURO DS CLINICAL IMAGING REPORT")
    pdf.setFillColor(colors.HexColor("#221D18"))
    pdf.setFont("Helvetica-Bold", 18)
    draw_wrapped_text(
        pdf,
        "Alzheimer's Disease Multimodal Screening Report",
        margin + 20,
        height - 82,
        width - (margin * 2) - 40,
        line_height=20,
        font_name="Helvetica-Bold",
        font_size=18,
        color=colors.HexColor("#221D18"),
    )

    pdf.setFillColor(colors.HexColor("#F8F5F0"))
    pdf.roundRect(margin, height - 208, 250, 82, 12, stroke=0, fill=1)
    pdf.roundRect(margin + 268, height - 208, width - (margin * 2) - 268, 82, 12, stroke=0, fill=1)
    pdf.setStrokeColor(colors.HexColor("#DCCFBE"))
    pdf.roundRect(margin, height - 208, 250, 82, 12, stroke=1, fill=0)
    pdf.roundRect(margin + 268, height - 208, width - (margin * 2) - 268, 82, 12, stroke=1, fill=0)

    pdf.setFillColor(colors.HexColor("#6F6357"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(margin + 14, height - 146, "PATIENT INFORMATION")
    pdf.drawString(margin + 282, height - 146, "DIAGNOSTIC SUMMARY")

    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.HexColor("#433E39"))
    pdf.drawString(margin + 14, height - 168, f"Patient ID: {row['patient_id']}")
    pdf.drawString(margin + 14, height - 186, f"Date of Birth: {row['dob']}")
    pdf.drawString(margin + 282, height - 168, f"Clinical Category: {row['diagnosis']}")
    pdf.drawString(margin + 282, height - 186, f"Concern Band: {row['diagnostic_band']}")

    top_y = height - 236
    left_col_width = 250
    right_col_x = margin + 268
    right_col_width = width - (margin * 2) - 268

    pdf.setFillColor(colors.white)
    pdf.roundRect(margin, top_y - 132, left_col_width, 132, 12, stroke=1, fill=1)
    pdf.roundRect(right_col_x, top_y - 132, right_col_width, 132, 12, stroke=1, fill=1)
    pdf.setFillColor(colors.HexColor("#6F6357"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(margin + 14, top_y - 18, "PATIENT-FACING INTERPRETATION")
    pdf.drawString(right_col_x + 14, top_y - 18, "CLINICAL REVIEW NOTES")

    text_y = draw_wrapped_text(pdf, row["patient_summary"], margin + 14, top_y - 40, left_col_width - 28, line_height=14)
    draw_wrapped_text(pdf, f"Next step: {row['recommended_next_step']}", margin + 14, text_y - 6, left_col_width - 28, line_height=14, font_name="Helvetica-Bold")

    note_y = draw_wrapped_text(pdf, row["clinician_summary"], right_col_x + 14, top_y - 40, right_col_width - 28, line_height=14)
    draw_wrapped_text(pdf, f"Certainty note: {row['certainty_note']}", right_col_x + 14, note_y - 6, right_col_width - 28, line_height=14)

    table_y = top_y - 154
    draw_score_table(pdf, "Three-Class Diagnostic Scores", diagnosis_scores, margin, table_y, width - (margin * 2))

    image_top = table_y - 122
    draw_image_panel(pdf, row["mri_image_path"], "MRI Input", margin, image_top, 120, 132)
    draw_image_panel(pdf, row["mri_gradcam_path"], "MRI Grad-CAM", margin + 130, image_top, 120, 132)
    draw_image_panel(pdf, row["pet_image_path"], "PET Input", right_col_x, image_top, 120, 132)
    draw_image_panel(pdf, row["pet_gradcam_path"], "PET Grad-CAM", right_col_x + 130, image_top, 120, 132)

    footer_y = image_top - 148
    pdf.setFillColor(colors.HexColor("#6B6258"))
    pdf.setFont("Helvetica", 8)
    disclaimer = (
        "This report is an AI-assisted screening summary based on uploaded MRI and PET images. "
        "It is intended to support clinical review and does not replace diagnosis by a qualified specialist."
    )
    draw_wrapped_text(pdf, disclaimer, margin, footer_y, width - (margin * 2), line_height=11, font_size=8, color=colors.HexColor("#6B6258"))

    pdf.showPage()
    pdf.setFillColor(colors.HexColor("#F8F4EE"))
    pdf.rect(0, 0, width, height, stroke=0, fill=1)
    pdf.setFillColor(colors.HexColor("#221D18"))
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(margin, height - 56, "Clinical Appendix")
    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.HexColor("#433E39"))
    pdf.drawString(margin, height - 80, f"Report created at: {row['created_at']}")

    appendix_rows = [
        ("Clinical Category", row["diagnosis"], "Primary three-class multimodal screening output."),
        ("Concern Band", row["diagnostic_band"], "Patient-facing interpretation band used in the summary section."),
        ("Certainty", row["certainty_label"], row["certainty_note"]),
        ("Recommended Step", "Follow-up", row["recommended_next_step"]),
        ("Clinician Summary", "Review", row["clinician_summary"]),
    ]
    appendix_bottom = draw_appendix_table(pdf, margin, height - 110, width - (margin * 2), appendix_rows)

    pdf.setFillColor(colors.HexColor("#6F6357"))
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(margin, appendix_bottom - 24, "Three-Class Diagnostic Table")
    score_rows = [
        ("CN", f"{round(diagnosis_scores['CN'] * 100, 1)}%", "Cognitively normal imaging pattern"),
        ("MCI", f"{round(diagnosis_scores['MCI'] * 100, 1)}%", "Intermediate impairment-related imaging pattern"),
        ("AD", f"{round(diagnosis_scores['AD'] * 100, 1)}%", "Higher-concern Alzheimer's-pattern imaging signal"),
    ]
    draw_appendix_table(pdf, margin, appendix_bottom - 42, width - (margin * 2), score_rows)

    pdf.save()
    return temp_file.name


def send_report_download(row, include_clinician_view=False):
    pdf_path = build_pdf_report(row, include_clinician_view=include_clinician_view)
    filename = f"report_{row['patient_id']}_{row['id']}.pdf"
    return send_file(pdf_path, as_attachment=True, download_name=filename, mimetype="application/pdf")


def run_inference(patient_id, dob, mri_file, pet_file):
    metadata, device, mri_extractor, pet_extractor, classifier_head = build_models()
    multimodal_model = MultimodalClassificationModel(mri_extractor, pet_extractor, classifier_head).to(device)
    multimodal_model.eval()

    mri_image, mri_array, mri_tensor = preprocess_image(mri_file)
    pet_image, pet_array, pet_tensor = preprocess_image(pet_file)
    mri_tensor = mri_tensor.to(device)
    pet_tensor = pet_tensor.to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patient_slug = f"{patient_id}_{dob.replace('-', '')}_{timestamp}"

    upload_mri_path = UPLOAD_DIR / f"{patient_slug}_mri.png"
    upload_pet_path = UPLOAD_DIR / f"{patient_slug}_pet.png"
    save_pil_image(mri_image, upload_mri_path)
    save_pil_image(pet_image, upload_pet_path)

    with torch.enable_grad():
        mri_cam = GradCAM(multimodal_model, multimodal_model.mri_extractor.custom_cnn[7])
        outputs = multimodal_model(mri_tensor, pet_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).detach().cpu().numpy()
        raw_cluster = int(np.argmax(probabilities))
        mri_heatmap = mri_cam.generate(outputs[0, raw_cluster], output_size=(256, 256))
        mri_cam.close()

        pet_cam = GradCAM(multimodal_model, multimodal_model.pet_extractor.custom_cnn[7])
        outputs = multimodal_model(mri_tensor, pet_tensor)
        pet_heatmap = pet_cam.generate(outputs[0, raw_cluster], output_size=(256, 256))
        pet_cam.close()

    class_groups = clinical_grouping(len(probabilities))
    diagnosis_scores = {
        diagnosis: float(sum(probabilities[index] for index in cluster_indices if index < len(probabilities)))
        for diagnosis, cluster_indices in class_groups.items()
    }
    diagnosis = max(diagnosis_scores, key=diagnosis_scores.get)
    confidence = float(diagnosis_scores[diagnosis])
    summaries = summarize_for_patient(diagnosis, confidence)

    gradcam_mri_path = GRADCAM_DIR / f"{patient_slug}_mri_gradcam.png"
    gradcam_pet_path = GRADCAM_DIR / f"{patient_slug}_pet_gradcam.png"
    save_overlay(mri_array, mri_heatmap, gradcam_mri_path)
    save_overlay(pet_array, pet_heatmap, gradcam_pet_path)

    raw_cluster_scores = {f"Cluster_{index}": float(score) for index, score in enumerate(probabilities.tolist())}

    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (
            patient_id, dob, diagnosis, diagnostic_band, certainty_label, certainty_note,
            clinician_summary, patient_summary, recommended_next_step, raw_cluster,
            raw_cluster_label, diagnosis_scores_json, raw_cluster_scores_json,
            mri_image_path, pet_image_path, mri_gradcam_path, pet_gradcam_path, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            dob,
            diagnosis,
            summaries["diagnostic_band"],
            summaries["certainty_label"],
            summaries["certainty_note"],
            summaries["clinician_summary"],
            summaries["patient_summary"],
            summaries["recommended_next_step"],
            raw_cluster,
            metadata["class_labels"][raw_cluster] if raw_cluster < len(metadata["class_labels"]) else f"Cluster_{raw_cluster}",
            json.dumps(diagnosis_scores),
            json.dumps(raw_cluster_scores),
            str(upload_mri_path),
            str(upload_pet_path),
            str(gradcam_mri_path),
            str(gradcam_pet_path),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    prediction_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return fetch_prediction_by_id(prediction_id, include_clinician_view=True)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "message": "Doctor module backend is running"})


@app.route("/api/doctor/login", methods=["POST"])
def doctor_login():
    try:
        payload = request.get_json(silent=True) or {}
        username = (payload.get("username") or "").strip()
        password = payload.get("password") or ""

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400

        if username != DOCTOR_USERNAME or password != DOCTOR_PASSWORD:
            return jsonify({"error": "Invalid doctor credentials"}), 401

        return jsonify({"status": "ok", "username": username})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/doctor/predict", methods=["POST"])
def doctor_predict():
    try:
        patient_id = (request.form.get("patient_id") or "").strip()
        dob = (request.form.get("dob") or "").strip()
        mri_file = request.files.get("mri_image")
        pet_file = request.files.get("pet_image")

        if not patient_id or not dob or mri_file is None or pet_file is None:
            return jsonify({"error": "patient_id, dob, mri_image, and pet_image are required"}), 400

        result = run_inference(patient_id, dob, mri_file, pet_file)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/patient/result", methods=["POST"])
def patient_result():
    try:
        payload = request.get_json(silent=True) or {}
        patient_id = (payload.get("patient_id") or "").strip()
        dob = (payload.get("dob") or "").strip()

        if not patient_id or not dob:
            return jsonify({"error": "patient_id and dob are required"}), 400

        connection = get_db_connection()
        row = connection.execute(
            """
            SELECT id FROM predictions
            WHERE patient_id = ? AND dob = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT 1
            """,
            (patient_id, dob),
        ).fetchone()
        connection.close()

        if row is None:
            return jsonify({"error": "No report found for that patient ID and date of birth"}), 404

        return jsonify(fetch_prediction_by_id(row["id"], include_clinician_view=False))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/patient/history", methods=["POST"])
def patient_history():
    try:
        payload = request.get_json(silent=True) or {}
        patient_id = (payload.get("patient_id") or "").strip()
        dob = (payload.get("dob") or "").strip()

        if not patient_id or not dob:
            return jsonify({"error": "patient_id and dob are required"}), 400

        connection = get_db_connection()
        rows = connection.execute(
            """
            SELECT id, diagnosis, diagnostic_band, certainty_label, created_at
            FROM predictions
            WHERE patient_id = ? AND dob = ?
            ORDER BY datetime(created_at) DESC, id DESC
            """,
            (patient_id, dob),
        ).fetchall()
        connection.close()

        return jsonify(
            [
                {
                    "prediction_id": row["id"],
                    "diagnosis": row["diagnosis"],
                    "diagnostic_band": row["diagnostic_band"],
                    "certainty_label": row["certainty_label"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/doctor/history/<patient_id>", methods=["GET"])
def doctor_history(patient_id):
    try:
        connection = get_db_connection()
        rows = connection.execute(
            """
            SELECT id, dob, diagnosis, diagnostic_band, certainty_label, created_at
            FROM predictions
            WHERE patient_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
            """,
            (patient_id,),
        ).fetchall()
        connection.close()

        return jsonify(
            [
                {
                    "prediction_id": row["id"],
                    "dob": row["dob"],
                    "diagnosis": row["diagnosis"],
                    "diagnostic_band": row["diagnostic_band"],
                    "certainty_label": row["certainty_label"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/doctor/report/<int:prediction_id>", methods=["GET"])
def doctor_report(prediction_id):
    try:
        report = fetch_prediction_by_id(prediction_id, include_clinician_view=True)
        if report is None:
            return jsonify({"error": "Report not found"}), 404
        return jsonify(report)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/doctor/report/<int:prediction_id>/download", methods=["GET"])
def doctor_report_download(prediction_id):
    try:
        connection = get_db_connection()
        row = connection.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
        connection.close()
        if row is None:
            return jsonify({"error": "Report not found"}), 404
        return send_report_download(row, include_clinician_view=True)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/patient/report/<int:prediction_id>", methods=["POST"])
def patient_report(prediction_id):
    try:
        payload = request.get_json(silent=True) or {}
        patient_id = (payload.get("patient_id") or "").strip()
        dob = (payload.get("dob") or "").strip()

        if not patient_id or not dob:
            return jsonify({"error": "patient_id and dob are required"}), 400

        connection = get_db_connection()
        row = connection.execute(
            "SELECT id FROM predictions WHERE id = ? AND patient_id = ? AND dob = ?",
            (prediction_id, patient_id, dob),
        ).fetchone()
        connection.close()

        if row is None:
            return jsonify({"error": "Report not found for those credentials"}), 404

        return jsonify(fetch_prediction_by_id(prediction_id, include_clinician_view=False))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/patient/report/<int:prediction_id>/download", methods=["POST"])
def patient_report_download(prediction_id):
    try:
        payload = request.get_json(silent=True) or {}
        patient_id = (payload.get("patient_id") or "").strip()
        dob = (payload.get("dob") or "").strip()

        if not patient_id or not dob:
            return jsonify({"error": "patient_id and dob are required"}), 400

        connection = get_db_connection()
        row = connection.execute(
            "SELECT * FROM predictions WHERE id = ? AND patient_id = ? AND dob = ?",
            (prediction_id, patient_id, dob),
        ).fetchone()
        connection.close()

        if row is None:
            return jsonify({"error": "Report not found for those credentials"}), 404

        return send_report_download(row, include_clinician_view=False)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


init_db()

if __name__ == "__main__":
    print("Doctor module backend running on http://localhost:5050")
    app.run(debug=True, host="0.0.0.0", port=5050)
