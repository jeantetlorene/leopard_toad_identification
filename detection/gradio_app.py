import os
import uuid
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import gradio as gr

# --- CONFIGURATION ---
# Models will be scanned from this directory
MODEL_SCAN_DIR = (
    "/home/Joshua/Downloads/leopard_toad_identification/detection/active learning"
)

# Global state to cache the loaded model
current_model_path = None
loaded_model = None
loaded_model_type = None
scanned_models = {}


def scan_available_models():
    """
    Scans active learning/ directory for best.pt model weights.
    Returns a dictionary mapping display names to absolute paths.
    """
    models = {}
    base_path = Path(MODEL_SCAN_DIR)

    if not base_path.exists():
        print(f"Warning: Scan directory {base_path} does not exist.")
        return models

    # Search for all best.pt files
    for pt_path in sorted(base_path.rglob("**/best.pt")):
        try:
            rel_path = pt_path.relative_to(base_path)
            parts = rel_path.parts

            # e.g., yolo_clahe/runs/cycle_4_pretrained_phase2/weights/best.pt
            if len(parts) >= 4:
                category = parts[0]  # e.g. yolo_clahe, rtdetr_clahe, faster_rcnn

                # Extract the run name (between category and weights)
                if "runs" in parts:
                    runs_idx = parts.index("runs")
                    weights_idx = (
                        parts.index("weights") if "weights" in parts else len(parts) - 1
                    )
                    run_name = " - ".join(parts[runs_idx + 1 : weights_idx])
                else:
                    run_name = parts[-2]
                display_name = f"{category} ({run_name})"
            else:
                display_name = f"{rel_path.parent.name} (custom)"

            models[display_name] = str(pt_path.resolve())
        except Exception as e:
            print(f"Error parsing model path {pt_path}: {e}")

    return models


def map_class_name(name):
    """
    Robust mapping of model classes to the requested output display names:
    WLT, Others, and Small Mammals.
    """
    n = str(name).lower().replace("_", " ").replace("-", " ").strip()
    if "leopard" in n or "toad" in n or "wlt" in n:
        return "WLT"
    if "amphibian" in n or "frog" in n or "other" in n:
        return "Others"
    if "mammal" in n or "rat" in n or "mouse" in n:
        return "Small Mammals"
    return name


def get_model(model_path):
    """
    Loads and caches a model based on path. Handles both YOLO/RT-DETR and Faster R-CNN.
    """
    global current_model_path, loaded_model, loaded_model_type

    if loaded_model is not None and current_model_path == model_path:
        return loaded_model, loaded_model_type

    print(f"Loading model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if the path indicates a Faster R-CNN model
    if "faster_rcnn" in str(model_path).lower():
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        num_classes = 4  # Background + 3 classes
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
        state_dict = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        loaded_model = model
        loaded_model_type = "faster_rcnn"
    else:
        # Load via ultralytics YOLO/RT-DETR
        from ultralytics import YOLO

        model = YOLO(model_path)
        model.to(device)
        loaded_model = model
        loaded_model_type = "yolo"

    current_model_path = model_path
    return loaded_model, loaded_model_type


def apply_clahe_preprocessing(image_rgb):
    """
    Input: RGB Numpy array
    Output: RGB Numpy array with CLAHE applied
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_img


def draw_detections(image_rgb, detections):
    """
    Draws custom, high-quality bounding boxes and class labels on the image.
    """
    annotated = image_rgb.copy()

    # Professional RGB Colors
    colors = {
        "WLT": (46, 204, 113),  # Emerald green
        "Others": (230, 126, 34),  # Carrot orange
        "Small Mammals": (52, 152, 219),  # Peter River blue
    }
    fallback_color = (231, 76, 60)  # Red

    for det in detections:
        class_name = det["Class"]
        conf_val = det["ConfValue"]
        box = [int(c) for c in det["Coordinates"]]

        color = colors.get(class_name, fallback_color)

        # Line thickness proportional to image dimensions
        h, w = annotated.shape[:2]
        thickness = max(2, int(max(h, w) / 450))

        # Bounding box
        cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, thickness)

        # Text label
        label_text = f"{class_name} {conf_val:.1%}"
        font_scale = max(0.5, max(h, w) / 1300.0)
        font_thickness = max(1, int(font_scale * 1.6))

        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        y_text = max(box[1], text_height + 10)

        # Text background
        cv2.rectangle(
            annotated,
            (box[0], y_text - text_height - 8),
            (box[0] + text_width + 4, y_text + baseline - 4),
            color,
            -1,
        )

        # Text text
        cv2.putText(
            annotated,
            label_text,
            (box[0] + 2, y_text - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated


def validate_and_read_image(path_str):
    """
    Security check: prevents path traversal and arbitrary file reads by checking
    extensions and validating that the file can be opened as an image.
    """
    path = Path(path_str.strip()).resolve()

    # Verify allowed image extension
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Extension '{path.suffix}' is not supported. Use JPG, PNG, BMP, or TIFF."
        )

    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Unable to read or parse image file at: {path}")

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def predict_toad(image, image_path, model_selection, custom_model_path, conf_threshold):
    """
    Unified inference function for Gradio. Runs model on preprocessed image,
    but prepares bounding box visualizations on BOTH the original and CLAHE images.
    """
    try:
        # 1. Resolve Model Path
        if model_selection == "Custom Path...":
            if not custom_model_path or not custom_model_path.strip():
                raise gr.Error("Custom model path is empty.")
            model_path_to_load = custom_model_path.strip()
        else:
            model_path_to_load = scanned_models.get(model_selection)
            if not model_path_to_load:
                raise gr.Error("Selected model path not found.")

        if not os.path.exists(model_path_to_load):
            raise gr.Error(f"Model file does not exist at {model_path_to_load}")

        # 2. Resolve and Validate Input Image
        if image is None and not image_path:
            raise gr.Error("No image uploaded or path provided.")

        if image is None and image_path:
            try:
                # Safe path validation and reading
                image = validate_and_read_image(image_path)
            except Exception as e:
                raise gr.Error(f"Error loading image path: {str(e)}")

        # Keep original image for visual output and visualization switching
        original_img = image.copy()

        # Resolve whether to apply CLAHE based on model path
        apply_clahe = "clahe" in model_path_to_load.lower()

        # 3. Preprocessing (CLAHE)
        processed_image = apply_clahe_preprocessing(image) if apply_clahe else image

        # 4. Load Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, model_type = get_model(model_path_to_load)

        # 5. Model Inference & Prediction Formatting
        detections = []

        if model_type == "yolo":
            results = model.predict(
                processed_image,
                conf=conf_threshold,
                imgsz=1280,
                verbose=False,
                device=device,
            )
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                raw_class_name = model.names[cls_id]
                class_name = map_class_name(raw_class_name)
                conf = float(box.conf[0])
                coords = [round(x, 1) for x in box.xyxy[0].tolist()]

                detections.append(
                    {
                        "Class": class_name,
                        "ConfValue": conf,
                        "Confidence": f"{conf:.2%}",
                        "Coordinates": coords,
                    }
                )
        elif model_type == "faster_rcnn":
            import torchvision.transforms.functional as TF

            img_tensor = TF.to_tensor(processed_image).to(device)
            with torch.no_grad():
                predictions = model([img_tensor])

            pred = predictions[0]
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            faster_rcnn_names = {
                1: "Other_Amphibian",
                2: "Small_Mammal",
                3: "Western_Leopard_Toad",
            }

            for idx, conf in enumerate(scores):
                if conf >= conf_threshold:
                    cls_id = int(labels[idx])
                    raw_class_name = faster_rcnn_names.get(cls_id, f"Unknown_{cls_id}")
                    class_name = map_class_name(raw_class_name)
                    coords = [round(x, 1) for x in boxes[idx].tolist()]

                    detections.append(
                        {
                            "Class": class_name,
                            "ConfValue": conf,
                            "Confidence": f"{conf:.2%}",
                            "Coordinates": coords,
                        }
                    )

        # 6. Create Visualizations
        annotated_orig = draw_detections(original_img, detections)
        annotated_clahe = (
            draw_detections(processed_image, detections)
            if apply_clahe
            else annotated_orig
        )

        # Save both annotated images for download links using temp files
        unique_id = uuid.uuid4().hex[:8]

        fd_orig, download_orig_path = tempfile.mkstemp(suffix=f"_orig_{unique_id}.png")
        os.close(fd_orig)
        cv2.imwrite(download_orig_path, cv2.cvtColor(annotated_orig, cv2.COLOR_RGB2BGR))

        if apply_clahe:
            fd_clahe, download_clahe_path = tempfile.mkstemp(
                suffix=f"_clahe_{unique_id}.png"
            )
            os.close(fd_clahe)
            cv2.imwrite(
                download_clahe_path,
                cv2.cvtColor(annotated_clahe, cv2.COLOR_RGB2BGR),
            )
        else:
            download_clahe_path = download_orig_path

        # 7. Create Detections DataFrame
        if detections:
            # Sort detections by confidence descending
            detections_sorted = sorted(
                detections, key=lambda x: x["ConfValue"], reverse=True
            )
            df_data = []
            for det in detections_sorted:
                df_data.append(
                    {
                        "Class": det["Class"],
                        "Confidence": det["Confidence"],
                        "Coordinates (xmin, ymin, xmax, ymax)": str(det["Coordinates"]),
                    }
                )
            df = pd.DataFrame(df_data)
            status_msg = f"Inference complete. Found {len(detections)} detection(s) using model type: {model_type}."
            gr.Info(status_msg)
        else:
            df = pd.DataFrame(
                columns=["Class", "Confidence", "Coordinates (xmin, ymin, xmax, ymax)"]
            )
            status_msg = "Inference complete. No objects detected."
            gr.Info(status_msg)

        # Return:
        # original_img, annotated_original, download_button_update, df,
        # view_mode_update, annotated_original, annotated_clahe, download_original_path_str, download_clahe_path_str
        view_mode_update = gr.update(
            choices=["Original Image", "CLAHE Enhanced Image"]
            if apply_clahe
            else ["Original Image"],
            value="Original Image",
        )
        return (
            annotated_orig,
            gr.update(value=str(download_orig_path), visible=True),
            df,
            view_mode_update,
            annotated_orig,
            annotated_clahe,
            str(download_orig_path),
            str(download_clahe_path),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise gr.Error(f"Error running detection: {str(e)}")


# Scan initially
scanned_models = scan_available_models()


# --- GRADIO INTERFACE ---
theme = gr.themes.Soft(primary_hue="emerald", secondary_hue="slate")

css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Font application */
.gradio-container, .gradio-container * {
    font-family: 'Outfit', sans-serif !important;
}

/* Make headers clean */
h1 {
    font-size: 2.0rem !important;
    font-weight: 600 !important;
    margin-top: 1.0rem !important;
    margin-bottom: 0.5rem !important;
}

p {
    color: #64748b !important;
    font-size: 1.05rem !important;
    margin-bottom: 1.5rem !important;
}
"""

with gr.Blocks(title="Western Leopard Toad Monitor") as demo:
    # State variables to hold images and files dynamically for instant client-side switching
    annotated_orig_state = gr.State(None)
    annotated_clahe_state = gr.State(None)
    download_orig_state = gr.State(None)
    download_clahe_state = gr.State(None)

    gr.Markdown("# Western Leopard Toad Monitor")
    gr.Markdown("Analyze images using YOLO, RT-DETR, or Faster R-CNN detection models.")

    with gr.Row():
        with gr.Column(scale=1):
            # Initial choices
            model_choices = list(scanned_models.keys()) + ["Custom Path..."]

            initial_value = "Custom Path..."
            if scanned_models:
                initial_value = model_choices[0]
                for choice in model_choices:
                    c_lower = choice.lower()
                    if (
                        "rtdetr_clahe" in c_lower
                        and "cycle_4_pretrained_phase2" in c_lower
                    ):
                        initial_value = choice
                        break

            model_dropdown = gr.Dropdown(
                choices=model_choices, value=initial_value, label="Detection Model"
            )

            # Custom path text field (initially hidden if we have models scanned)
            custom_model_textbox = gr.Textbox(
                label="Custom Model Path",
                placeholder="/path/to/weights/best.pt",
                value="",
                visible=(initial_value == "Custom Path..."),
            )

            conf_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="Confidence Threshold",
            )

            gr.HTML("<hr style='margin-top: 1rem; margin-bottom: 1rem;'/>")

            input_img = gr.Image(label="Upload Image File", type="numpy")
            input_path = gr.Textbox(
                label="Or Specify Absolute File Path",
                placeholder="/path/to/image.jpg",
            )

            run_btn = gr.Button("Analyze Image", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Detection Output"):
                    output_img = gr.Image(
                        label="Annotated Detections", type="numpy", format="png"
                    )
                    with gr.Row():
                        download_btn = gr.DownloadButton(
                            "Download Image",
                            variant="secondary",
                            visible=False,
                        )
                        view_mode_radio = gr.Radio(
                            choices=["Original Image", "CLAHE Enhanced Image"],
                            value="Original Image",
                            show_label=False,
                            container=False,
                        )

                with gr.Tab("Results Table"):
                    results_table = gr.Dataframe(
                        headers=[
                            "Class",
                            "Confidence",
                            "Coordinates (xmin, ymin, xmax, ymax)",
                        ],
                        datatype=["str", "str", "str"],
                        label="Detections Table",
                        interactive=False,
                    )

    # --- EVENT BINDINGS ---

    # Update custom textbox visibility based on dropdown selection
    def update_textbox_visibility(choice):
        return gr.update(visible=(choice == "Custom Path..."))

    model_dropdown.change(
        fn=update_textbox_visibility,
        inputs=[model_dropdown],
        outputs=[custom_model_textbox],
    )

    # Dynamic scan on dropdown focus
    def dynamic_scan_choices():
        global scanned_models
        scanned_models = scan_available_models()
        choices = list(scanned_models.keys()) + ["Custom Path..."]
        return gr.update(choices=choices)

    model_dropdown.focus(fn=dynamic_scan_choices, inputs=[], outputs=[model_dropdown])

    # Visualization switcher binding (instantly toggles output based on cache state variables)
    def switch_view(view_choice, orig_img, clahe_img, orig_path, clahe_path):
        if view_choice == "CLAHE Enhanced Image" and clahe_img is not None:
            return clahe_img, gr.update(value=clahe_path)
        return orig_img, gr.update(value=orig_path)

    view_mode_radio.change(
        fn=switch_view,
        inputs=[
            view_mode_radio,
            annotated_orig_state,
            annotated_clahe_state,
            download_orig_state,
            download_clahe_state,
        ],
        outputs=[output_img, download_btn],
    )

    # Analyze Action
    run_btn.click(
        fn=predict_toad,
        inputs=[
            input_img,
            input_path,
            model_dropdown,
            custom_model_textbox,
            conf_slider,
        ],
        outputs=[
            output_img,
            download_btn,
            results_table,
            view_mode_radio,
            annotated_orig_state,
            annotated_clahe_state,
            download_orig_state,
            download_clahe_state,
        ],
    )

# Launch local-only for security
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False, debug=True, theme=theme, css=css)
