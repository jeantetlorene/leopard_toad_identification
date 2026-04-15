import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "/home/Joshua/Downloads/leopard_toad_identification/detection/runs/detect/Western_Leopard_Toad_Project/yolov8n_clahe_run2/weights/best.pt"

# Load the model once
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)


def apply_clahe_preprocessing(image_rgb):
    """
    Exact replication of the training preprocessing.
    Input: RGB Numpy array
    Output: RGB Numpy array with CLAHE applied
    """
    # 1. Convert RGB (Gradio default) to LAB
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    # 2. Split channels
    l, a, b = cv2.split(lab)

    # 3. Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # 4. Merge and convert back to RGB
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return final_img


def predict_toad(image, image_path, conf_threshold):
    if image is None and not image_path:
        return None, {"message": "No image uploaded or path provided."}

    if image is None and image_path:
        try:
            image_bgr = cv2.imread(image_path.strip())
            if image_bgr is None:
                return None, {"message": f"Could not read image from {image_path}"}
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            return None, {"message": f"Error loading image: {str(e)}"}

    # 1. Preprocess
    processed_image = apply_clahe_preprocessing(image)

    # 2. Inference
    # imgsz=1280 matches your training config
    results = model.predict(processed_image, conf=conf_threshold, imgsz=1280)

    # 3. Visualization
    # Plot the boxes on the image.
    annotated_img = results[0].plot()

    # 4. Extract Text Results for the Table
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])

        detections.append(
            {
                "Class": class_name,
                "Confidence": f"{conf:.2%}",
                "Coordinates": [round(x, 1) for x in box.xyxy[0].tolist()],
            }
        )

    if not detections:
        return annotated_img, {"message": "No detections found."}

    return annotated_img, detections


# --- GRADIO INTERFACE ---
with gr.Blocks(title="Western Leopard Toad Detector") as demo:
    gr.Markdown("# Western Leopard Toad AI Monitor")
    gr.Markdown(
        "Upload a tunnel image. The system will apply **CLAHE enhancement** and detect Toads, Frogs, or Rats."
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Image", type="numpy")
            input_path = gr.Textbox(
                label="Or Provide Image Path", placeholder="/absolute/path/to/image.jpg"
            )
            conf_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.25, label="Confidence Threshold"
            )
            run_btn = gr.Button("Analyze Image", variant="primary")

        with gr.Column():
            output_img = gr.Image(label="Detections (CLAHE Enhanced)")
            output_text = gr.JSON(label="Detailed Results")

    run_btn.click(
        fn=predict_toad,
        inputs=[input_img, input_path, conf_slider],
        outputs=[output_img, output_text],
    )

demo.launch(share=True, debug=True)
