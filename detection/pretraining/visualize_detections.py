import gradio as gr
import json
import os
from PIL import Image, ImageDraw, ImageFont

JSON_PATH = "/home/Joshua/Downloads/leopard_toad_identification/detection/pretraining/output.json"

# Load the JSON data
with open(JSON_PATH, "r") as f:
    data = json.load(f)

images_data = data.get("images", [])
categories = data.get(
    "detection_categories", {"1": "animal", "2": "person", "3": "vehicle"}
)

# Define colors for different categories
category_colors = {
    "1": "lime",  # animal
    "2": "red",  # person
    "3": "cyan",  # vehicle
}


def load_image_with_bboxes(idx):
    if not images_data:
        # Create a blank image with error message if no data
        img = Image.new("RGB", (800, 600), color=(255, 255, 255))
        return img, "No images found in the JSON."

    # Ensure index is within bounds
    idx = idx % len(images_data)
    img_info = images_data[idx]
    filename = img_info.get("file", "")

    # Check if the filename is an absolute path that exists
    if os.path.exists(filename):
        filepath = filename
    else:
        # Check relative to JSON path
        json_dir = os.path.dirname(JSON_PATH)
        rel_path = os.path.join(json_dir, filename)
        if os.path.exists(rel_path):
            filepath = rel_path
        else:
            filepath = filename

    if not os.path.exists(filepath):
        # Create a blank image with error message if file doesn't exist
        img = Image.new("RGB", (800, 600), color=(200, 200, 200))
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"Image not found at:\n{filepath}", fill=(255, 0, 0))
        return (
            img,
            f"**Image {idx + 1} / {len(images_data)}** - `{filename}` (NOT FOUND)",
        )

    try:
        img = Image.open(filepath).convert("RGB")
    except Exception as e:
        img = Image.new("RGB", (800, 600), color=(200, 200, 200))
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"Error loading image:\n{e}", fill=(255, 0, 0))
        return img, f"**Image {idx + 1} / {len(images_data)}** - `{filename}` (ERROR)"

    draw = ImageDraw.Draw(img)
    width, height = img.size

    for det in img_info.get("detections", []):
        cat_id = det.get("category", "1")
        if cat_id != "1":
            continue
        conf = det.get("conf", 0)
        bbox = det.get("bbox")  # [x_min, y_min, width_box, height_box]

        label = f"{categories.get(cat_id, 'unknown')} ({conf:.2f})"
        color = category_colors.get(cat_id, "orange")

        if bbox and len(bbox) == 4:
            x_min = int(bbox[0] * width)
            y_min = int(bbox[1] * height)
            box_width = int(bbox[2] * width)
            box_height = int(bbox[3] * height)

            x_max = x_min + box_width
            y_max = y_min + box_height

            # draw box
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

            # draw text background block for better visibility (optional)
            text_bbox = draw.textbbox((x_min, max(0, y_min - 15)), label)
            draw.rectangle(
                [text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill="black"
            )

            # draw text
            draw.text((x_min, max(0, y_min - 15)), label, fill=color)

    status = f"**Image {idx + 1} / {len(images_data)}** - `{filename}`"
    return img, status


def go_next(idx):
    new_idx = (idx + 1) % len(images_data)
    img, status = load_image_with_bboxes(new_idx)
    return new_idx, img, status


def go_prev(idx):
    new_idx = (idx - 1) % len(images_data)
    img, status = load_image_with_bboxes(new_idx)
    return new_idx, img, status


with gr.Blocks(title="MegaDetector Viewer") as demo:
    gr.Markdown("# MegaDetector Bounding Box Viewer")

    # Keep track of the current image index
    idx_state = gr.State(value=0)

    with gr.Row():
        prev_btn = gr.Button("⬅️ Previous")
        next_btn = gr.Button("Next ➡️")

    status_text = gr.Markdown()
    image_display = gr.Image(type="pil", interactive=False)

    # Load initial image on startup
    demo.load(
        fn=load_image_with_bboxes,
        inputs=[idx_state],
        outputs=[image_display, status_text],
    )

    # Button click events
    prev_btn.click(
        fn=go_prev, inputs=[idx_state], outputs=[idx_state, image_display, status_text]
    )

    next_btn.click(
        fn=go_next, inputs=[idx_state], outputs=[idx_state, image_display, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
