import gradio as gr
import os
from PIL import Image, ImageDraw

def load_image_with_bboxes(img_idx, images_dir, labels_dir):
    if not os.path.isdir(images_dir):
        return None, f"Error: {images_dir} is not a valid directory."
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    
    if not image_files:
        return None, f"No images found in {images_dir}."
    
    # Ensure index is within bounds
    img_idx = img_idx % len(image_files)
    img_name = image_files[img_idx]
    img_path = os.path.join(images_dir, img_name)
    
    # Try to find corresponding label file
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_name)
    
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return None, f"Error opening image {img_name}: {e}"
        
    draw = ImageDraw.Draw(img)
    
    bboxes_info = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    label = parts[0]
                    try:
                        xmin = float(parts[1])
                        ymin = float(parts[2])
                        xmax = float(parts[3])
                        ymax = float(parts[4])
                        
                        color = "lime" if "frog" in label.lower() else "cyan"
                        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
                        
                        # Label text
                        text_pos = (xmin, max(0, ymin - 20))
                        text_bbox = draw.textbbox(text_pos, label)
                        draw.rectangle(text_bbox, fill="black")
                        draw.text(text_pos, label, fill=color)
                        
                        bboxes_info.append(f"{label}: [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}]")
                    except ValueError:
                        continue
    else:
        bboxes_info.append("No label file found.")
        
    status = f"**Image {img_idx + 1} / {len(image_files)}** - `{img_name}`\n\n" + "\n".join(bboxes_info)
    return img, status

def go_next(idx, images_dir, labels_dir):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    new_idx = (idx + 1) % len(image_files) if image_files else 0
    img, status = load_image_with_bboxes(new_idx, images_dir, labels_dir)
    return new_idx, img, status

def go_prev(idx, images_dir, labels_dir):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    new_idx = (idx - 1) % len(image_files) if image_files else 0
    img, status = load_image_with_bboxes(new_idx, images_dir, labels_dir)
    return new_idx, img, status

with gr.Blocks(title="YOLO Dataset Viewer") as demo:
    gr.Markdown("# YOLO Dataset Bounding Box Viewer")
    
    with gr.Row():
        images_dir_input = gr.Textbox(label="Images Directory", value="/home/Joshua/Downloads/leopard_toad_identification/dataset/dataset1/images")
        labels_dir_input = gr.Textbox(label="Labels Directory", value="/home/Joshua/Downloads/leopard_toad_identification/dataset/dataset1/labels")
    
    idx_state = gr.State(value=0)
    
    with gr.Row():
        prev_btn = gr.Button("⬅️ Previous")
        load_btn = gr.Button("🔄 Load/Refresh")
        next_btn = gr.Button("Next ➡️")
        
    status_text = gr.Markdown()
    image_display = gr.Image(type="pil", interactive=False)
    
    # Event handlers
    load_btn.click(
        fn=load_image_with_bboxes,
        inputs=[idx_state, images_dir_input, labels_dir_input],
        outputs=[image_display, status_text]
    )
    
    prev_btn.click(
        fn=go_prev,
        inputs=[idx_state, images_dir_input, labels_dir_input],
        outputs=[idx_state, image_display, status_text]
    )
    
    next_btn.click(
        fn=go_next,
        inputs=[idx_state, images_dir_input, labels_dir_input],
        outputs=[idx_state, image_display, status_text]
    )
    
    # Auto-load on start
    demo.load(
        fn=load_image_with_bboxes,
        inputs=[idx_state, images_dir_input, labels_dir_input],
        outputs=[image_display, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
