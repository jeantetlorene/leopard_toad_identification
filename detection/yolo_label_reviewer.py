import os
import cv2
import json
import numpy as np
import pandas as pd
import gradio as gr


def load_classes(base_path):
    classes_file = os.path.join(base_path, "classes.txt")
    if os.path.isfile(classes_file):
        try:
            with open(classes_file, "r") as f:
                classes = [line.strip() for line in f if line.strip()]
                if classes:
                    return classes
        except Exception as e:
            print(f"Error reading classes.txt: {e}")

    notes_file = os.path.join(base_path, "notes.json")
    if os.path.isfile(notes_file):
        try:
            with open(notes_file, "r") as f:
                data = json.load(f)
                categories = data.get("categories", [])
                categories.sort(key=lambda x: x.get("id", 0))
                names = [c.get("name") for c in categories if c.get("name")]
                if names:
                    return names
        except Exception as e:
            print(f"Error reading notes.json: {e}")
    return []


def load_dataset(base_path):
    if not base_path:
        return [], [], "Please enter a base path.", gr.update(choices=[])

    base_path = base_path.strip()
    if not os.path.isdir(base_path):
        return (
            [],
            [],
            f"Error: Base directory '{base_path}' does not exist.",
            gr.update(choices=[]),
        )

    images_dir = os.path.join(base_path, "images")
    if not os.path.isdir(images_dir):
        return (
            [],
            [],
            f"Error: Images subdirectory '{images_dir}' not found.",
            gr.update(choices=[]),
        )

    classes = load_classes(base_path)

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".JPG",
        ".JPEG",
        ".PNG",
        ".BMP",
    )
    try:
        images = [f for f in os.listdir(images_dir) if f.endswith(image_extensions)]
        images.sort()
    except Exception as e:
        return [], [], f"Error scanning images directory: {e}", gr.update(choices=[])

    if not images:
        return (
            [],
            [],
            f"No images found in '{images_dir}'. Check extensions.",
            gr.update(choices=[]),
        )

    msg = f"Successfully loaded {len(images)} images and {len(classes)} classes."
    image_choices = [f"{i + 1}: {img}" for i, img in enumerate(images)]
    return (
        images,
        classes,
        msg,
        gr.update(
            choices=image_choices, value=image_choices[0] if image_choices else None
        ),
    )


def load_boxes(base_path, image_name, classes):
    labels_dir = os.path.join(base_path, "labels")
    base_name, _ = os.path.splitext(image_name)
    label_path = os.path.join(labels_dir, base_name + ".txt")

    boxes = []
    if os.path.isfile(label_path):
        try:
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    confidence = float(parts[5]) if len(parts) >= 6 else None

                    class_name = (
                        classes[class_id]
                        if class_id < len(classes)
                        else f"Class_{class_id}"
                    )
                    boxes.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "coords": coords,
                            "keep": True,
                            "confidence": confidence,
                        }
                    )
        except Exception as e:
            print(f"Error loading labels from {label_path}: {e}")
    return boxes


def save_boxes(base_path, image_name, boxes):
    labels_dir = os.path.join(base_path, "labels")
    if not os.path.isdir(labels_dir):
        try:
            os.makedirs(labels_dir, exist_ok=True)
        except Exception as e:
            return f"Error creating labels directory: {e}"

    base_name, _ = os.path.splitext(image_name)
    label_path = os.path.join(labels_dir, base_name + ".txt")

    lines = []
    for box in boxes:
        if box["keep"]:
            coords_str = " ".join(f"{x:.6f}" for x in box["coords"])
            conf_str = (
                f" {box['confidence']:.6f}" if box["confidence"] is not None else ""
            )
            lines.append(f"{box['class_id']} {coords_str}{conf_str}\n")

    try:
        with open(label_path, "w") as f:
            f.writelines(lines)
        return "Saved successfully."
    except Exception as e:
        return f"Error saving labels: {e}"


def draw_image_boxes(image_path, boxes):
    img = cv2.imread(image_path)
    if img is None:
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        canvas[:, :] = (15, 23, 42)
        cv2.putText(
            canvas,
            "Error: Image file not found",
            (150, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (231, 76, 60),
            2,
            cv2.LINE_AA,
        )
        return canvas

    h, w, _ = img.shape
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    thickness = max(2, int(max(h, w) / 500))
    font_scale = max(0.5, thickness / 3)

    for idx, box in enumerate(boxes):
        x_c, y_c, bw, bh = box["coords"]
        xmin = int((x_c - bw / 2) * w)
        ymin = int((y_c - bh / 2) * h)
        xmax = int((x_c + bw / 2) * w)
        ymax = int((y_c + bh / 2) * h)

        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))

        if box["keep"]:
            color = (46, 204, 113)
            status_suffix = ""
        else:
            color = (231, 76, 60)
            status_suffix = " [DELETE]"

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

        conf_str = (
            f" ({box['confidence']:.2%})" if box["confidence"] is not None else ""
        )
        label_text = f"#{idx + 1}: {box['class_name']}{conf_str}{status_suffix}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness // 2 + 1
        )

        label_ymin = max(0, ymin - text_h - 12)
        cv2.rectangle(img, (xmin, label_ymin), (xmin + text_w + 6, ymin), color, -1)

        cv2.putText(
            img,
            label_text,
            (xmin + 3, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness // 2 + 1,
            cv2.LINE_AA,
        )

    return img


def get_control_updates(boxes, classes, selected_idx):
    if not boxes or selected_idx < 0 or selected_idx >= len(boxes):
        return (
            gr.update(choices=[], value=None, visible=False),
            gr.update(visible=False),
            gr.update(choices=[], value=None, visible=False),
        )

    selector_choices = [
        f"Box #{i + 1}: {box['class_name']}{'' if box['keep'] else ' [DELETE]'}"
        for i, box in enumerate(boxes)
    ]
    selector_val = selector_choices[selected_idx]

    current_box = boxes[selected_idx]
    keep_val = "Keep" if current_box["keep"] else "Delete"

    return (
        gr.update(choices=selector_choices, value=selector_val, visible=True),
        gr.update(value=keep_val, visible=True),
        gr.update(choices=classes, value=current_box["class_name"], visible=True),
    )


def update_ui_for_image(base_path, image_name, classes, index, total_images):
    image_path = os.path.join(base_path, "images", image_name)
    boxes = load_boxes(base_path, image_name, classes)
    annotated_img = draw_image_boxes(image_path, boxes)

    progress_text = f"**Image {index + 1} of {total_images}**: `{image_name}`"
    jump_dropdown_val = f"{index + 1}: {image_name}"

    sel_idx = 0 if boxes else -1
    sel_update, keep_update, cls_update = get_control_updates(boxes, classes, sel_idx)

    return [
        annotated_img,
        index,
        boxes,
        progress_text,
        jump_dropdown_val,
        "",
        sel_idx,
        sel_update,
        keep_update,
        cls_update,
    ]


def handle_load_dataset(base_path):
    images, classes, msg, dropdown_update = load_dataset(base_path)
    if not images:
        return [
            [],
            [],
            [],
            msg,
            dropdown_update,
            gr.update(choices=["All"], value="All"),
            None,
            0,
            [],
            "No dataset loaded",
            "",
            -1,
            gr.update(choices=[], value=None, visible=False),
            gr.update(visible=False),
            gr.update(choices=[], value=None, visible=False),
        ]

    first_img = images[0]
    ui_updates = update_ui_for_image(base_path, first_img, classes, 0, len(images))
    filter_dropdown_update = gr.update(choices=["All"] + classes, value="All")

    return [
        images,
        images,
        classes,
        msg,
        dropdown_update,
        filter_dropdown_update,
        ui_updates[0],
        ui_updates[1],
        ui_updates[2],
        ui_updates[3],
        ui_updates[5],
        ui_updates[6],
        ui_updates[7],
        ui_updates[8],
        ui_updates[9],
    ]


def handle_navigate(base_path, images, classes, current_index, boxes, action):
    if not images:
        return [gr.skip()] * 9

    current_img = images[current_index]
    save_boxes(base_path, current_img, boxes)

    new_index = current_index
    if action == "next":
        new_index = min(len(images) - 1, current_index + 1)
    elif action == "prev":
        new_index = max(0, current_index - 1)
    elif isinstance(action, str) and ":" in action:
        try:
            parts = action.split(":", 1)
            new_index = int(parts[0]) - 1
        except:
            pass

    return update_ui_for_image(
        base_path, images[new_index], classes, new_index, len(images)
    )


def handle_save(base_path, images, current_index, boxes):
    if not images or current_index < 0 or current_index >= len(images) or not boxes:
        return "No image annotations to save."
    img_name = images[current_index]
    msg = save_boxes(base_path, img_name, boxes)
    return f"Manual save for {img_name}: {msg}"


def handle_class_filter(
    base_path, raw_images, classes, current_index, current_boxes, filter_class
):
    if not raw_images:
        return [gr.skip()] * 9

    if 0 <= current_index < len(raw_images) and current_boxes is not None:
        save_boxes(base_path, raw_images[current_index], current_boxes)

    if not filter_class or filter_class == "All":
        filtered_images = list(raw_images)
    else:
        filtered_images = []
        for img in raw_images:
            boxes = load_boxes(base_path, img, classes)
            if any(box["class_name"] == filter_class for box in boxes):
                filtered_images.append(img)

    if not filtered_images:
        empty_msg = f"No images contain class '{filter_class}'."
        return [
            [],
            0,
            [],
            None,
            f"**0 of 0**: {empty_msg}",
            gr.update(choices=[], value=None),
            empty_msg,
            -1,
            gr.update(choices=[], value=None, visible=False),
            gr.update(visible=False),
            gr.update(choices=[], value=None, visible=False),
        ]

    new_index = 0
    ui_updates = update_ui_for_image(
        base_path, filtered_images[new_index], classes, new_index, len(filtered_images)
    )

    dropdown_choices = [f"{i + 1}: {img}" for i, img in enumerate(filtered_images)]
    dropdown_update = gr.update(choices=dropdown_choices, value=dropdown_choices[0])

    outputs = [
        filtered_images,
        0,
        ui_updates[2],
        ui_updates[0],
        ui_updates[3],
        dropdown_update,
        ui_updates[5],
        ui_updates[6],
        ui_updates[7],
        ui_updates[8],
        ui_updates[9],
    ]
    return outputs


def handle_box_selector_change(box_selector_val, boxes):
    if not box_selector_val or not boxes:
        return -1, gr.skip(), gr.skip(), gr.skip()

    try:
        parts = box_selector_val.split(":", 1)
        lbl_part = parts[0].replace("Box #", "").strip()
        idx = int(lbl_part) - 1
    except:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()

    if idx < 0 or idx >= len(boxes):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()

    box = boxes[idx]
    keep_val = "Keep" if box["keep"] else "Delete"

    return idx, True, keep_val, box["class_name"]


def handle_single_box_edit(
    is_updating,
    selected_idx,
    boxes,
    current_index,
    images,
    base_path,
    classes,
    keep_val,
    class_val,
):
    if is_updating or selected_idx < 0 or not boxes or selected_idx >= len(boxes):
        return gr.skip(), gr.skip(), gr.skip()

    box = boxes[selected_idx]
    expected_keep = keep_val == "Keep"
    updated = False

    if box["keep"] != expected_keep:
        box["keep"] = expected_keep
        updated = True

    if class_val and box["class_name"] != class_val:
        box["class_name"] = class_val
        if class_val in classes:
            box["class_id"] = classes.index(class_val)
        updated = True

    if not updated:
        return gr.skip(), gr.skip(), gr.skip()

    img_name = images[current_index]
    save_boxes(base_path, img_name, boxes)

    image_path = os.path.join(base_path, "images", img_name)
    annotated_img = draw_image_boxes(image_path, boxes)

    selector_choices = [
        f"Box #{i + 1}: {b['class_name']}{'' if b['keep'] else ' [DELETE]'}"
        for i, b in enumerate(boxes)
    ]
    selector_val = selector_choices[selected_idx]
    selector_update = gr.update(choices=selector_choices, value=selector_val)

    return annotated_img, boxes, selector_update


css = """
body {
    background-color: #0f172a;
    color: #f8fafc;
}
.gradio-container {
    max-width: 98% !important;
    width: 98% !important;
    margin: 0 auto !important;
}
.save-btn {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.save-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}
.nav-btn {
    font-weight: 600 !important;
    border: 1px solid #334155 !important;
    transition: all 0.2s ease !important;
}
.nav-btn:hover {
    background-color: #334155 !important;
}
.annotation-panel {
    background-color: #1e293b;
    border-radius: 8px;
    padding: 16px;
    border: 1px solid #334155;
}
"""

theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
)

with gr.Blocks(title="YOLO Annotation Reviewer") as demo:
    gr.Markdown("# YOLO Annotation Reviewer")
    gr.Markdown(
        "Load any YOLO dataset, filter by class, step through images, review labels, and **change classes** or delete bounding boxes on the fly. "
        "**Changes are auto-saved** when you navigate."
    )

    images_state = gr.State([])
    raw_images_state = gr.State([])
    classes_state = gr.State([])
    current_index_state = gr.State(0)
    boxes_state = gr.State([])
    selected_box_index_state = gr.State(-1)
    is_updating_state = gr.State(False)

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                base_path_input = gr.Textbox(
                    label="Dataset Train/Val Path",
                    placeholder="/absolute/path/to/dataset/train",
                )
                with gr.Row():
                    filter_class_drop = gr.Dropdown(
                        choices=["All"],
                        value="All",
                        label="Filter by Class",
                        interactive=True,
                        scale=2,
                    )
                    load_btn = gr.Button("Load Dataset", variant="primary", scale=1)

            status_msg = gr.Markdown("Enter path and click **Load Dataset** to begin.")

            jump_dropdown = gr.Dropdown(
                label="Search / Select Image",
                choices=[],
                interactive=True,
                filterable=True,
            )

            progress_info = gr.Markdown("No dataset loaded.")

            image_display = gr.Image(type="numpy", interactive=False)

        with gr.Column(scale=2):
            with gr.Column(elem_classes="annotation-panel"):
                gr.Markdown("### Bounding Box Controls")

                box_selector = gr.Dropdown(
                    choices=[],
                    label="Select Bounding Box to Edit",
                    interactive=True,
                    visible=False,
                )

                box_keep_radio = gr.Radio(
                    choices=["Keep", "Delete"],
                    value="Keep",
                    label="Status",
                    interactive=True,
                    visible=False,
                )

                box_class_dropdown = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Class",
                    interactive=True,
                    allow_custom_value=True,
                    visible=False,
                )

            with gr.Row():
                prev_btn = gr.Button("⬅️ Previous Image", elem_classes="nav-btn")
                next_btn = gr.Button("Next Image ➡️", elem_classes="nav-btn")

            with gr.Row():
                save_btn = gr.Button(
                    "💾 Save Current Image Changes", elem_classes="save-btn"
                )

            save_status = gr.Markdown("")

    load_btn.click(
        fn=handle_load_dataset,
        inputs=[base_path_input],
        outputs=[
            images_state,
            raw_images_state,
            classes_state,
            status_msg,
            jump_dropdown,
            filter_class_drop,
            image_display,
            current_index_state,
            boxes_state,
            progress_info,
            save_status,
            selected_box_index_state,
            box_selector,
            box_keep_radio,
            box_class_dropdown,
        ],
    )

    filter_class_drop.change(
        fn=handle_class_filter,
        inputs=[
            base_path_input,
            raw_images_state,
            classes_state,
            current_index_state,
            boxes_state,
            filter_class_drop,
        ],
        outputs=[
            images_state,
            current_index_state,
            boxes_state,
            image_display,
            progress_info,
            jump_dropdown,
            save_status,
            selected_box_index_state,
            box_selector,
            box_keep_radio,
            box_class_dropdown,
        ],
    )

    nav_outputs = [
        image_display,
        current_index_state,
        boxes_state,
        progress_info,
        jump_dropdown,
        save_status,
        selected_box_index_state,
        box_selector,
        box_keep_radio,
        box_class_dropdown,
    ]

    next_btn.click(
        fn=handle_navigate,
        inputs=[
            base_path_input,
            images_state,
            classes_state,
            current_index_state,
            boxes_state,
            gr.State("next"),
        ],
        outputs=nav_outputs,
    )

    prev_btn.click(
        fn=handle_navigate,
        inputs=[
            base_path_input,
            images_state,
            classes_state,
            current_index_state,
            boxes_state,
            gr.State("prev"),
        ],
        outputs=nav_outputs,
    )

    jump_dropdown.change(
        fn=handle_navigate,
        inputs=[
            base_path_input,
            images_state,
            classes_state,
            current_index_state,
            boxes_state,
            jump_dropdown,
        ],
        outputs=nav_outputs,
    )

    save_btn.click(
        fn=handle_save,
        inputs=[base_path_input, images_state, current_index_state, boxes_state],
        outputs=[save_status],
    )

    box_selector.input(
        fn=handle_box_selector_change,
        inputs=[box_selector, boxes_state],
        outputs=[
            selected_box_index_state,
            is_updating_state,
            box_keep_radio,
            box_class_dropdown,
        ],
    ).then(fn=lambda: False, inputs=[], outputs=[is_updating_state])

    box_keep_radio.change(
        fn=handle_single_box_edit,
        inputs=[
            is_updating_state,
            selected_box_index_state,
            boxes_state,
            current_index_state,
            images_state,
            base_path_input,
            classes_state,
            box_keep_radio,
            box_class_dropdown,
        ],
        outputs=[image_display, boxes_state, box_selector],
    )
    box_class_dropdown.change(
        fn=handle_single_box_edit,
        inputs=[
            is_updating_state,
            selected_box_index_state,
            boxes_state,
            current_index_state,
            images_state,
            base_path_input,
            classes_state,
            box_keep_radio,
            box_class_dropdown,
        ],
        outputs=[image_display, boxes_state, box_selector],
    )

    js_code = """
    () => {
        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) {
                return;
            }
            const key = e.key.toLowerCase();
            if (key === 'arrowright' || key === 'n') {
                const btn = Array.from(document.querySelectorAll('button')).find(el => el.textContent.includes('Next Image'));
                if (btn) btn.click();
            } else if (key === 'arrowleft' || key === 'p' || key === 'b') {
                const btn = Array.from(document.querySelectorAll('button')).find(el => el.textContent.includes('Previous Image'));
                if (btn) btn.click();
            } else if (key === 'd') {
                const spans = Array.from(document.querySelectorAll('span'));
                const deleteLabel = spans.find(el => el.textContent.trim() === 'Delete');
                if (deleteLabel) deleteLabel.click();
            } else if (key === 'k') {
                const spans = Array.from(document.querySelectorAll('span'));
                const keepLabel = spans.find(el => el.textContent.trim() === 'Keep');
                if (keepLabel) keepLabel.click();
            }
        });
    }
    """
    demo.load(js=js_code)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=theme, css=css)
