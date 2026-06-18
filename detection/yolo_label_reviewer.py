import os
import cv2
import json
import numpy as np
import pandas as pd
import gradio as gr

# Maximum number of annotations we support reviewing per image
MAX_BOXES = 20

# Default classes
DEFAULT_CLASSES = ["Other_Amphibian", "Small_Mammal", "Western_Leopard_Toad"]


def load_classes(base_path):
    """
    Robustly load classes from classes.txt or notes.json.
    """
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
            print(f"Error parsing notes.json: {e}")

    return DEFAULT_CLASSES


def load_dataset(base_path):
    """
    Validate base path, load images list and classes list.
    """
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

    # Load classes
    classes = load_classes(base_path)

    # Scan for images
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
    """
    Load YOLO format boxes from labels/ directory.
    """
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
    """
    Save keeping boxes to YOLO label file. Delete file or empty it if no boxes left.
    """
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
    """
    Draw bounding boxes on the image. Green for Keep, Red for Delete.
    """
    img = cv2.imread(image_path)
    if img is None:
        # Create a nice dark fallback canvas
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        canvas[:, :] = (15, 23, 42)  # Slate-900 background
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

    # Optimize performance by downscaling large images before drawing & rendering
    h, w, _ = img.shape
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Adaptive text scaling & thickness based on image size
    thickness = max(2, int(max(h, w) / 500))
    font_scale = max(0.5, thickness / 3)

    for idx, box in enumerate(boxes):
        x_c, y_c, bw, bh = box["coords"]
        xmin = int((x_c - bw / 2) * w)
        ymin = int((y_c - bh / 2) * h)
        xmax = int((x_c + bw / 2) * w)
        ymax = int((y_c + bh / 2) * h)

        # Clamp boundaries
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))

        if box["keep"]:
            color = (46, 204, 113)  # Emerald Green
            status_suffix = ""
        else:
            color = (231, 76, 60)  # Alizarin Red
            status_suffix = " [DELETE]"

        # Draw box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

        # Draw box label with background box
        conf_str = (
            f" ({box['confidence']:.2%})" if box["confidence"] is not None else ""
        )
        label_text = f"#{idx + 1}: {box['class_name']}{conf_str}{status_suffix}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness // 2 + 1
        )

        # Label background
        label_ymin = max(0, ymin - text_h - 12)
        cv2.rectangle(img, (xmin, label_ymin), (xmin + text_w + 6, ymin), color, -1)

        # Put label text
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


def get_dataframe_data(boxes):
    """
    Format current boxes info as a pandas DataFrame for nice rendering.
    """
    if not boxes:
        return pd.DataFrame(
            columns=[
                "Box #",
                "Class Name",
                "Confidence",
                "Coordinates (x,y,w,h)",
                "Status",
            ]
        )

    data = []
    for idx, box in enumerate(boxes):
        status = "✅ Keep" if box["keep"] else "❌ Delete"
        coords_str = ", ".join(f"{x:.4f}" for x in box["coords"])
        conf_str = (
            f"{box['confidence']:.2%}" if box["confidence"] is not None else "N/A"
        )
        data.append(
            {
                "Box #": idx + 1,
                "Class Name": box["class_name"],
                "Confidence": conf_str,
                "Coordinates (x,y,w,h)": coords_str,
                "Status": status,
            }
        )
    return pd.DataFrame(data)


def update_ui_for_image(base_path, image_name, classes, index, total_images):
    """
    Load data for image at new index and generate component outputs.
    """
    image_path = os.path.join(base_path, "images", image_name)
    boxes = load_boxes(base_path, image_name, classes)
    annotated_img = draw_image_boxes(image_path, boxes)

    progress_text = f"**Image {index + 1} of {total_images}**: `{image_name}`"
    jump_dropdown_val = f"{index + 1}: {image_name}"
    df_data = get_dataframe_data(boxes)

    outputs = [
        annotated_img,
        index,
        boxes,
        progress_text,
        jump_dropdown_val,
        df_data,
        "",
    ]

    # Generate visibility & values for all MAX_BOXES control rows
    for i in range(MAX_BOXES):
        if i < len(boxes):
            box = boxes[i]
            conf_str = (
                f" ({box['confidence']:.1%})" if box["confidence"] is not None else ""
            )
            lbl_val = f"**Box #{i + 1}{conf_str}**"
            outputs.append(gr.Row(visible=True))
            outputs.append(lbl_val)
            outputs.append("Keep")
            outputs.append(
                gr.Dropdown(choices=classes, value=box["class_name"], visible=True)
            )
        else:
            outputs.append(gr.Row(visible=False))
            outputs.append("")
            outputs.append("Keep")
            outputs.append(gr.Dropdown(choices=[], value=None, visible=False))

    return outputs


def handle_load_dataset(base_path):
    """
    Handle click on Load Dataset. Sets both raw_images_state and images_state,
    resets filter dropdown to "All" and populated choices list.
    """
    images, classes, msg, dropdown_update = load_dataset(base_path)
    if not images:
        return [
            pd.DataFrame(),
            0,
            [],
            [],
            [],
            "",
            dropdown_update,
            gr.update(choices=["All"], value="All"),
            None,
            msg,
        ] + [gr.update(visible=False) for _ in range(MAX_BOXES * 4)]

    # Load first image of full list
    first_img = images[0]
    ui_updates = update_ui_for_image(base_path, first_img, classes, 0, len(images))

    # Update filter choices
    filter_dropdown_update = gr.update(choices=["All"] + classes, value="All")

    # Returns [images_state, raw_images_state, classes_state, msg, jump_dropdown, filter_class_drop] + UI updates
    return [
        images,
        images,
        classes,
        msg,
        dropdown_update,
        filter_dropdown_update,
    ] + ui_updates


def handle_navigate(base_path, images, classes, current_index, boxes, action):
    """
    Handle clicking next/prev or selecting a new image from dropdown.
    """
    if not images:
        return [gr.skip()] * (7 + MAX_BOXES * 4)

    # 1. Auto-save current annotations
    current_img = images[current_index]
    save_boxes(base_path, current_img, boxes)

    # 2. Compute new index
    new_index = current_index
    if action == "next":
        new_index = min(len(images) - 1, current_index + 1)
    elif action == "prev":
        new_index = max(0, current_index - 1)
    elif isinstance(action, str) and ":" in action:
        # Selecting from dropdown (format: "index: filename")
        try:
            parts = action.split(":", 1)
            new_index = int(parts[0]) - 1
        except:
            pass

    # 3. Load UI for new image
    ui_updates = update_ui_for_image(
        base_path, images[new_index], classes, new_index, len(images)
    )
    return ui_updates


def handle_save(base_path, images, current_index, boxes):
    """
    Manually save the current image annotation boxes.
    """
    if not images or current_index < 0 or current_index >= len(images) or not boxes:
        return "No image annotations to save."
    img_name = images[current_index]
    msg = save_boxes(base_path, img_name, boxes)
    return f"Manual save for {img_name}: {msg}"


def handle_class_filter(
    base_path, raw_images, classes, current_index, current_boxes, filter_class
):
    """
    Auto-saves current image annotations, filters images based on selected class,
    resets navigation, and loads the first image of the filtered list.
    """
    if not raw_images:
        return [gr.skip()] * (11 + MAX_BOXES * 4)

    # 1. Auto-save current annotations
    if 0 <= current_index < len(raw_images) and current_boxes is not None:
        save_boxes(base_path, raw_images[current_index], current_boxes)

    # 2. Filter images list
    if not filter_class or filter_class == "All":
        filtered_images = list(raw_images)
    else:
        filtered_images = []
        for img in raw_images:
            boxes = load_boxes(base_path, img, classes)
            if any(box["class_name"] == filter_class for box in boxes):
                filtered_images.append(img)

    # 3. Handle empty filtered list
    if not filtered_images:
        empty_msg = f"No images contain class '{filter_class}'."
        outputs = [
            [],  # images_state
            0,  # current_index_state
            [],  # boxes_state
            None,  # image_display
            f"**0 of 0**: {empty_msg}",  # progress_info
            gr.update(choices=[], value=None),  # jump_dropdown
            pd.DataFrame(),  # data_table
            empty_msg,  # save_status
        ]
        for _ in range(MAX_BOXES):
            outputs.extend(
                [gr.Row(visible=False), "", "Keep", gr.Dropdown(visible=False)]
            )
        return outputs

    # 4. Load first image of the filtered list
    new_index = 0
    ui_updates = update_ui_for_image(
        base_path, filtered_images[new_index], classes, new_index, len(filtered_images)
    )

    dropdown_choices = [f"{i + 1}: {img}" for i, img in enumerate(filtered_images)]
    dropdown_update = gr.update(choices=dropdown_choices, value=dropdown_choices[0])

    outputs = [
        filtered_images,  # images_state
        0,  # current_index_state
        ui_updates[2],  # boxes_state (boxes)
        ui_updates[0],  # image_display (annotated_img)
        ui_updates[3],  # progress_info (progress_text)
        dropdown_update,  # jump_dropdown
        ui_updates[5],  # data_table (df_data)
        ui_updates[6],  # save_status ("")
    ]
    outputs.extend(ui_updates[7:])
    return outputs


def on_control_change(boxes, current_index, images, base_path, classes, *control_vals):
    """
    Fires whenever any of the radio buttons or class dropdowns changes value.
    Updates keeping flags and classes, redraws image, and updates Pandas DataFrame.
    """
    if not boxes or not images or current_index >= len(images):
        return gr.skip(), gr.skip(), gr.skip()

    # control_vals split into radio values and dropdown values
    radio_vals = control_vals[:MAX_BOXES]
    dropdown_vals = control_vals[MAX_BOXES:]

    updated = False
    for i in range(min(len(boxes), MAX_BOXES)):
        # 1. Check keep status
        radio_val = radio_vals[i]
        expected_keep = radio_val == "Keep"
        if boxes[i]["keep"] != expected_keep:
            boxes[i]["keep"] = expected_keep
            updated = True

        # 2. Check class value
        dropdown_val = dropdown_vals[i]
        if dropdown_val and boxes[i]["class_name"] != dropdown_val:
            boxes[i]["class_name"] = dropdown_val
            # Update class_id to match list index of loaded classes
            if dropdown_val in classes:
                boxes[i]["class_id"] = classes.index(dropdown_val)
            updated = True

    if not updated:
        return gr.skip(), gr.skip(), gr.skip()

    # Instant auto-save to disk
    img_name = images[current_index]
    save_boxes(base_path, img_name, boxes)

    # Redraw
    image_path = os.path.join(base_path, "images", img_name)
    annotated_img = draw_image_boxes(image_path, boxes)
    df_data = get_dataframe_data(boxes)

    return annotated_img, boxes, df_data


# --- GRADIO INTERFACE ---

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
    # App Header
    gr.Markdown("# 🐸 Western Leopard Toad YOLO Annotation Reviewer")
    gr.Markdown(
        "Load any YOLO dataset, filter by class, step through images, review labels, and **change classes** or delete bounding boxes on the fly. "
        "**Changes are auto-saved** when you navigate."
    )

    # State variables
    images_state = gr.State([])  # Filtered images list
    raw_images_state = gr.State([])  # Unfiltered original images list
    classes_state = gr.State([])
    current_index_state = gr.State(0)
    boxes_state = gr.State([])

    with gr.Row():
        # Left Panel: Path input and list selector
        with gr.Column(scale=3):
            with gr.Group():
                base_path_input = gr.Textbox(
                    label="Dataset Train/Val Path",
                    placeholder="/absolute/path/to/dataset/train",
                    value="/home/Joshua/Downloads/leopard_toad_identification/detection/active learning/data/detect_1/train",
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

            # Image selection dropdown/search
            jump_dropdown = gr.Dropdown(
                label="Search / Select Image",
                choices=[],
                interactive=True,
                filterable=True,
            )

            progress_info = gr.Markdown("No dataset loaded.")

            # Image display
            image_display = gr.Image(
                label="Reviewed Image (Real-time Feedback)",
                type="numpy",
                interactive=False,
            )

        # Right Panel: Controls
        with gr.Column(scale=2):
            with gr.Column(elem_classes="annotation-panel"):
                gr.Markdown("### Bounding Box Controls")

                # Dynamic controls lists
                box_rows = []
                box_labels = []
                box_radios = []
                box_dropdowns = []

                for idx in range(MAX_BOXES):
                    with gr.Row(visible=False, variant="compact") as row:
                        lbl = gr.Markdown(f"**Box #{idx + 1}**")
                        # Panel/button style for Keep/Delete
                        rad = gr.Radio(
                            choices=["Keep", "Delete"],
                            value="Keep",
                            show_label=False,
                            interactive=True,
                            scale=2,
                        )
                        cls_drop = gr.Dropdown(
                            choices=DEFAULT_CLASSES,
                            value=None,
                            show_label=False,
                            interactive=True,
                            container=False,
                            scale=3,
                            allow_custom_value=True,
                        )
                        box_rows.append(row)
                        box_labels.append(lbl)
                        box_radios.append(rad)
                        box_dropdowns.append(cls_drop)

            # Action & Navigation buttons
            with gr.Row():
                prev_btn = gr.Button("⬅️ Previous Image", elem_classes="nav-btn")
                next_btn = gr.Button("Next Image ➡️", elem_classes="nav-btn")

            with gr.Row():
                save_btn = gr.Button(
                    "💾 Save Current Image Changes", elem_classes="save-btn"
                )

            save_status = gr.Markdown("")

            # Bounding Box detailed Table
            gr.Markdown("### Image Annotations Table")
            data_table = gr.Dataframe(
                headers=[
                    "Box #",
                    "Class Name",
                    "Confidence",
                    "Coordinates (x,y,w,h)",
                    "Status",
                ],
                interactive=False,
                wrap=True,
            )

    # Define Event Listeners

    # 1. Loading dataset
    # Output schema:
    # [images_state, raw_images_state, classes_state, status_msg, jump_dropdown, filter_class_drop, image_display, current_index_state, boxes_state, progress_info, jump_dropdown, data_table, save_status] + MAX_BOXES * [row, label, radio, dropdown]
    load_outputs = [
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
        jump_dropdown,
        data_table,
        save_status,
    ]
    for i in range(MAX_BOXES):
        load_outputs.extend(
            [box_rows[i], box_labels[i], box_radios[i], box_dropdowns[i]]
        )

    load_btn.click(
        fn=handle_load_dataset, inputs=[base_path_input], outputs=load_outputs
    )

    # 2. Class Filter selection
    # Output schema is exactly the same as handle_load_dataset outputs, minus raw_images_state and classes_state:
    # [images_state, current_index_state, boxes_state, image_display, progress_info, jump_dropdown, data_table, save_status] + dynamic controls
    filter_outputs = [
        images_state,
        current_index_state,
        boxes_state,
        image_display,
        progress_info,
        jump_dropdown,
        data_table,
        save_status,
    ]
    for i in range(MAX_BOXES):
        filter_outputs.extend(
            [box_rows[i], box_labels[i], box_radios[i], box_dropdowns[i]]
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
        outputs=filter_outputs,
    )

    # 3. Navigation & Dropdown Selection
    nav_outputs = [
        image_display,
        current_index_state,
        boxes_state,
        progress_info,
        jump_dropdown,
        data_table,
        save_status,
    ]
    for i in range(MAX_BOXES):
        nav_outputs.extend(
            [box_rows[i], box_labels[i], box_radios[i], box_dropdowns[i]]
        )

    # Next image
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

    # Previous image
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

    # Dropdown select
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

    # 4. Manual save button
    save_btn.click(
        fn=handle_save,
        inputs=[base_path_input, images_state, current_index_state, boxes_state],
        outputs=[save_status],
    )

    # 5. Live Radio and Dropdown changes
    for idx in range(MAX_BOXES):
        box_radios[idx].change(
            fn=on_control_change,
            inputs=[
                boxes_state,
                current_index_state,
                images_state,
                base_path_input,
                classes_state,
            ]
            + box_radios
            + box_dropdowns,
            outputs=[image_display, boxes_state, data_table],
        )
        box_dropdowns[idx].change(
            fn=on_control_change,
            inputs=[
                boxes_state,
                current_index_state,
                images_state,
                base_path_input,
                classes_state,
            ]
            + box_radios
            + box_dropdowns,
            outputs=[image_display, boxes_state, data_table],
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", server_port=7860, share=True, theme=theme, css=css
    )
