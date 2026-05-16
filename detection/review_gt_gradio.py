import gradio as gr
import pandas as pd
import cv2
import os
import numpy as np
import sys

# Add evaluation directory to sys.path for project-specific imports
project_root = "/home/Joshua/Downloads/leopard_toad_identification/detection/evaluation"
if project_root not in sys.path:
    sys.path.append(project_root)

from eval_utils.config import CLASSES


def load_gt_data(root_dir):
    if not root_dir or not os.path.exists(root_dir):
        return None, "Error: Directory does not exist.", {}

    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        return (
            None,
            f"Error: Ensure 'images' and 'labels' folders exist in {root_dir}",
            {},
        )

    all_data = []
    image_files = sorted(
        [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id, xn, yn, wn, hn = map(float, parts[:5])
                            all_data.append(
                                {
                                    "image_path": img_path,
                                    "label_path": label_path,
                                    "cls": int(cls_id),
                                    "yolo_bbox": [xn, yn, wn, hn],
                                    "line_idx": line_idx,
                                    "image_name": img_name,
                                }
                            )
                        except ValueError:
                            continue
        else:
            # Optionally include images with no labels
            pass

    if not all_data:
        return None, "No annotations found in the directory.", {}

    df = pd.DataFrame(all_data)

    # Load existing evaluations if any
    eval_path = os.path.join(root_dir, "gt_review_results.csv")
    evaluations = {}
    if os.path.exists(eval_path):
        try:
            eval_df = pd.read_csv(eval_path)
            for _, row in eval_df.iterrows():
                key = f"{row['image_name']}_{row['line_idx']}"
                evaluations[key] = row["evaluation"]
        except:
            pass

    return (
        df,
        f"Successfully loaded {len(all_data)} annotations from {len(image_files)} images.",
        evaluations,
    )


def draw_gt_boxes(df_row, df_full):
    img_path = df_row["image_path"]
    image = cv2.imread(img_path)
    if image is None:
        blank = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            "Image not found",
            (50, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        return blank

    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get all boxes for this image to show context
    img_boxes = df_full[df_full["image_path"] == img_path]

    for _, row in img_boxes.iterrows():
        xn, yn, wn, hn = row["yolo_bbox"]
        x1 = int((xn - wn / 2) * w)
        y1 = int((yn - hn / 2) * h)
        x2 = int((xn + wn / 2) * w)
        y2 = int((yn + hn / 2) * h)

        is_current = row["line_idx"] == df_row["line_idx"]
        color = (
            (255, 0, 0) if is_current else (0, 255, 0)
        )  # Red for current, Green for others
        thickness = (
            max(3, int(max(h, w) / 400)) if is_current else max(2, int(max(h, w) / 600))
        )

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label_text = f"{CLASSES.get(row['cls'], row['cls'])}"
        if is_current:
            label_text = f"CURRENT: {label_text}"

        cv2.putText(
            image,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.6, thickness / 3),
            color,
            max(2, thickness // 2),
        )

    return image


def update_ui(df, index, evaluations):
    if df is None or df.empty or index >= len(df) or index < 0:
        return None, "No data to display", 1

    row = df.iloc[index]
    img = draw_gt_boxes(row, df)

    key = f"{row['image_name']}_{row['line_idx']}"
    status = evaluations.get(key, "Not evaluated yet")

    progress_text = f"**Annotation {index + 1} of {len(df)}**\n\n"
    progress_text += f"**Image:** `{row['image_name']}`\n\n"
    progress_text += f"**Class:** {CLASSES.get(row['cls'], row['cls'])}\n\n"
    progress_text += f"**Status:** {status}"

    return img, progress_text, index + 1


def save_evaluations(root_dir, evaluations, df):
    if not root_dir or df is None or df.empty:
        return

    eval_path = os.path.join(root_dir, "gt_review_results.csv")
    eval_list = []

    # Use the dataframe to ensure we have all fields
    for _, row in df.iterrows():
        key = f"{row['image_name']}_{row['line_idx']}"
        if key in evaluations:
            eval_list.append(
                {
                    "image_name": row["image_name"],
                    "line_idx": row["line_idx"],
                    "evaluation": evaluations[key],
                    "image_path": row["image_path"],
                    "class": CLASSES.get(row["cls"], row["cls"]),
                }
            )

    if eval_list:
        pd.DataFrame(eval_list).to_csv(eval_path, index=False)


def flag_item(evaluation, index, evaluations, root_dir, df):
    if df is None or df.empty or index >= len(df):
        return index, evaluations

    row = df.iloc[index]
    key = f"{row['image_name']}_{row['line_idx']}"
    evaluations[key] = evaluation

    save_evaluations(root_dir, evaluations, df)

    # Auto-advance
    new_index = min(index + 1, len(df) - 1)
    return new_index, evaluations


# Main App Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# YOLO Ground Truth Reviewer")
    gr.Markdown(
        "Load a directory containing `images/` and `labels/` to manually verify annotations."
    )

    with gr.Row():
        dir_input = gr.Textbox(
            label="Root Directory Path",
            placeholder="/home/Joshua/.../detection/evaluation/data/test",
            scale=4,
        )
        load_btn = gr.Button("Load Directory", variant="primary", scale=1)

    status_msg = gr.Markdown("Please enter a directory path to start.")

    # States
    df_state = gr.State(None)
    index_state = gr.State(0)
    evals_state = gr.State({})

    with gr.Row():
        progress_label = gr.Markdown("Progress info will appear here.")

    image_output = gr.Image(label="Current Annotation", type="numpy", interactive=False)

    with gr.Row():
        prev_btn = gr.Button("⬅️ Previous")
        jump_num = gr.Number(
            label="Jump to Annotation #",
            precision=0,
            minimum=1,
            show_label=True,
            step=1,
            value=1,
        )
        jump_btn = gr.Button("Jump")
        next_btn = gr.Button("Next ➡️")

    with gr.Row():
        correct_btn = gr.Button("✅ Correct", variant="primary")
        incorrect_btn = gr.Button("❌ Incorrect", variant="stop")
        unsure_btn = gr.Button("❓ Unsure / Need Fix", variant="secondary")

    # Event Handlers
    def handle_load(path):
        df, msg, evals = load_gt_data(path)
        return df, 0, msg, evals

    load_btn.click(
        handle_load,
        inputs=[dir_input],
        outputs=[df_state, index_state, status_msg, evals_state],
    ).then(
        update_ui,
        inputs=[df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    def go_next(idx, df):
        if df is not None and idx < len(df) - 1:
            return idx + 1
        return idx

    def go_prev(idx):
        return max(0, idx - 1)

    next_btn.click(go_next, inputs=[index_state, df_state], outputs=[index_state]).then(
        update_ui,
        inputs=[df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    prev_btn.click(go_prev, inputs=[index_state], outputs=[index_state]).then(
        update_ui,
        inputs=[df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    jump_btn.click(
        lambda j, df: max(0, min(int(j) - 1, len(df) - 1)) if df is not None else 0,
        inputs=[jump_num, df_state],
        outputs=[index_state],
    ).then(
        update_ui,
        inputs=[df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    correct_btn.click(
        lambda idx, evals, path, df: flag_item("Correct", idx, evals, path, df),
        inputs=[index_state, evals_state, dir_input, df_state],
        outputs=[index_state, evals_state],
    ).then(
        update_ui,
        inputs=[df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    incorrect_btn.click(
        lambda idx, evals, path, df: flag_item("Incorrect", idx, evals, path, df),
        inputs=[index_state, evals_state, dir_input, df_state],
        outputs=[index_state, evals_state],
    ).then(
        update_ui,
        inputs=[df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    unsure_btn.click(
        lambda idx, evals, path, df: flag_item("Unsure", idx, evals, path, df),
        inputs=[index_state, evals_state, dir_input, df_state],
        outputs=[index_state, evals_state],
    ).then(
        update_ui,
        inputs=[df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
