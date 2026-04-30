import gradio as gr
import pandas as pd
import cv2
import os


def load_csv(csv_path, show_reps_only=False):
    if not os.path.exists(csv_path):
        return None, 0, f"Error: File {csv_path} not found.", {}, []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, 0, f"Error reading CSV: {e}", {}, []

    if "image_path" not in df.columns:
        return None, 0, "Error: CSV must contain 'image_path' column", {}, []

    # Filter for representatives if requested
    if show_reps_only and "is_representative" in df.columns:
        df_filtered = df[df["is_representative"] == True]
    else:
        df_filtered = df

    items_to_evaluate = df_filtered.index.tolist()
    if not items_to_evaluate:
        return df, 0, "No matching annotations found in CSV.", {}, []

    # Attempt to load existing evaluations if they exist
    eval_path = csv_path.replace(".csv", "_evaluations.csv")
    evaluations = {}
    if os.path.exists(eval_path):
        try:
            eval_df = pd.read_csv(eval_path)
            if "row_idx" in eval_df.columns:
                for _, r in eval_df.iterrows():
                    if not pd.isna(r["row_idx"]):
                        evaluations[str(int(r["row_idx"]))] = r["evaluation"]
            else:
                for _, r in eval_df.iterrows():
                    evaluations[str(r["image_path"])] = r["evaluation"]
        except:
            pass

    return (
        df,
        0,
        f"Successfully loaded {len(items_to_evaluate)} annotations.",
        evaluations,
        items_to_evaluate,
    )


def draw_boxes(image_path, df, current_row_idx=None):
    image = cv2.imread(image_path)
    if image is None:
        # Return a blank image if the original cannot be found
        import numpy as np

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

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = df[df["image_path"] == image_path]
    for idx, row in boxes.iterrows():
        try:
            xmin, ymin, xmax, ymax = (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
            )

            if current_row_idx is not None and idx == current_row_idx:
                color = (255, 0, 0)  # Red for current evaluation
                label = f"CURRENT: {row['class_name']} ({row.get('confidence', 0):.2f})"
                thickness = max(3, int(max(image.shape[0], image.shape[1]) / 400))
            else:
                color = (0, 255, 0)  # Green for others
                label = f"{row['class_name']} ({row.get('confidence', 0):.2f})"
                thickness = max(2, int(max(image.shape[0], image.shape[1]) / 500))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
            cv2.putText(
                image,
                label,
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.5, thickness / 3),
                color,
                thickness,
            )
        except Exception as e:
            print(f"Error drawing box: {e}")

    return image


def update_ui(items_to_evaluate, df, index, evaluations):
    if not items_to_evaluate or index >= len(items_to_evaluate) or index < 0:
        return None, "No annotation to display", 1

    current_row_idx = items_to_evaluate[index]

    if type(df) is dict:
        # Gradio strips DataFrames sometimes depending on state transfer, so we use the raw state
        df = pd.DataFrame(df)

    img_path = df.loc[current_row_idx, "image_path"]

    img = draw_boxes(img_path, df, current_row_idx)

    status = evaluations.get(
        str(current_row_idx), evaluations.get(img_path, "Not evaluated yet")
    )
    progress_text = f"**Annotation {index + 1} of {len(items_to_evaluate)}**\n\n**Path:** `{img_path}`\n\n**Current Status:** {status}"

    return img, progress_text, index + 1


def next_item(index, items_to_evaluate):
    if items_to_evaluate and index < len(items_to_evaluate) - 1:
        return index + 1
    return index


def prev_item(index):
    if index > 0:
        return index - 1
    return index


def jump_to_item(jump_num, items_to_evaluate):
    if not items_to_evaluate or jump_num is None:
        return 0
    try:
        idx = int(jump_num) - 1
        idx = max(0, min(idx, len(items_to_evaluate) - 1))
        return idx
    except (ValueError, TypeError):
        return 0


def save_evaluations(csv_path, evaluations, df):
    if not csv_path:
        return
    eval_path = csv_path.replace(".csv", "_evaluations.csv")
    eval_list = []
    for k, v in evaluations.items():
        if str(k).isdigit():
            # it's a row index
            try:
                img_path = df.loc[int(k), "image_path"]
                eval_list.append(
                    {"image_path": img_path, "evaluation": v, "row_idx": int(k)}
                )
            except KeyError:
                pass
        else:
            # old format, image_path
            eval_list.append({"image_path": k, "evaluation": v})

    pd.DataFrame(eval_list).to_csv(eval_path, index=False)


def flag_item(evaluation, index, items_to_evaluate, evaluations, csv_path, df):
    if not items_to_evaluate or index >= len(items_to_evaluate) or index < 0:
        return index, evaluations

    if type(df) is dict:
        df = pd.DataFrame(df)

    current_row_idx = items_to_evaluate[index]
    evaluations[str(current_row_idx)] = evaluation
    save_evaluations(csv_path, evaluations, df)

    # Auto advance
    return next_item(index, items_to_evaluate), evaluations


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Leopard Toad YOLO Detection Visualizer & Evaluator")

    with gr.Row():
        csv_input = gr.Textbox(
            label="CSV File Path",
            placeholder="Enter full path to .csv file (e.g. /home/.../results/detect_2/.../3Z.csv)",
        )
        reps_checkbox = gr.Checkbox(
            label="Show Representative Boundary Cases Only", value=False
        )
        load_btn = gr.Button("Load CSV", variant="primary", scale=0)

    status_msg = gr.Markdown("Please load a CSV file to start.")

    # States
    df_state = gr.State(pd.DataFrame())
    index_state = gr.State(0)
    evals_state = gr.State({})
    items_state = gr.State([])

    with gr.Row():
        progress_label = gr.Markdown("Progress info will appear here.")

    image_output = gr.Image(label="Detections", type="numpy", interactive=False)

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
        correct_btn = gr.Button("✅ Flag as Correct", variant="primary")
        incorrect_btn = gr.Button("❌ Flag as Incorrect", variant="stop")
        missed_btn = gr.Button("⚠️ Missed Animal", variant="secondary")

    def handle_load(csv_path, show_reps):
        df, idx, msg, evals, items = load_csv(csv_path, show_reps)
        if df is None:
            return pd.DataFrame(), idx, msg, evals, items
        return df, idx, msg, evals, items

    # Handlers
    load_btn.click(
        handle_load,
        inputs=[csv_input, reps_checkbox],
        outputs=[df_state, index_state, status_msg, evals_state, items_state],
    ).then(
        update_ui,
        inputs=[items_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    reps_checkbox.change(
        handle_load,
        inputs=[csv_input, reps_checkbox],
        outputs=[df_state, index_state, status_msg, evals_state, items_state],
    ).then(
        update_ui,
        inputs=[items_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    next_btn.click(
        next_item, inputs=[index_state, items_state], outputs=[index_state]
    ).then(
        update_ui,
        inputs=[items_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    prev_btn.click(prev_item, inputs=[index_state], outputs=[index_state]).then(
        update_ui,
        inputs=[items_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    jump_btn.click(
        jump_to_item, inputs=[jump_num, items_state], outputs=[index_state]
    ).then(
        update_ui,
        inputs=[items_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    jump_num.submit(
        jump_to_item, inputs=[jump_num, items_state], outputs=[index_state]
    ).then(
        update_ui,
        inputs=[items_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    correct_btn.click(
        lambda idx, items, evals, path, d: flag_item(
            "Correct", idx, items, evals, path, d
        ),
        inputs=[index_state, items_state, evals_state, csv_input, df_state],
        outputs=[index_state, evals_state],
    ).then(
        update_ui,
        inputs=[items_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    incorrect_btn.click(
        lambda idx, items, evals, path, d: flag_item(
            "Incorrect", idx, items, evals, path, d
        ),
        inputs=[index_state, items_state, evals_state, csv_input, df_state],
        outputs=[index_state, evals_state],
    ).then(
        update_ui,
        inputs=[items_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

    missed_btn.click(
        lambda idx, items, evals, path, d: flag_item(
            "Missed Animal", idx, items, evals, path, d
        ),
        inputs=[index_state, items_state, evals_state, csv_input, df_state],
        outputs=[index_state, evals_state],
    ).then(
        update_ui,
        inputs=[items_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
