import gradio as gr
import pandas as pd
import cv2
import os

def load_csv(csv_path):
    if not os.path.exists(csv_path):
        return None, 0, f"Error: File {csv_path} not found.", {}, []
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, 0, f"Error reading CSV: {e}", {}, []
        
    if "image_path" not in df.columns:
        return None, 0, "Error: CSV must contain 'image_path' column", {}, []
        
    unique_images = df["image_path"].unique().tolist()
    if not unique_images:
        return None, 0, "No images found in CSV.", {}, []
        
    # Attempt to load existing evaluations if they exist
    eval_path = csv_path.replace('.csv', '_evaluations.csv')
    evaluations = {}
    if os.path.exists(eval_path):
        try:
            eval_df = pd.read_csv(eval_path)
            evaluations = dict(zip(eval_df['image_path'], eval_df['evaluation']))
        except:
            pass
            
    return df, 0, f"Successfully loaded {len(unique_images)} images.", evaluations, unique_images

def draw_boxes(image_path, df):
    image = cv2.imread(image_path)
    if image is None:
        # Return a blank image if the original cannot be found
        import numpy as np
        blank = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.putText(blank, "Image not found", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return blank
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes = df[df["image_path"] == image_path]
    for _, row in boxes.iterrows():
        try:
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['class_name']} ({row.get('confidence', 0):.2f})"
            
            thickness = max(2, int(max(image.shape[0], image.shape[1]) / 500))
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness)
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, thickness/3), (0, 255, 0), thickness)
        except Exception as e:
            print(f"Error drawing box: {e}")
            
    return image

def update_ui(unique_images, df, index, evaluations):
    if not unique_images or index >= len(unique_images) or index < 0:
        return None, "No image to display", 1
        
    img_path = unique_images[index]
    
    if type(df) is dict:
        # Gradio strips DataFrames sometimes depending on state transfer, so we use the raw state
        df = pd.DataFrame(df)
        
    img = draw_boxes(img_path, df)
    
    status = evaluations.get(img_path, "Not evaluated yet")
    progress_text = f"**Image {index + 1} of {len(unique_images)}**\n\n**Path:** `{img_path}`\n\n**Current Status:** {status}"
    
    return img, progress_text, index + 1

def next_image(index, unique_images):
    if unique_images and index < len(unique_images) - 1:
        return index + 1
    return index

def prev_image(index):
    if index > 0:
        return index - 1
    return index

def jump_to_image(jump_num, unique_images):
    if not unique_images or jump_num is None:
        return 0
    try:
        idx = int(jump_num) - 1
        idx = max(0, min(idx, len(unique_images) - 1))
        return idx
    except (ValueError, TypeError):
        return 0

def save_evaluations(csv_path, evaluations):
    if not csv_path:
        return
    eval_path = csv_path.replace('.csv', '_evaluations.csv')
    eval_list = [{"image_path": k, "evaluation": v} for k, v in evaluations.items()]
    pd.DataFrame(eval_list).to_csv(eval_path, index=False)

def flag_image(evaluation, index, unique_images, evaluations, csv_path):
    if not unique_images or index >= len(unique_images) or index < 0:
        return index, evaluations
        
    img_path = unique_images[index]
    evaluations[img_path] = evaluation
    save_evaluations(csv_path, evaluations)
    
    # Auto advance
    return next_image(index, unique_images), evaluations

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Leopard Toad YOLO Detection Visualizer & Evaluator")
    
    with gr.Row():
        csv_input = gr.Textbox(label="CSV File Path", placeholder="Enter full path to .csv file (e.g. /home/.../results/detect_2/.../3Z.csv)")
        load_btn = gr.Button("Load CSV", variant="primary", scale=0)
        
    status_msg = gr.Markdown("Please load a CSV file to start.")
    
    # States
    df_state = gr.State(pd.DataFrame())
    index_state = gr.State(0)
    evals_state = gr.State({})
    images_state = gr.State([])
    
    with gr.Row():
        progress_label = gr.Markdown("Progress info will appear here.")
    
    image_output = gr.Image(label="Detections", type="numpy", interactive=False)
    
    with gr.Row():
        prev_btn = gr.Button("⬅️ Previous")
        jump_num = gr.Number(label="Jump to Image #", precision=0, minimum=1, show_label=True, step=1, value=1)
        jump_btn = gr.Button("Jump")
        next_btn = gr.Button("Next ➡️")
        
    with gr.Row():
        correct_btn = gr.Button("✅ Flag as Correct", variant="primary")
        incorrect_btn = gr.Button("❌ Flag as Incorrect", variant="stop")
        
    def handle_load(csv_path):
        df, idx, msg, evals, imgs = load_csv(csv_path)
        if df is None:
            return pd.DataFrame(), idx, msg, evals, imgs
        return df, idx, msg, evals, imgs

    # Handlers
    load_btn.click(
        handle_load,
        inputs=[csv_input],
        outputs=[df_state, index_state, status_msg, evals_state, images_state]
    ).then(
        update_ui,
        inputs=[images_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num]
    )
    
    next_btn.click(
        next_image,
        inputs=[index_state, images_state],
        outputs=[index_state]
    ).then(
        update_ui,
        inputs=[images_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num]
    )
    
    prev_btn.click(
        prev_image,
        inputs=[index_state],
        outputs=[index_state]
    ).then(
        update_ui,
        inputs=[images_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num]
    )
    
    jump_btn.click(
        jump_to_image,
        inputs=[jump_num, images_state],
        outputs=[index_state]
    ).then(
        update_ui,
        inputs=[images_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num]
    )

    jump_num.submit(
        jump_to_image,
        inputs=[jump_num, images_state],
        outputs=[index_state]
    ).then(
        update_ui,
        inputs=[images_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num]
    )
    
    correct_btn.click(
        lambda idx, imgs, evals, path: flag_image("Correct", idx, imgs, evals, path),
        inputs=[index_state, images_state, evals_state, csv_input],
        outputs=[index_state, evals_state]
    ).then(
        update_ui,
        inputs=[images_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num]
    )
    
    incorrect_btn.click(
        lambda idx, imgs, evals, path: flag_image("Incorrect", idx, imgs, evals, path),
        inputs=[index_state, images_state, evals_state, csv_input],
        outputs=[index_state, evals_state]
    ).then(
        update_ui,
        inputs=[images_state, df_state, index_state, evals_state],
        outputs=[image_output, progress_label, jump_num]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
