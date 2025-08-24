# main_gradio_final_fixed.py
# FIX: Resolved a critical bug where removing a filtered tag would delete
#      all other non-filtered tags. Changed the event listener for the
#      current tags list from `.input()` to the more precise `.select()`,
#      ensuring only the specifically unchecked tag is removed from the state.
# NEW: Added numeric input boxes for width and height for precise image resizing.
# NEW: Made the "Image Position" box interactive for direct navigation.
# FIX: Corrected a return value mismatch in the handle_folder_selection function
#      that caused a crash when loading a new folder.

import gradio as gr
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import csv
from collections import Counter
import re
import multiprocessing as mp
import queue

# --- Configuration ---
AI_THRESHOLD = 0.9999
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

# --- App State Class (For UI logic only) ---
class AppState:
    def __init__(self):
        self.image_files = []
        self.current_index = -1
        self.current_image_path = None
        self.current_tags = []
        self.folder_suggestions = []
        self.model_loaded = False
        self.current_ai_suggestions = []

# --- Independent AI Worker Process (No changes) ---
def inference_worker(task_queue: mp.Queue, result_queue: mp.Queue):
    model_session = None
    model_tags = []
    print("AI Worker Process Started.")
    while True:
        command, data = task_queue.get()
        if command == "LOAD":
            model_path, tags_path = data
            try:
                model_session = ort.InferenceSession(model_path)
                with open(tags_path, 'r', encoding='utf-8') as f:
                    next(f); model_tags = [row[1] for row in csv.reader(f)]
                result_queue.put(("LOAD_SUCCESS", f"Model '{os.path.basename(model_path)}' loaded."))
            except Exception as e:
                result_queue.put(("LOAD_FAIL", str(e)))
        elif command == "UNLOAD":
            model_session, model_tags = None, []; result_queue.put(("UNLOAD_SUCCESS", "Model unloaded."))
        elif command == "INFER":
            if not model_session: result_queue.put([]); continue
            try:
                image = Image.open(data).convert("RGB")
                input_size = model_session.get_inputs()[0].shape[1]
                image = image.resize((input_size, input_size))
                image_np = np.expand_dims(np.array(image, dtype=np.float32), axis=0)
                input_name, output_name = model_session.get_inputs()[0].name, model_session.get_outputs()[0].name
                probs = model_session.run([output_name], {input_name: image_np})[0][0]
                result_tags = {model_tags[i]: prob for i, prob in enumerate(probs) if prob > AI_THRESHOLD}
                tags = [tag for tag, prob in sorted(result_tags.items(), key=lambda i: i[1], reverse=True)]
                result_queue.put(tags)
            except Exception as e:
                print(f"AI Worker Inference Error: {e}"); result_queue.put([])
        elif command == "STOP":
            print("AI Worker Process Shutting Down."); break

# --- Helper Functions (No changes) ---
def natural_sort_key(s): return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', os.path.basename(s))]
def load_tags_from_file(image_path):
    if not image_path: return []
    tag_file = os.path.splitext(image_path)[0] + '.txt'
    if os.path.exists(tag_file):
        with open(tag_file, 'r', encoding='utf-8') as f: return [t.strip() for t in f.read().split(',') if t.strip()]
    return []
def save_tags_to_file(image_path, tags):
    if not image_path: return
    tag_file = os.path.splitext(image_path)[0] + '.txt'
    with open(tag_file, 'w', encoding='utf-8') as f: f.write(', '.join(sorted(tags if tags else [])))
def get_image_files(folder_path):
    if not folder_path or not os.path.isdir(folder_path): return []
    return sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(SUPPORTED_EXTENSIONS)], key=natural_sort_key)
def get_folder_suggestions(folder_path):
    if not folder_path or not os.path.isdir(folder_path): return []
    all_tags = []
    for filename in [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]:
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            all_tags.extend([t.strip() for t in f.read().split(',') if t.strip()])
    return sorted(list(set([t for t, c in Counter(all_tags).most_common(1000)])))

# --- Gradio Application Controller ---
class GradioAppController:
    def __init__(self, task_queue, result_queue):
        self.task_queue = task_queue
        self.result_queue = result_queue

    def _run_ai_tagging(self, image_path, state):
        if not image_path or not state.model_loaded: return []
        self.task_queue.put(("INFER", image_path))
        try: return self.result_queue.get(timeout=20)
        except queue.Empty: print("AI worker timed out."); return []

    def _refresh_all_lists(self, state, filter_text):
        current_tags = state.current_tags or []
        filter_text = filter_text.lower() if filter_text else ""
        filtered_current = [t for t in current_tags if filter_text in t.lower()]
        current_tags_update = gr.update(choices=filtered_current, value=filtered_current)
        ai_suggestions = state.current_ai_suggestions or []
        folder_choices = [t for t in state.folder_suggestions if t not in current_tags and filter_text in t.lower()]
        ai_choices = [t for t in ai_suggestions if t not in current_tags and filter_text in t.lower()]
        folder_sugg_update = gr.update(choices=folder_choices, value=[])
        ai_sugg_update = gr.update(choices=ai_choices, value=[])
        return state, current_tags_update, folder_sugg_update, ai_sugg_update

    def _navigate_to_index(self, state, target_index, filter_text):
        """Core logic to switch the viewer to a specific image index."""
        if state.current_index != -1:
            save_tags_to_file(state.current_image_path, state.current_tags)
        state.current_index = target_index
        state.current_image_path = state.image_files[state.current_index]
        state.current_tags = load_tags_from_file(state.current_image_path)
        state.current_ai_suggestions = self._run_ai_tagging(state.current_image_path, state)
        counter_text = f"{state.current_index + 1} / {len(state.image_files)}"
        image = Image.open(state.current_image_path)
        state, c, f, a = self._refresh_all_lists(state, filter_text)
        return state, image, counter_text, c, f, a

    def update_viewer(self, state, direction, filter_text):
        if not state.image_files:
            state.current_ai_suggestions = []
            state, c, f, a = self._refresh_all_lists(state, filter_text)
            return state, None, "0 / 0", c, f, a
        new_index = (state.current_index + direction) % len(state.image_files)
        return self._navigate_to_index(state, new_index, filter_text)

    def jump_to_image(self, state, image_number_str, filter_text):
        """Navigates to an image number provided by the user."""
        if not state.image_files:
            return state, None, "0 / 0", *self._refresh_all_lists(state, filter_text)[1:]

        try:
            target_number = int(str(image_number_str).split('/')[0].strip())
        except (ValueError, IndexError):
            counter_text = f"{state.current_index + 1} / {len(state.image_files)}"
            return state, gr.update(), counter_text, gr.update(), gr.update(), gr.update()

        if 1 <= target_number <= len(state.image_files):
            target_index = target_number - 1
            return self._navigate_to_index(state, target_index, filter_text)
        else:
            counter_text = f"{state.current_index + 1} / {len(state.image_files)}"
            return state, gr.update(), counter_text, gr.update(), gr.update(), gr.update()

    def filter_text_changed(self, state, filter_text):
        _, c, f, a = self._refresh_all_lists(state, filter_text)
        return c, f, a

    def add_tag_from_submit(self, new_tag, state, filter_text):
        if new_tag and new_tag not in state.current_tags:
            state.current_tags.append(new_tag); state.current_tags.sort()
        state, c, f, a = self._refresh_all_lists(state, filter_text)
        return state, c, f, a

    def add_tag_from_selection(self, evt: gr.SelectData, state, filter_text):
        tag_to_add = evt.value
        if tag_to_add and tag_to_add not in state.current_tags:
            state.current_tags.append(tag_to_add)
            state.current_tags.sort()
        state, c, f, a = self._refresh_all_lists(state, filter_text)
        return state, c, f, a

    def handle_current_tag_selection(self, evt: gr.SelectData, state, filter_text):
        tag_toggled = evt.value
        is_selected = evt.selected
        if not is_selected:
            if tag_toggled in state.current_tags:
                state.current_tags.remove(tag_toggled)
        state, c, f, a = self._refresh_all_lists(state, filter_text)
        return state, c, f, a

    # --- MODIFIED: This function now correctly returns all 7 expected output values ---
    def handle_folder_selection(self, folder_path, state):
        if not folder_path or not os.path.isdir(folder_path):
            state, c, f, a = self._refresh_all_lists(state, "")
            # This branch returns the correct 7 values
            return state, "Invalid folder path.", None, "0 / 0", c, f, a

        state.image_files = get_image_files(folder_path)
        state.folder_suggestions = get_folder_suggestions(folder_path)

        if not state.image_files:
            state.current_index, state.current_image_path, state.current_tags = -1, None, []
            state.current_ai_suggestions = []
            state, c, f, a = self._refresh_all_lists(state, "")
            # This branch also returns the correct 7 values
            return state, "No images found.", None, "0/0", c, f, a
        
        state.current_index = -1
        
        # --- FIX STARTS HERE ---
        # Call the navigation function to get the 6 navigation-related values
        state, image, counter, c, f, a = self.update_viewer(state, 1, "")
        # Now, return all 7 values, correctly inserting the status message
        return state, "Folder loaded.", image, counter, c, f, a
        # --- FIX ENDS HERE ---

    def load_model_and_refresh(self, model_folder_path, state, filter_text):
        status_message, load_btn_update, unload_btn_update = "", None, None
        try:
            if not model_folder_path or not os.path.isdir(model_folder_path): raise FileNotFoundError(f"Directory not found: {model_folder_path}")
            onnx_files = [f for f in os.listdir(model_folder_path) if f.lower().endswith(".onnx")]
            if not onnx_files: raise FileNotFoundError(f"No .onnx file in: {model_folder_path}")
            model_path = os.path.join(model_folder_path, onnx_files[0])
            csv_path = os.path.splitext(model_path)[0] + '.csv'
            if not os.path.isfile(csv_path): raise FileNotFoundError(f"Tag file not found: {csv_path}")
            self.task_queue.put(("LOAD", (model_path, csv_path)))
            status_code, message = self.result_queue.get(timeout=30)
            if status_code == "LOAD_SUCCESS":
                state.model_loaded = True; status_message = message
                load_btn_update = gr.update(interactive=False)
                unload_btn_update = gr.update(interactive=True)
                if state.current_image_path:
                    state.current_ai_suggestions = self._run_ai_tagging(state.current_image_path, state)
            else:
                raise RuntimeError(f"Worker failed to load model: {message}")
        except Exception as e:
            state.model_loaded = False; status_message = f"Error: {e}"
            load_btn_update = gr.update(interactive=True)
            unload_btn_update = gr.update(interactive=False)
        state, c, f, a = self._refresh_all_lists(state, filter_text)
        return state, status_message, load_btn_update, unload_btn_update, c, f, a

    def unload_model(self, state):
        self.task_queue.put(("UNLOAD", None))
        _, message = self.result_queue.get(timeout=10)
        state.model_loaded = False
        state.current_ai_suggestions = []
        ai_sugg_update = gr.update(choices=[], value=[])
        return state, message, gr.update(interactive=True), gr.update(interactive=False), ai_sugg_update

    def update_image_size(self, width, height):
        return gr.update(width=int(width), height=int(height))

# --- Gradio UI Definition (No changes from previous version) ---
js_hotkeys = """
() => {
    function handleKeyPress(event) {
        const isInputElement = event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA';
        if (event.altKey && !isInputElement) {
            let prevButton = document.getElementById('prev_button');
            let nextButton = document.getElementById('next_button');
            if (event.key === 'ArrowLeft' && prevButton) {
                prevButton.click();
                event.preventDefault();
            } else if (event.key === 'ArrowRight' && nextButton) {
                nextButton.click();
                event.preventDefault();
            }
        }
    }
    document.addEventListener('keydown', handleKeyPress);
}
"""

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    task_queue, result_queue = mp.Queue(), mp.Queue()
    worker_process = mp.Process(target=inference_worker, args=(task_queue, result_queue), daemon=True)
    worker_process.start()
    controller = GradioAppController(task_queue, result_queue)
    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue"), title="Image Tagger", js=js_hotkeys) as demo:
        app_state = gr.State(AppState())
        with gr.Row():
            with gr.Column(scale=1, min_width=425):
                gr.Markdown("### Current Image Tags\n_Uncheck a tag to remove it._")
                current_tags_display = gr.CheckboxGroup(label="Current Tags", interactive=True)
            with gr.Column(scale=3):
                image_display = gr.Image(label="Image Viewer", type="pil", width=600, height=600)
                with gr.Row():
                    width_input = gr.Number(value=600, label="Width", step=50)
                    height_input = gr.Number(value=600, label="Height", step=50)
                status_display = gr.Textbox(label="Status", interactive=False, value="Welcome!")
                image_counter = gr.Textbox(label="Image Position (Enter to Jump)", interactive=True, value="0 / 0")
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous (Alt+Left)", elem_id="prev_button")
                    next_btn = gr.Button("Next ➡️ (Alt+Right)", elem_id="next_button")
                with gr.Accordion("Folder & Model Selection", open=True):
                    folder_input = gr.Textbox(label="Image Folder Path", placeholder="C:\\path\\to\\your\\images")
                    load_folder_btn = gr.Button("Load Folder")
                    with gr.Row():
                        model_path_input = gr.Textbox(label="ONNX Model Folder Path", placeholder="C:\\path\\to\\your\\model_folder", scale=3)
                        load_model_btn = gr.Button("Load Model", scale=1)
                        unload_model_btn = gr.Button("Unload Model", interactive=False, scale=1)
            with gr.Column(scale=1, min_width=425):
                tag_filter_input = gr.Textbox(label="Filter / Add New Tag", placeholder="Type to filter or press Enter to add...")
                with gr.Tabs():
                    with gr.TabItem("Folder Suggestions"):
                        folder_suggestions_display = gr.CheckboxGroup(label="Click to add", interactive=True)
                    with gr.TabItem("AI Suggestions"):
                        ai_suggestions_display = gr.CheckboxGroup(label="Click to add", interactive=True)

        all_list_outputs = [current_tags_display, folder_suggestions_display, ai_suggestions_display]
        state_and_lists = [app_state] + all_list_outputs
        viewer_outputs = [app_state, image_display, image_counter] + all_list_outputs

        width_input.submit(fn=controller.update_image_size, inputs=[width_input, height_input], outputs=[image_display])
        height_input.submit(fn=controller.update_image_size, inputs=[width_input, height_input], outputs=[image_display])
        load_folder_btn.click(fn=controller.handle_folder_selection, inputs=[folder_input, app_state], outputs=[app_state, status_display, image_display, image_counter] + all_list_outputs)
        load_model_btn.click(fn=controller.load_model_and_refresh, inputs=[model_path_input, app_state, tag_filter_input], outputs=[app_state, status_display, load_model_btn, unload_model_btn] + all_list_outputs)
        unload_model_btn.click(fn=controller.unload_model, inputs=[app_state], outputs=[app_state, status_display, load_model_btn, unload_model_btn, ai_suggestions_display])
        prev_btn.click(fn=controller.update_viewer, inputs=[app_state, gr.State(-1), tag_filter_input], outputs=viewer_outputs)
        next_btn.click(fn=controller.update_viewer, inputs=[app_state, gr.State(1), tag_filter_input], outputs=viewer_outputs)
        image_counter.submit(fn=controller.jump_to_image, inputs=[app_state, image_counter, tag_filter_input], outputs=viewer_outputs)
        tag_filter_input.change(fn=controller.filter_text_changed, inputs=[app_state, tag_filter_input], outputs=all_list_outputs)
        tag_filter_input.submit(fn=controller.add_tag_from_submit, inputs=[tag_filter_input, app_state, tag_filter_input], outputs=state_and_lists).then(lambda: gr.update(value=""), outputs=[tag_filter_input])
        current_tags_display.select(fn=controller.handle_current_tag_selection, inputs=[app_state, tag_filter_input], outputs=state_and_lists, show_progress="hidden")
        folder_suggestions_display.select(fn=controller.add_tag_from_selection, inputs=[app_state, tag_filter_input], outputs=state_and_lists)
        ai_suggestions_display.select(fn=controller.add_tag_from_selection, inputs=[app_state, tag_filter_input], outputs=state_and_lists)

    demo.launch(share=False, server_port=7877)
    print("Gradio App closed. Stopping worker process...")
    task_queue.put(("STOP", None))
    worker_process.join(timeout=5)