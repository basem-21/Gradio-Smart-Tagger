# Gradio Smart Image Tagger

An interactive Gradio UI for efficiently tagging image datasets. This tool accelerates your workflow by providing smart tag suggestions from a pre-trained ONNX model.

![Screenshot of the Tagger UI](https://i.imgur.com/nJ2uV3S.png)
*(This is an example image. I'll show you how to add your own screenshot at the end!)*

## Features

-   **Responsive UI**: A clean, multi-column layout that's easy to navigate.
-   **AI-Powered Suggestions**: Load any compatible ONNX tagging model to get instant tag suggestions, saving you from typing common tags.
-   **Folder-Wide Suggestions**: See a list of the most common tags already used in the current folder.
-   **Efficient Tag Management**: Add tags by typing, clicking suggestions, and remove them just as easily.
-   **Hotkeys**: Navigate between images using `Alt + Left Arrow` and `Alt + Right Arrow` without taking your hands off the keyboard.
-   **Direct Navigation**: Jump directly to any image number by typing it in the position box.
-   **Fast & Stable**: The ONNX model runs in a separate process, ensuring the user interface remains smooth and responsive at all times.

## Installation

To get started, you will need Python 3.8 or newer.

1.  **Clone the repository:**
    Open your terminal or command prompt and run this command:
    ```bash
    git clone https://github.com/basem-21/Gradio-Smart-Tagger.git
    cd Gradio-Smart-Tagger
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *For significantly faster AI suggestions on an NVIDIA GPU, please see the notes inside the `requirements.txt` file.*

## How to Use

1.  **Run the application:**
    In your terminal, while inside the `Gradio-Smart-Tagger` folder, run the script:
    ```bash
    python app.py
    ```

2.  **Open the UI:**
    The terminal will provide a local URL, typically `http://127.0.0.1:7877`. Open this address in your web browser.

3.  **Start Tagging:**
    -   **Load Images**: Paste the full path to your image folder (e.g., `C:\Users\YourName\Pictures\MyDataset`) into the "Image Folder Path" box and click "Load Folder".
    -   **Load AI Model (Optional)**: Paste the path to the folder containing your `.onnx` and `.csv` files and click "Load Model" to enable AI suggestions.
    -   **Tag**: Use the "Filter / Add New Tag" box to add new tags, or click on tags from the suggestion lists.
    -   **Navigate**: Use the Next/Previous buttons or the hotkeys to move between images. Your tags are saved automatically to a `.txt` file when you navigate away from an image.
