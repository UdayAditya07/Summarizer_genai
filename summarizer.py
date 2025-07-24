
from transformers import pipeline
import gradio as gr

# Use better summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, min_length, max_length, progress=gr.Progress()):
    if not text or not text.strip():
        return "Please enter some text to summarize."
    try:
        progress(0.2)
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        progress(1)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("""
    # 🧠 BART-Large Text Summarizer
    Summarizes long text using Facebook's BART-Large CNN model.<br>
    **Model:** facebook/bart-large-cnn
    """)
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(lines=10, placeholder="Paste your text here...")
        with gr.Column():
            min_length = gr.Slider(20, 100, value=40, label="Minimum Summary Length")
            max_length = gr.Slider(100, 300, value=200, label="Maximum Summary Length")
    output = gr.Textbox(label="Summary")
    summarize_btn = gr.Button("Summarize")

    summarize_btn.click(
        summarize_text,
        inputs=[text_input, min_length, max_length],
        outputs=output,
        api_name="summarize"
    )

demo.launch()
