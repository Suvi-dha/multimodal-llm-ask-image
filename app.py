import gradio as gr
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

def answer_question(image, question):
    inputs = processor(image, question, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)

iface = gr.Interface(
    fn=answer_question,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Ask a question")],
    outputs="text",
    title="Ask Me About This Image",
    description="Upload an image and ask any question. Powered by BLIP-2"
)

iface.launch()
