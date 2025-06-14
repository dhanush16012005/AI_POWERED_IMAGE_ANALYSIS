import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import matplotlib.pyplot as plt

# Model setup
device = torch.device('cpu')  # Use 'cuda' if GPU is available
dtype = torch.float32
model_name_or_path = 'GoodBaiBai88/M3D-LaMed-Phi-3-4B'
proj_out_num = 256

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float32,
    device_map='cpu',
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)

# Chat history storage
chat_history = []
current_image = None

def extract_and_display_images(image_path):
    npy_data = np.load(image_path)
    if npy_data.ndim == 4 and npy_data.shape[1] == 32:
        npy_data = npy_data[0]
    elif npy_data.ndim != 3 or npy_data.shape[0] != 32:
        return "Invalid .npy file format. Expected shape (1, 32, 256, 256) or (32, 256, 256)."
    
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(npy_data[i], cmap='gray')
        ax.axis('off')
    
    image_output = "extracted_images.png"
    plt.savefig(image_output, bbox_inches='tight')
    plt.close()
    return image_output


def process_image(question):
    global current_image
    if current_image is None:
        return "Please upload an image first."
    
    image_np = np.load(current_image)
    image_tokens = "<im_patch>" * proj_out_num
    input_txt = image_tokens + question
    input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
    
    image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
    generation = model.generate(image_pt, input_id, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
    generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
    return generated_texts[0]


def chat_interface(question):
    global chat_history
    response = process_image(question)
    chat_history.append((question, response))
    return chat_history


def upload_image(image):
    global current_image
    current_image = image.name
    extracted_image_path = extract_and_display_images(current_image)
    return "Image uploaded and processed successfully!", extracted_image_path

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as chat_ui:
    gr.Markdown("ICliniq AI-Powered Medical Image Analysis Workspace")
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            chat_list = gr.Chatbot(value=[], label="Chat History", elem_id="chat-history")
        with gr.Column(scale=4):
            uploaded_image = gr.File(label="Upload .npy Image", type="filepath")
            upload_status = gr.Textbox(label="Status", interactive=False)
            extracted_image = gr.Image(label="Extracted Images")
            question_input = gr.Textbox(label="Ask a question", placeholder="Ask something about the image...")
            submit_button = gr.Button("Send")
    
    uploaded_image.upload(upload_image, uploaded_image, [upload_status, extracted_image])
    submit_button.click(chat_interface, question_input, chat_list)
    question_input.submit(chat_interface, question_input, chat_list)

chat_ui.launch()