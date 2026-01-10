import gradio as gr
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

# --- 1. DEFINE MODEL ARCHITECTURE ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.LATENT_DIM = 100
        self.main = nn.Sequential(
            # Input: 100 -> 7x7x512
            nn.ConvTranspose2d(self.LATENT_DIM, 512, 7, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 7x7 -> 14x14
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x.view(-1, self.LATENT_DIM, 1, 1))

# --- 2. LOAD MODEL ---
device = torch.device('cpu') 
model = Generator()

try:
    # Load weights
    model.load_state_dict(torch.load("synthetic_generator.pth", map_location=device))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Did you upload 'synthetic_generator.pth'?")

# --- 3. GENERATION FUNCTION ---
def generate_fashion(seed, num_images):
    # Set seed for reproducibility
    torch.manual_seed(int(seed))
    
    # Generate batch of images (Always 9 for a 3x3 grid)
    noise = torch.randn(9, 100).to(device)
    
    with torch.no_grad():
        fake_imgs = model(noise)
    
    # Process images for display
    fake_imgs = fake_imgs.cpu().numpy()
    fake_imgs = (fake_imgs + 1) / 2 # Normalize to [0, 1]
    
    # Create a Matplotlib Figure
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    for i, ax in enumerate(axs.flatten()):
        img = fake_imgs[i].squeeze()
        # 'gray_r' = White background, black item (looks like a sketch)
        ax.imshow(img, cmap='gray_r') 
        ax.axis('off')
        
    plt.suptitle(f"AI Generated Collection (Seed: {seed})", fontsize=15)
    return fig

# --- 4. GRADIO INTERFACE ---
theme = gr.themes.Soft()

with gr.Blocks(theme=theme, title="Synthetic Data Generator") as app:
    gr.Markdown("# 🧥 Synthetic Data Generator (DCGAN)")
    gr.Markdown("This AI generates **new, unique fashion items** (trousers, bags, shirts) from random noise. It was trained on the Fashion-MNIST dataset using a Deep Convolutional GAN.")
    
    with gr.Row():
        with gr.Column(scale=1):
            seed_slider = gr.Slider(0, 1000, value=42, step=1, label="Random Seed (Change for new designs)")
            btn = gr.Button("✨ Generate New Collection", variant="primary", size="lg")
            
            gr.Markdown("### ℹ️ Technical Details")
            gr.Markdown("""
            * **Architecture:** DCGAN (Deep Convolutional Generative Adversarial Network)
            * **Framework:** PyTorch
            * **Input:** Random Gaussian Noise (Vector size 100)
            * **Output:** 28x28 Grayscale Image
            """)
            
        with gr.Column(scale=2):
            output_plot = gr.Plot(label="Generated Output")

    btn.click(fn=generate_fashion, inputs=[seed_slider, seed_slider], outputs=output_plot)

if __name__ == "__main__":
    app.launch()
