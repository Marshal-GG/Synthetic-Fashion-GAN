# 🧥 Synthetic Fashion Generator (DCGAN)

![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)
![GAN](https://img.shields.io/badge/Architecture-DCGAN-blue)

A Generative AI project that creates unique, synthetic fashion images (T-shirts, sneakers, bags) from random noise. Built using a **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on the Fashion-MNIST dataset.

## 🚀 Live Demo
**[👉 Try the Live App on Hugging Face](https://huggingface.co/RupamG/Synthetic-Fashion-GAN)**

---

## 🧠 How it Works
This project uses two competing neural networks:
1.  **The Generator (The Forger):** Takes random noise (latent vector) and tries to generate a realistic image of clothing.
2.  **The Discriminator (The Police):** Looks at images and tries to tell if they are real (from the dataset) or fake (from the generator).

Over time, the Generator gets so good that the Discriminator can no longer tell the difference.

---

## 🛠️ Project Structure
* `synthetic_data_generator.ipynb`: The training loop for the GAN.
* `app.py`: The deployment script using Gradio.
* `synthetic_generator.pth`: The saved weights of the trained Generator.

## 📊 Results
The model learns to generate various categories of fashion items, including:
* T-shirts / Tops
* Trousers
* Sneakers
* Bags
* Ankle boots

## 💻 Tech Stack
* **Core:** Python, PyTorch
* **Visualization:** Matplotlib
* **Deployment:** Gradio, Hugging Face Spaces

### 🔧 Local Installation (GPU Support)
If you have an NVIDIA GPU, install PyTorch with CUDA support for faster training:
```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements.txt

---
*Created by Rupam*