# 🌊 Underwater Image Restoration using GAN

This project focuses on restoring underwater images using a Generative Adversarial Network (GAN). It aims to enhance visual clarity and correct color distortions caused by the underwater environment, utilizing deep learning techniques on a large-scale dataset.

---

## 🎯 Objective

- 1. To Reduce haze and scattering effects that obscure details, leading to a clearer image. millets along with their recipes.
- 2. To Correct the dominant blue/green color cast caused by light scattering and restore a more accurate and balanced color representation of the underwater environment.
---

## 🧪 Methodology

### (a) 📥 Data Collection
- Utilized the **LSUI (Large Scale Underwater Image)** dataset.
- Includes a diverse set of underwater scenes for robust model training.

### (b) 🛠️ Data Pre-processing
- **Resizing**: All images are standardized to **256x256 pixels**.
- **Normalization**: Pixel values scaled between **0 and 1** for efficient model convergence.

### (c) 🤖 GAN Model Training
- Trained a **Generative Adversarial Network (GAN)** on the LSUI dataset.
- Model includes a **Generator** to restore images and a **Discriminator** to distinguish between real and generated images.

### (d) 🎨 Underwater Image Restoration
- The **Generator model** reconstructs high-quality images from degraded underwater samples.

### (e) 🧠 Discriminator Evaluation
- Evaluated results using:
  - **PSNR (Peak Signal-to-Noise Ratio)** – measures image quality.
  - **SSIM (Structural Similarity Index Measure)** – assesses structural similarity between original and restored images.

---

## 🧰 Tech Stack

- **Framework**: TensorFlow / PyTorch
- **Programming Language**: Python
- **Model Type**: Deep Convolutional GAN (DCGAN / Custom GAN)
- **Metrics**: PSNR, SSIM
- **Dataset**: LSUI (Large Scale Underwater Images)

---

### 📽️ Demo Video

[![Watch the demo]( https://drive.google.com/drive/folders/1vDQxhgS__UYpDrCnZim67fpTNs5V7YqE?usp=drive_link)


![Screenshot 2024-06-14 095530](https://github.com/user-attachments/assets/7a5ff04b-21b6-43bc-866c-2b5090b0421a)

![Screenshot 2024-06-19 144340](https://github.com/user-attachments/assets/4fb6affb-93da-45bd-9beb-81f18a6668c5)

