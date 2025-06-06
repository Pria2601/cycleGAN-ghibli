# 🎨 CycleGAN for Ghibli-Style Transfer

This repository contains an implementation of a CycleGAN model that performs unpaired image-to-image translation, specifically transforming real-world photos into Ghibli-style artwork. This project was developed and trained on a Kaggle environment using TPU acceleration.

---

## 📌 About the Project

As part of a deep learning exploration, this notebook implements a CycleGAN model to translate photo images into Studio Ghibli-style frames. The CycleGAN framework is ideal for tasks where paired datasets aren't available.

We trained this model using:
- TPU strategy (if available)
- Custom data loading pipeline using `tf.data`
- PatchGAN Discriminator and ResNet-based Generator
- Losses: adversarial, cycle-consistency, identity

---

## 👩‍💻 Contributors

- **Priya Gawshinde**
- **Sakshi Gupta**

This was a collaborative group project combining our interest in deep learning with creative image translation.

---

## 📂 Files

- `cyclegan.ipynb`: Main notebook containing the end-to-end pipeline for model training, visualization, and testing.

---

## 🧠 Model Architecture

### Generator (Photo → Ghibli and vice versa):
- ResNet-based with residual blocks
- Uses instance normalization and ReLU

### Discriminator:
- PatchGAN structure
- Classifies image patches instead of full images

### Losses:
- **Adversarial Loss**
- **Cycle Consistency Loss**
- **Identity Loss** (optional but used for stability)

---

## 🏃 Training Details

| Parameter         | Value              |
|------------------|--------------------|
| Epochs           | 10                 |
| Steps per Epoch  | 500                |
| Batch Size       | 1 (default CycleGAN) |
| Strategy         | TPU (if available) |

Intermediate outputs (losses and images) are printed every 100 steps to monitor training progress.

---

## 🖼 Dataset

- The model was trained on:
  - Unpaired **real photos**
  - Stylized **Ghibli-style images**

Kaggle datasets were used, with `KaggleDatasets().get_gcs_path()` for TPU compatibility.

---

## 📈 Sample Outputs

During training, the notebook displays:
- Ghibli-style generation results from input photos
- Comparison of real vs generated images
- Epoch-wise loss metrics

---

## ⚙️ Setup

### Dependencies

Make sure the following packages are installed:

bash
```
pip install tensorflow matplotlib numpy tensorflow-addons
```

### Running

You can run the notebook in:
- **Kaggle Kernels** (TPU recommended)
- **Google Colab** (with minor path changes)
- **Local Jupyter Notebook** (GPU preferred)

---

## 🚧 Future Improvements

- Add model checkpoint saving/loading
- Export trained models for inference
- Add GUI or web interface for real-time translation
- Fine-tune with larger datasets or specific styles

---

## 🙋‍♀️ Personal Motivation

This project was a creative application of GANs undertaken by us to explore the power of unpaired image translation. Inspired by the dreamy visuals of Studio Ghibli, we aimed to:

- Understand adversarial training mechanics deeply
- Implement CycleGANs from scratch
- Use TPUs efficiently in a Kaggle environment
- Produce aesthetic, stylized outputs through deep learning

---

## 📚 References

- [CycleGAN Paper (Zhu et al., 2017)](https://arxiv.org/abs/1703.10593)
- [TensorFlow CycleGAN Tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan)
- Kaggle TPU Notebooks
