# Projected Gradient Descent (PGD) Adversarial Attack 

## 1. Introduction

This project implements a robust and perceptually-aware **Projected Gradient Descent (PGD)** adversarial attack pipeline for evaluating the resilience of image classification models. Adversarial attacks are small, often imperceptible perturbations deliberately added to input images to induce misclassification by neural networks. PGD is one of the most widely-used iterative adversarial attack methods due to its effectiveness in exploring the worst-case perturbation space around an input image.

The goal of this implementation is not only to test model robustness but also to generate adversarial examples that are visually coherent and realistic, particularly for high-resolution or multi-spectral images.

---

## 2. PGD Attack Overview

### 2.1 Concept

**Projected Gradient Descent (PGD)** is an iterative adversarial attack that generalizes the Fast Gradient Sign Method (FGSM). At each iteration, the attack updates the input image in the direction that maximally increases (or decreases, for targeted attacks) the model's loss, while ensuring that the cumulative perturbation remains within a defined **ε-ball** around the original image.

Mathematically, the iterative update is:

[
x_{t+1} = \text{Proj}_\epsilon \big(x_t + \alpha \cdot \text{step_direction}\big)
]

Where:

* `step_direction` is derived from the gradient of the loss with respect to the input.
* `Proj_ε` projects the perturbed image back into the allowed perturbation region.
* `α` is the step size controlling perturbation increment per iteration.

By iterating multiple times, PGD explores adversarial directions more thoroughly than single-step methods, yielding stronger and more transferable attacks.

---

## 3. Project Workflow and Functionalities

This implementation enhances classical PGD with **perceptual and stability improvements** that increase both attack effectiveness and visual realism.

### 3.1 Iterative Perturbation Application

* Input images are iteratively modified using gradient information from the model.
* Each iteration applies a small step towards increasing the model's loss.
* The perturbation is constrained within a predefined maximum magnitude (`eps`) to keep changes subtle.

### 3.2 Gradient-based Importance Masking

* Perturbations are applied preferentially to the most influential pixels, identified by the gradient magnitude.
* A **soft mask** ensures that important regions are targeted while preserving spatial coherence.
* This reduces unnecessary noise and focuses the attack where it is most likely to succeed.

### 3.3 Gaussian Smoothing

* Perturbations are smoothed to reduce high-frequency noise that can be visually conspicuous.
* Applied both in the iterative loop and during perceptual reconstruction of saved images.

### 3.4 Random Dithering

* Small stochastic noise is added to break up structured patterns in the perturbation.
* Helps avoid easy detection and increases attack variability.

### 3.5 LAB Color-space Perceptual Scaling

* Perturbations are optionally applied in LAB color space, allowing separate scaling of lightness and chromaticity channels.
* This preserves visual realism, especially when saving adversarial images in formats like TIFF or high dynamic range imagery.

---

## 4. Batch Evaluation and Image Generation

The pipeline is designed for batch processing of datasets:

1. **Clean Evaluation**

   * Computes model accuracy and loss on unaltered images.

2. **Adversarial Generation**

   * PGD is applied iteratively on image batches.
   * Importance masks, smoothing, and dithering are applied automatically per batch.
   * Perturbations are clipped to remain within the allowed `eps` bounds.

3. **Adversarial Evaluation**

   * Computes model accuracy and loss on adversarially perturbed images.
   * Metrics are aggregated across the dataset for benchmarking.

4. **Image Saving**

   * Adversarial images are optionally saved for inspection.
   * Perceptual adjustments in LAB space ensure the adversarial examples are realistic.
   * Perturbation strength is scaled according to a perceptual factor to avoid visible artifacts.

---

## 5. Key Functional Enhancements

| Functionality           | Purpose                                                      |
| ----------------------- | ------------------------------------------------------------ |
| Iterative perturbation  | Strengthens attack by exploring worst-case directions        |
| Gradient-based masking  | Focuses perturbation on most influential pixels              |
| Gaussian smoothing      | Ensures spatially coherent and visually smooth perturbations |
| Random dithering        | Reduces structured noise patterns and increases stealth      |
| LAB color-space scaling | Preserves perceptual realism and color balance               |
| Controlled clipping     | Keeps perturbations within the defined `ε` bounds            |

---

## 6. Output Metrics

The evaluation routine provides a structured metrics summary:

* `clean_acc` — model accuracy on unaltered images.
* `adv_acc` — model accuracy on adversarially perturbed images.
* `clean_loss` — mean loss on clean images.
* `adv_loss` — mean loss on adversarial images.
* `eps` — maximum perturbation magnitude used.
* `saved` — number of adversarial images saved.
* `out_dir` — directory path for saved images.

These metrics provide insight into the model's vulnerability and the effectiveness of the attack.
