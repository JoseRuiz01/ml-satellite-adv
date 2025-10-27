# Projected Gradient Descent (PGD) Adversarial Attack

## 1. Introduction

This document explains the implementation of an **improved Projected Gradient Descent (PGD)** adversarial attack for evaluating the robustness of image classification models, particularly those trained on the **EuroSAT** dataset.

Adversarial attacks are small, human-imperceptible perturbations intentionally added to input images to deceive neural networks into making incorrect predictions. The **PGD attack** is one of the most effective iterative methods for generating such perturbations.

---

## 2. PGD Attack Overview

### 2.1 Concept

The **Projected Gradient Descent (PGD)** attack extends the basic **Fast Gradient Sign Method (FGSM)** by applying it iteratively. At each step, it adjusts the input image in the direction that maximally increases (or decreases, for targeted attacks) the model’s loss, while ensuring the total perturbation remains within a defined constraint (the *ε-ball*).

Formally, given:

* Input image ( x )
* True label ( y )
* Model ( f_\theta )
* Loss function ( L )
* Perturbation bound ( \epsilon )

The iterative update rule is:

[
x_{t+1} = \text{Proj}*{\epsilon}(x_t + \alpha \cdot \text{sign}(\nabla_x L(f*\theta(x_t), y)))
]

where:

* ( \alpha ) is the step size,
* ( \text{Proj}_{\epsilon}(\cdot) ) projects back into the allowed perturbation region.

This method effectively explores adversarial directions within a bounded region of the input space, yielding stronger attacks than single-step methods.

---

## 3. Implementation Details

### 3.1 Core Function: `pgd_attack_batch`

The attack is implemented in the function **`pgd_attack_batch()`**, which applies iterative, masked, and perceptually smoothed perturbations on image batches.

#### Key Parameters:

* `eps`: Maximum allowed perturbation (attack strength).
* `alpha`: Step size per iteration (default fraction of `eps`).
* `iters`: Number of attack iterations.
* `targeted`: If `True`, directs the attack toward a specific target class.
* `grad_mask_fraction`: Fraction of most important pixels to perturb.
* `smooth_perturb_sigma`: Gaussian smoothing applied to perturbations.
* `dither_scale`: Random dithering applied to reduce visible artifacts.

#### Attack Enhancements:

Unlike the classical PGD attack, this version introduces **several perceptual and stability improvements**:

1. **Gradient-based importance masking**

   * Only perturbs spatial regions most relevant to the model’s decision.
   * Computed using the gradient magnitude per pixel.
2. **Gaussian smoothing**

   * Reduces high-frequency artifacts that make adversarial noise visually detectable.
3. **Random dithering**

   * Adds subtle stochastic noise to break up structured patterns.
4. **LAB color-space scaling**

   * Perturbations are adjusted in perceptually uniform color space for realism.

---

## 4. Importance Mask Generation

### Function: `_make_importance_mask()`

This helper function computes a soft attention mask based on gradient magnitudes:

1. The absolute gradient is averaged over channels to get an importance map.
2. The top `keep_fraction` of pixels (most influential) are retained.
3. The mask is then **Gaussian-blurred** to ensure spatial smoothness.
4. The mask is normalized between `[0, 1]` and applied to subsequent gradient updates.

This ensures perturbations are **spatially coherent** and localized to the most sensitive areas of the image.

---

## 5. Evaluation and Image Saving

### Function: `evaluate_pgd()`

This higher-level function applies the PGD attack across a dataset and saves the adversarial images for inspection.

#### Workflow:

1. **Evaluate Clean Accuracy**

   * Compute model accuracy on unaltered images.
2. **Generate Adversarial Samples**

   * Apply `pgd_attack_batch()` to create adversarial versions of each image.
3. **Evaluate Adversarial Accuracy**

   * Test the model’s predictions on adversarial images.
4. **Save Adversarial Images**

   * Convert normalized tensors back to raw `.tif` format.
   * Apply **LAB color smoothing** and **perceptual scaling** to minimize visible distortions.

#### Output:

The function returns key performance metrics:

| Metric       | Description                                   |
| ------------ | --------------------------------------------- |
| `clean_acc`  | Accuracy on original clean images             |
| `adv_acc`    | Accuracy on adversarially perturbed images    |
| `clean_loss` | Mean loss on clean images                     |
| `adv_loss`   | Mean loss on adversarial images               |
| `eps`        | Perturbation magnitude used                   |
| `saved`      | Number of adversarial samples saved           |
| `out_dir`    | Output directory path for saved `.tif` images |

---

## 6. Visualization Improvements

To ensure the generated adversarial examples remain **visually realistic**, several perceptual enhancements are applied:

1. **LAB Color Transformation**
   Converts images to LAB color space where L (lightness) and a,b (chromaticity) channels are adjusted separately, preserving color balance.

2. **Perceptual Scaling Factor (`perceptual_eps_factor`)**
   Reduces the perturbation strength in raw pixel space to avoid visible brightness or color shifts.

3. **Spatial Smoothing**
   Gaussian smoothing is applied to the adversarial deltas in LAB space, further blending noise into natural textures.

---

## 7. Summary of Key Features

| Enhancement            | Purpose                                      |
| ---------------------- | -------------------------------------------- |
| Gradient-based masking | Focus attack on important pixels             |
| Gaussian smoothing     | Remove high-frequency noise artifacts        |
| Dithering              | Break pattern repetition and enhance stealth |
| LAB color adjustment   | Preserve perceptual realism                  |
| Controlled clipping    | Keep perturbations within `ε` bounds         |

---

## 8. Conclusion

The implemented **PGD attack pipeline** provides a robust and perceptually-aware framework for evaluating the adversarial resilience of satellite image classifiers.
By combining gradient-based targeting, spatial smoothing, and perceptual adjustments, this approach generates realistic adversarial examples that effectively test model robustness without visibly degrading image quality.

This attack will be used to assess the vulnerability of the trained **ResNet18** and **SimpleCNN** models in the EuroSAT classification experiment.
