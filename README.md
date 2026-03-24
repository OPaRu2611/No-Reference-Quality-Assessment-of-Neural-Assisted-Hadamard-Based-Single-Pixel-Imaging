## Overview

This project provides a set of tools for **image processing, noise simulation, and denoising**, mainly based on the **Hadamard transform** and **Deep Image Prior (DIP)**.

It consists of two main files:

* `main.ipynb` → interactive notebook to run experiments
* `utility.py` → collection of reusable functions

---

## File Structure

### 1. `utility.py`

This file contains all the core functions used in the project, including:

#### Image Transformations

* `hadamard_zigzag(...)`
* `hadamard_zigzag_normalized(...)`
* `inverse_hadamard_zigzag(...)`

These functions convert images to and from the Hadamard domain.

---

#### Noise Simulation

* `add_poisson_noise(...)`
* `hadamard_poisson_experiment(...)`

Used to simulate realistic acquisition noise (Poisson noise).

---

#### Denoising

* `dip(...)`

Applies **Deep Image Prior**.

---

#### Metrics & Evaluation

* `ssim`, `psnr` (via skimage)
* `image_autocorrelation(...)`
* `cross_correlation_2d(...)`
* `dfa2d_vectorized(...)`
* `plot_dfa_with_fit(...)`

Used to evaluate image quality.

---

#### Visualization

* `plot_comparative_histogram(...)`
* `plot_colored_scatter(...)`

---

### 2. `main.ipynb`

This is the main notebook where you:

* Load images
* Run experiments
* Apply transformations
* Add noise
* Denoise images
* Visualize and analyze results

---

## Installation

Make sure you have Python 3.8+ and install dependencies:

```bash
pip install numpy pandas matplotlib opencv-python torch scikit-image
```

The project will automatically clone the required DIP repository:

```python
deep-image-prior
```

---

## Basic Usage

### 1. Import the utilities

In `main.ipynb`:

```python
from utility import *
or 
import utility
```

---

### 2. Load an image

```python
import numpy as np
import cv2

img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
img = img.astype(float)
```

---

### 3. Apply Hadamard Transform

```python
coeffs = hadamard_zigzag_normalized(img)
```

---

### 4. Add Noise (Poisson simulation)

```python
result = hadamard_poisson_experiment(img, lvl=5.0)

noisy_img = result["corrupted"]
```

---

### 5. Denoise using DIP

```python
denoised = dip(noisy_img, device="cpu")
```

---

### 6. Evaluate Results

```python
from skimage.metrics import structural_similarity as ssim

score = ssim(img, denoised, data_range=img.max() - img.min())
print("SSIM:", score)
```

---

### 7. Run Full Experiment

```python
noise_levels = [1, 2, 5, 10]

df = numerical_evaluation(
    img=img,
    noise_lvls=noise_levels,
    device="cpu",
    referenced=True,
    no_referenced=True,
    show=True
)
```

---

## Typical Workflow

1. Load image
2. Transform to Hadamard domain
3. Add noise
4. Reconstruct image
5. Apply denoising (DIP)
6. Evaluate quality

---

## Notes

* Images must be:

  * Square (N × N)
  * Size must be a **power of 2** (e.g., 64×64, 128×128)
* DIP training may take time
* GPU is recommended for faster performance (`device="cuda"`)

---

## Example Pipeline

```python
img = cv2.imread("image.png", 0).astype(float)

# Simulate noisy acquisition
exp = hadamard_poisson_experiment(img, lvl=3.0)
noisy = exp["corrupted"]

# Denoise
denoised = dip(noisy, device="cpu")

# Compare
plot_comparative_histogram(img, denoised)
```

---

## Purpose of the Project

This project is useful for:

* Studying **compressive imaging**
* Understanding **Hadamard transforms**
* Simulating **photon-limited imaging**
* Testing **denoising algorithms**
* Research in **computational imaging**

---

## Troubleshooting

### Error: "Image must be square"

→ Resize or crop the image to N × N

### Error: "N must be power of 2"

→ Use sizes like 64, 128, 256

### Slow performance

→ Use GPU:

```python
device = "cuda"
```

---

## License

This project uses external code from:

* Deep Image Prior repository
* ski.image for processing tools

Check their license for details.


