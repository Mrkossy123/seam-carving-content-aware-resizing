# Seam Carving for Content-Aware Image Resizing

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Processing-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Repository Information

**Repository name:**

```text
seam-carving-content-aware-resizing
```

**Short description:**

```text
Python implementation of the Seam Carving algorithm for content-aware image resizing using dynamic programming.
```

**Recommended visibility:** Public  
**Recommended `.gitignore`:** Python  
**Recommended license:** MIT License  

---

## Overview

This project is a Python implementation of the **Seam Carving** algorithm for **content-aware image resizing**, based on the SIGGRAPH 2007 paper:

> Shai Avidan and Ariel Shamir,  
> **Seam Carving for Content-Aware Image Resizing**,  
> SIGGRAPH 2007.

The goal of the project is to resize images intelligently by removing pixels that are less visually important, instead of scaling the whole image uniformly.

Unlike traditional resizing methods, seam carving tries to preserve the important visual content of an image, such as objects, people, buildings, or strong edges.

---

## What Problem Does This Project Solve?

Standard image resizing changes the size of an image by scaling all pixels equally.  
This often causes important objects to become distorted, stretched, or compressed.

Seam carving solves this problem by removing paths of pixels with low visual importance.

These paths are called **seams**.

A seam is a connected path of pixels that crosses the image:

- From top to bottom, called a **vertical seam**
- From left to right, called a **horizontal seam**

By removing seams one by one, the image can be resized while trying to preserve important content.

---

## Is This Artificial Intelligence?

This project belongs to the broader field of **Artificial Intelligence**, specifically **Computer Vision**.

However, it does **not** use Machine Learning or Deep Learning.

There is:

- No neural network
- No training phase
- No dataset
- No model weights
- No prediction model

Instead, this is a classical Computer Vision and Image Processing algorithm based on:

- Image gradients
- Energy functions
- Dynamic programming
- Optimization
- Pixel-level image manipulation

---

## Main Idea

The algorithm assigns an **energy value** to every pixel in the image.

Pixels with high energy are usually visually important.  
Examples include:

- Edges
- Object boundaries
- Texture changes
- Strong color changes
- Important image structures

Pixels with low energy are usually less important.  
Examples include:

- Smooth backgrounds
- Sky
- Empty areas
- Repetitive or flat regions

The algorithm removes the lowest-energy seams first.

---

## Energy Function

The energy function used in this implementation is based on image gradients:

```text
energy = |dI/dx| + |dI/dy|
```

Where:

- `dI/dx` is the horizontal image gradient
- `dI/dy` is the vertical image gradient

The image is first converted to grayscale before computing gradients.

This means that areas with strong changes in intensity receive higher energy values.

---

## Algorithm Steps

The seam carving process follows these steps:

1. Load the input image.
2. Convert the image to grayscale.
3. Compute the energy map using image gradients.
4. Use dynamic programming to compute the cumulative minimum energy map.
5. Backtrack through the cumulative map to find the optimal seam.
6. Remove the selected seam from the image.
7. Repeat the process until the requested number of pixels has been removed.

---

## Vertical Seam

A **vertical seam** is a connected path from the top of the image to the bottom.

It contains exactly one pixel from each row.

Removing vertical seams reduces the image width.

Example:

```text
Input image size:  800 x 600
Remove 100 seams
Output image size: 700 x 600
```

---

## Horizontal Seam

A **horizontal seam** is a connected path from the left side of the image to the right side.

It contains exactly one pixel from each column.

Removing horizontal seams reduces the image height.

Example:

```text
Input image size:  800 x 600
Remove 100 seams
Output image size: 800 x 500
```

---

## Features

This project includes:

- Energy map computation
- Vertical seam detection
- Horizontal seam detection
- Width reduction
- Height reduction
- Seam visualization
- Cumulative minimum energy map visualization
- Comparison with standard image resizing
- Command-line execution
- Fully commented Python code
- Example workflow for experimentation

---

## Core Functions

### `compute_energy(image)`

Computes the energy value of every pixel using image gradients.

```python
energy = compute_energy(image)
```

---

### `find_vertical_seam(energy)`

Finds the optimal vertical seam with the lowest total energy.

```python
seam, cumulative_map = find_vertical_seam(energy)
```

---

### `find_horizontal_seam(energy)`

Finds the optimal horizontal seam with the lowest total energy.

```python
seam, cumulative_map = find_horizontal_seam(energy)
```

---

### `reduce_width(image_in, num_pixels)`

Reduces the image width by removing vertical seams.

```python
output = reduce_width(image_in, 100)
```

---

### `reduce_height(image_in, num_pixels)`

Reduces the image height by removing horizontal seams.

```python
output = reduce_height(image_in, 100)
```

---

### `draw_seam(image, seam, orientation)`

Draws a selected seam on top of the image.

```python
image_with_seam = draw_seam(image, seam, "vertical")
```

---

## Technologies Used

- Python 3
- NumPy
- Pillow
- Matplotlib
- SciPy

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/seam-carving-content-aware-resizing.git
cd seam-carving-content-aware-resizing
```

Install the required dependencies:

```bash
pip install numpy pillow matplotlib scipy
```

---

## Usage

### Run the assignment-style demo

```bash
python seam_carving_assignment.py --austin austin.jpg --disney disney.jpg --num_pixels 100 --out results
```

This command generates:

- Seam-carved output images
- Energy maps
- Cumulative minimum energy maps
- First selected vertical seam
- First selected horizontal seam
- Comparison results with standard resizing

---

### Reduce width of a custom image

```bash
python seam_carving_assignment.py --image input.jpg --reduce_width 100 --out results
```

This removes 100 vertical seams from the input image.

---

### Reduce height of a custom image

```bash
python seam_carving_assignment.py --image input.jpg --reduce_height 100 --out results
```

This removes 100 horizontal seams from the input image.

---

### Reduce both width and height

```bash
python seam_carving_assignment.py --image input.jpg --reduce_width 80 --reduce_height 50 --out results
```

This removes:

- 80 vertical seams
- 50 horizontal seams

---

## Example Output

The system can produce the following types of results:

```text
Original Image
Energy Map
Cumulative Energy Map
Image with First Vertical Seam
Image with First Horizontal Seam
Seam-Carved Resized Image
Standard Resized Image
Comparison Between Seam Carving and Standard Resizing
```

---

## Expected Project Structure

```text
seam-carving-content-aware-resizing/
├── README.md
├── seam_carving_assignment.py
├── requirements.txt
├── .gitignore
├── LICENSE
├── images/
│   ├── austin.jpg
│   ├── disney.jpg
│   └── custom_examples/
├── results/
│   ├── energy_map.png
│   ├── vertical_seam.png
│   ├── horizontal_seam.png
│   ├── seam_carved_output.png
│   └── standard_resize_output.png
└── report/
    └── assignment_report.pdf
```

---

## Suggested `requirements.txt`

```text
numpy
pillow
matplotlib
scipy
```

---

## Suggested `.gitignore`

```gitignore
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv
venv/
results/temp/
.DS_Store
.ipynb_checkpoints/
```

---

## GitHub Repository Setup

When creating the GitHub repository, the recommended settings are:

| Field | Value |
|---|---|
| Repository name | `seam-carving-content-aware-resizing` |
| Description | `Python implementation of the Seam Carving algorithm for content-aware image resizing using dynamic programming.` |
| Visibility | Public |
| Add README | Off, if uploading this custom README manually |
| Add `.gitignore` | Python |
| Add license | MIT License |

---

## Seam Carving vs Standard Resizing

### Standard Resizing

Standard resizing scales the whole image uniformly.

Advantages:

- Fast
- Simple
- Works well for many cases

Disadvantages:

- Can distort important objects
- Can stretch people, buildings, or shapes
- Does not understand image content

---

### Seam Carving

Seam carving removes low-energy paths from the image.

Advantages:

- Preserves important regions better
- Can produce more natural results for certain images
- Works well when the image has large low-detail areas

Disadvantages:

- Can create artifacts
- Can distort structures
- Does not always work well on dense images
- Can fail when every part of the image is important

---

## When Seam Carving Works Well

Seam carving usually works well when the image contains:

- Large background areas
- Sky
- Sea
- Grass
- Roads
- Walls
- Empty space
- Smooth regions
- Objects separated from the background

---

## When Seam Carving Fails

Seam carving may produce bad results when the image contains:

- Faces
- Text
- Buildings with straight lines
- Dense crowds
- Repeating patterns
- Important objects across the whole image
- Geometric structures
- No clear background

In such cases, standard resizing or cropping may give better results.

---

## Educational Purpose

This project was developed as part of a Computer Vision assignment on content-aware image resizing.

The assignment focuses on:

- Understanding image energy
- Computing optimal seams
- Applying dynamic programming
- Comparing intelligent resizing with standard resizing
- Evaluating good and bad visual results

---

## Possible Improvements

Future improvements could include:

- Seam insertion for image enlargement
- Object removal using negative energy masks
- Protected regions using user-defined masks
- Face-aware seam carving
- Saliency-based energy functions
- Forward energy seam carving
- GUI for interactive resizing
- Batch processing for multiple images
- Video seam carving

---

## Example Analysis

A good seam carving result occurs when seams pass through low-energy background regions instead of important objects.

A bad result occurs when seams are forced to pass through important structures, causing visible distortions.

For example:

```text
Good case:
An image with sky and sea can be resized effectively because many low-energy seams pass through the background.

Bad case:
An image with buildings, faces, or text may be distorted because the algorithm may remove pixels from important structures.
```

---

## Reference

Avidan, S., & Shamir, A.  
**Seam Carving for Content-Aware Image Resizing**  
ACM Transactions on Graphics, SIGGRAPH 2007.

---

## Author

Developed by **Your Name**

Computer Vision / Image Processing Project

GitHub: [your-username](https://github.com/your-username)

---

## License

This project is licensed under the MIT License.

```text
MIT License

Copyright (c) 2025 Kossyvakis Konstantinos-Andrianos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
subject to the conditions of the MIT License.
```
