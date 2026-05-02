# Seam Carving for Content-Aware Image Resizing

A Python implementation of the **Seam Carving** algorithm for content-aware image resizing, based on the SIGGRAPH 2007 paper by **Shai Avidan and Ariel Shamir**.

Unlike standard image resizing, which scales the entire image uniformly, seam carving removes paths of low-importance pixels called **seams**. This allows an image to be resized while preserving visually important content as much as possible.

---

## Overview

This project implements a classical Computer Vision algorithm for intelligent image resizing.  
The method identifies low-energy paths in an image and removes them iteratively in order to reduce either the width or the height of the image.

The project includes:

- Energy function computation using image gradients
- Vertical seam detection
- Horizontal seam detection
- Width reduction using vertical seam removal
- Height reduction using horizontal seam removal
- Seam visualization on the original image
- Comparison between seam carving and standard resizing
- Demo scripts for experimentation with different images

---

## What Is Seam Carving?

A **seam** is a connected path of pixels that crosses the image from one side to the other.

There are two types of seams:

- **Vertical seam**: a connected path from top to bottom
- **Horizontal seam**: a connected path from left to right

Each seam contains one pixel per row or column, and the goal is to find the seam with the lowest total energy.

Pixels with low energy usually belong to less important regions of the image, such as background areas, smooth textures, or visually insignificant parts.

---

## Algorithm

The algorithm follows these main steps:

1. Convert the input image to grayscale.
2. Compute the energy of each pixel using the magnitude of the image gradients.
3. Build a cumulative minimum energy map using dynamic programming.
4. Backtrack through the cumulative map to find the optimal seam.
5. Remove the selected seam from the image.
6. Repeat the process until the desired number of pixels has been removed.

The energy function used is:

```text
energy = |dI/dx| + |dI/dy|


