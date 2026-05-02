"""
seam_carving_assignment.py

Υλοποίηση Seam Carving σε Python για την εργασία "Content-aware image resizing".

Καλύπτει:
  1. Υπολογισμό ενέργειας e1(I) = |dI/dx| + |dI/dy| σε grayscale εικόνα.
  2. Βέλτιστο vertical seam με δυναμικό προγραμματισμό.
  3. Βέλτιστο horizontal seam με δυναμικό προγραμματισμό.
  4. Μείωση πλάτους με αφαίρεση vertical seams.
  5. Μείωση ύψους με αφαίρεση horizontal seams.
  6. Εμφάνιση seam πάνω στην αρχική εικόνα.
  7. Παραγωγή των ζητούμενων plots/αποτελεσμάτων για austin.jpg και disney.jpg.
  8. Εναλλακτική ενέργεια Sobel + Gaussian smoothing για το ερώτημα αλλαγής ενέργειας.

Απαιτούμενα πακέτα:
  pip install numpy pillow matplotlib scipy

Παράδειγμα εκτέλεσης για τα ζητούμενα της εργασίας:
  python seam_carving_assignment.py --austin austin.jpg --disney disney.jpg --num_pixels 100 --out results

Σημείωση ορολογίας:
  - reduce_width  => μειώνει το width, άρα αφαιρεί vertical seams.
  - reduce_height => μειώνει το height, άρα αφαιρεί horizontal seams.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# SciPy χρησιμοποιείται μόνο για την προαιρετική εναλλακτική ενέργεια.
# Η βασική ενέργεια της εργασίας δουλεύει και χωρίς SciPy, με np.gradient.
try:
    from scipy.ndimage import gaussian_filter, sobel
except Exception:  # pragma: no cover
    gaussian_filter = None
    sobel = None


Array = np.ndarray
EnergyMode = Literal["basic", "sobel_smooth"]
Orientation = Literal["vertical", "horizontal"]


# -----------------------------------------------------------------------------
# 1. Βοηθητικές συναρτήσεις για είσοδο/έξοδο εικόνας
# -----------------------------------------------------------------------------

def load_image(path: str | Path) -> Array:
    """
    Διαβάζει εικόνα από δίσκο και επιστρέφει RGB numpy array τύπου float64
    με τιμές στο [0, 1].

    Χρησιμοποιούμε float/double για τους υπολογισμούς, όπως ζητά η εργασία.
    """
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def save_image(image: Array, path: str | Path) -> None:
    """
    Αποθηκεύει εικόνα που βρίσκεται είτε στο [0, 1] είτε στο [0, 255].
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = np.asarray(image)
    if img.dtype.kind == "f":
        img = np.clip(img, 0.0, 1.0)
        img_u8 = (255.0 * img + 0.5).astype(np.uint8)
    else:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)

    Image.fromarray(img_u8).save(path)


def as_float_rgb(image: Array) -> Array:
    """
    Κανονικοποιεί είσοδο εικόνας σε float64 RGB array με τιμές [0, 1].

    Δέχεται:
      - grayscale εικόνα HxW,
      - RGB εικόνα HxWx3,
      - RGBA εικόνα HxWx4, όπου αγνοείται το alpha κανάλι.
    """
    img = np.asarray(image)

    # Αν είναι uint8, τη φέρνουμε στο [0, 1].
    if img.dtype.kind in "ui":
        img = img.astype(np.float64) / 255.0
    else:
        img = img.astype(np.float64)
        # Αν κάποιος περάσει float εικόνα στο [0, 255], τη φέρνουμε στο [0, 1].
        if img.size > 0 and img.max() > 1.0:
            img = img / 255.0

    # Grayscale HxW -> RGB HxWx3 για ομοιομορφία στις συναρτήσεις αφαίρεσης.
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)

    # RGBA -> RGB.
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Η εικόνα πρέπει να είναι HxW, HxWx3 ή HxWx4.")

    return np.clip(img, 0.0, 1.0)


def rgb_to_gray(image: Array) -> Array:
    """
    Μετατρέπει RGB εικόνα σε grayscale με τα κλασικά luminance βάρη.

    Η εργασία ζητά η προβολή/αποτέλεσμα να είναι έγχρωμο, αλλά τα gradients
    να υπολογίζονται στην grayscale μετατροπή της εικόνας.
    """
    img = as_float_rgb(image)
    return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]


# -----------------------------------------------------------------------------
# 2. Ενέργεια εικόνας
# -----------------------------------------------------------------------------

def compute_energy(
    image: Array,
    mode: EnergyMode = "basic",
    sigma: float = 1.0,
) -> Array:
    """
    Υπολογίζει την ενέργεια κάθε pixel.

    mode="basic":
        Υλοποιεί την εξίσωση e1(I) = |dI/dx| + |dI/dy|.
        Τα παράγωγα υπολογίζονται με κεντρικές διαφορές μέσω np.gradient.

    mode="sobel_smooth":
        Εναλλακτική ενέργεια για το ερώτημα "κάντε κάποια αλλαγή στην ενέργεια".
        Πρώτα γίνεται Gaussian smoothing και μετά Sobel gradient.
        Αυτό συνήθως μειώνει την ευαισθησία σε θόρυβο και μικρολεπτομέρειες,
        οπότε τα seams τείνουν να καθοδηγούνται από πιο μακροσκοπικές ακμές.
    """
    gray = rgb_to_gray(image)

    if mode == "basic":
        # np.gradient επιστρέφει πρώτα παράγωγο ως προς τον άξονα 0 (y/rows)
        # και μετά ως προς τον άξονα 1 (x/columns).
        d_y, d_x = np.gradient(gray)
        energy = np.abs(d_x) + np.abs(d_y)
        return energy.astype(np.float64)

    if mode == "sobel_smooth":
        if gaussian_filter is None or sobel is None:
            raise ImportError(
                "Για mode='sobel_smooth' χρειάζεται scipy. "
                "Εγκατάσταση: pip install scipy"
            )
        # Εξομάλυνση πριν τα gradients ώστε να μη μετράμε υπερβολικά θόρυβο.
        smooth = gaussian_filter(gray, sigma=sigma, mode="reflect")
        d_x = sobel(smooth, axis=1, mode="reflect")
        d_y = sobel(smooth, axis=0, mode="reflect")
        energy = np.abs(d_x) + np.abs(d_y)
        return energy.astype(np.float64)

    raise ValueError("Άγνωστο mode ενέργειας. Επιλογές: 'basic', 'sobel_smooth'.")


# -----------------------------------------------------------------------------
# 3. Δυναμικός προγραμματισμός για cumulative minimum energy map
# -----------------------------------------------------------------------------

def cumulative_map_vertical(energy: Array) -> Tuple[Array, Array]:
    """
    Υπολογίζει τον cumulative minimum energy map M για vertical seams.

    Για κάθε pixel (i, j), το M[i, j] είναι το ελάχιστο συνολικό κόστος seam
    που ξεκινά από την πρώτη γραμμή και καταλήγει στο pixel (i, j).

    Αναδρομή:
      M[i, j] = energy[i, j] + min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])

    Επιστρέφει:
      M         : cumulative energy map, ίδιο μέγεθος με energy.
      backtrack : πίνακας ίδιου μεγέθους που κρατά offset {-1, 0, +1}.
                  Αν είμαστε στο pixel (i, j), τότε το προηγούμενο pixel
                  του seam βρίσκεται στο (i-1, j + backtrack[i, j]).
    """
    e = np.asarray(energy, dtype=np.float64)
    if e.ndim != 2:
        raise ValueError("Το energy πρέπει να είναι 2D array.")

    h, w = e.shape
    M = e.copy()
    backtrack = np.zeros((h, w), dtype=np.int8)

    # Δεν υπάρχει προηγούμενη γραμμή για i=0, άρα ξεκινάμε από i=1.
    for i in range(1, h):
        prev = M[i - 1]

        # Για κάθε στήλη j, δημιουργούμε τα τρία πιθανά προηγούμενα κόστη.
        # Στα όρια βάζουμε inf ώστε να μην επιλεγεί ανύπαρκτος γείτονας.
        from_left_diag = np.empty(w, dtype=np.float64)   # M[i-1, j-1]
        from_up = prev                                  # M[i-1, j]
        from_right_diag = np.empty(w, dtype=np.float64)  # M[i-1, j+1]

        from_left_diag[0] = np.inf
        from_left_diag[1:] = prev[:-1]

        from_right_diag[:-1] = prev[1:]
        from_right_diag[-1] = np.inf

        candidates = np.vstack((from_left_diag, from_up, from_right_diag))
        best_idx = np.argmin(candidates, axis=0)

        # Με best_idx 0,1,2 αντιστοιχίζουμε offsets -1,0,+1.
        offsets = np.array([-1, 0, 1], dtype=np.int8)
        best_offsets = offsets[best_idx]
        best_costs = candidates[best_idx, np.arange(w)]

        M[i] = e[i] + best_costs
        backtrack[i] = best_offsets

    return M, backtrack


def cumulative_map_horizontal(energy: Array) -> Tuple[Array, Array]:
    """
    Υπολογίζει cumulative minimum energy map για horizontal seams.

    Αντί να ξαναγράψουμε όλο τον αλγόριθμο, λύνουμε το αντίστοιχο vertical
    πρόβλημα πάνω στο transpose της ενέργειας. Έτσι ο seam διασχίζει την
    εικόνα από αριστερά προς δεξιά.

    Επιστρέφει M και backtrack σε διαστάσεις της αρχικής εικόνας.
    """
    M_t, back_t = cumulative_map_vertical(np.asarray(energy).T)
    return M_t.T, back_t.T


# -----------------------------------------------------------------------------
# 4. Εύρεση βέλτιστου seam
# -----------------------------------------------------------------------------

def find_vertical_seam(energy: Array) -> Tuple[Array, float, Array]:
    """
    Βρίσκει το βέλτιστο vertical seam.

    Επιστρέφει:
      seam      : 1D array μήκους H. seam[i] = στήλη j του seam στη γραμμή i.
      seam_cost : συνολικό ελάχιστο κόστος του seam.
      M         : cumulative minimum energy map για πιθανή απεικόνιση.
    """
    M, backtrack = cumulative_map_vertical(energy)
    h, w = M.shape

    seam = np.zeros(h, dtype=np.int32)

    # Το seam τελειώνει στην ελάχιστη τιμή της τελευταίας γραμμής του M.
    j = int(np.argmin(M[-1]))
    seam_cost = float(M[-1, j])

    # Backtracking από κάτω προς τα πάνω.
    for i in range(h - 1, -1, -1):
        seam[i] = j
        if i > 0:
            j = j + int(backtrack[i, j])
            # Προστασία από αριθμητικά/λογικά σφάλματα στα όρια.
            j = int(np.clip(j, 0, w - 1))

    return seam, seam_cost, M


def find_horizontal_seam(energy: Array) -> Tuple[Array, float, Array]:
    """
    Βρίσκει το βέλτιστο horizontal seam.

    Επιστρέφει:
      seam      : 1D array μήκους W. seam[j] = γραμμή i του seam στη στήλη j.
      seam_cost : συνολικό ελάχιστο κόστος του seam.
      M         : cumulative minimum energy map στο αρχικό σχήμα HxW.
    """
    # Horizontal seam στην αρχική εικόνα = vertical seam στην transposed εικόνα.
    seam_t, seam_cost, M_t = find_vertical_seam(np.asarray(energy).T)
    M = M_t.T
    return seam_t, seam_cost, M


# -----------------------------------------------------------------------------
# 5. Αφαίρεση seam από έγχρωμη εικόνα
# -----------------------------------------------------------------------------

def remove_vertical_seam(image: Array, seam: Array) -> Array:
    """
    Αφαιρεί ένα vertical seam από RGB εικόνα.

    Είσοδος:
      image : HxWx3
      seam  : μήκους H, seam[i] = στήλη που αφαιρείται από τη γραμμή i

    Έξοδος:
      Hx(W-1)x3 εικόνα.
    """
    img = as_float_rgb(image)
    h, w, c = img.shape
    seam = np.asarray(seam, dtype=np.int32)

    if seam.shape != (h,):
        raise ValueError(f"Το vertical seam πρέπει να έχει μήκος H={h}.")
    if w <= 1:
        raise ValueError("Δεν μπορεί να αφαιρεθεί seam από εικόνα με width <= 1.")

    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False

    # Το boolean indexing δίνει H*(W-1) pixels, τα ξανακάνουμε εικόνα.
    out = img[mask].reshape(h, w - 1, c)
    return out


def remove_horizontal_seam(image: Array, seam: Array) -> Array:
    """
    Αφαιρεί ένα horizontal seam από RGB εικόνα.

    Είσοδος:
      image : HxWx3
      seam  : μήκους W, seam[j] = γραμμή που αφαιρείται από τη στήλη j

    Έξοδος:
      (H-1)xWx3 εικόνα.
    """
    img = as_float_rgb(image)
    h, w, c = img.shape
    seam = np.asarray(seam, dtype=np.int32)

    if seam.shape != (w,):
        raise ValueError(f"Το horizontal seam πρέπει να έχει μήκος W={w}.")
    if h <= 1:
        raise ValueError("Δεν μπορεί να αφαιρεθεί seam από εικόνα με height <= 1.")

    # Για απλότητα και ασφάλεια, δουλεύουμε ανά στήλη.
    out = np.zeros((h - 1, w, c), dtype=img.dtype)
    for j in range(w):
        i_remove = seam[j]
        out[:, j, :] = np.delete(img[:, j, :], i_remove, axis=0)

    return out


# -----------------------------------------------------------------------------
# 6. Public API αντίστοιχο με ReduceWidth / ReduceHeight
# -----------------------------------------------------------------------------

def reduce_width(
    image_in: Array,
    num_pixels: int,
    energy_mode: EnergyMode = "basic",
    sigma: float = 1.0,
    verbose: bool = True,
) -> Array:
    """
    Μειώνει το πλάτος της εικόνας κατά num_pixels.

    Κάθε επανάληψη:
      1. Υπολογίζει ενέργεια στην τρέχουσα εικόνα.
      2. Βρίσκει το βέλτιστο vertical seam.
      3. Αφαιρεί αυτό το seam.

    Επιστρέφει float RGB εικόνα στο [0, 1].
    """
    img = as_float_rgb(image_in).copy()
    h, w, _ = img.shape

    if num_pixels < 0:
        raise ValueError("Το num_pixels πρέπει να είναι μη αρνητικό.")
    if num_pixels >= w:
        raise ValueError(f"Δεν γίνεται να αφαιρεθούν {num_pixels} seams από width={w}.")

    for k in range(num_pixels):
        energy = compute_energy(img, mode=energy_mode, sigma=sigma)
        seam, _, _ = find_vertical_seam(energy)
        img = remove_vertical_seam(img, seam)

        if verbose and ((k + 1) % 20 == 0 or k == num_pixels - 1):
            print(f"reduce_width: αφαιρέθηκαν {k + 1}/{num_pixels} vertical seams")

    return img


def reduce_height(
    image_in: Array,
    num_pixels: int,
    energy_mode: EnergyMode = "basic",
    sigma: float = 1.0,
    verbose: bool = True,
) -> Array:
    """
    Μειώνει το ύψος της εικόνας κατά num_pixels.

    Κάθε επανάληψη:
      1. Υπολογίζει ενέργεια στην τρέχουσα εικόνα.
      2. Βρίσκει το βέλτιστο horizontal seam.
      3. Αφαιρεί αυτό το seam.

    Επιστρέφει float RGB εικόνα στο [0, 1].
    """
    img = as_float_rgb(image_in).copy()
    h, w, _ = img.shape

    if num_pixels < 0:
        raise ValueError("Το num_pixels πρέπει να είναι μη αρνητικό.")
    if num_pixels >= h:
        raise ValueError(f"Δεν γίνεται να αφαιρεθούν {num_pixels} seams από height={h}.")

    for k in range(num_pixels):
        energy = compute_energy(img, mode=energy_mode, sigma=sigma)
        seam, _, _ = find_horizontal_seam(energy)
        img = remove_horizontal_seam(img, seam)

        if verbose and ((k + 1) % 20 == 0 or k == num_pixels - 1):
            print(f"reduce_height: αφαιρέθηκαν {k + 1}/{num_pixels} horizontal seams")

    return img


# -----------------------------------------------------------------------------
# 7. Οπτικοποίηση seams, ενέργειας, cumulative maps
# -----------------------------------------------------------------------------

def draw_seam(
    image: Array,
    seam: Array,
    orientation: Orientation,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Σχεδιάζει seam πάνω στην εικόνα.

    orientation="vertical":
      seam[i] δίνει τη στήλη σε κάθε γραμμή i.

    orientation="horizontal":
      seam[j] δίνει τη γραμμή σε κάθε στήλη j.
    """
    img = as_float_rgb(image)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    ax.imshow(img)
    ax.axis("off")

    if orientation == "vertical":
        rows = np.arange(img.shape[0])
        cols = np.asarray(seam)
        ax.plot(cols, rows, linewidth=1.5)
    elif orientation == "horizontal":
        cols = np.arange(img.shape[1])
        rows = np.asarray(seam)
        ax.plot(cols, rows, linewidth=1.5)
    else:
        raise ValueError("orientation πρέπει να είναι 'vertical' ή 'horizontal'.")

    if title:
        ax.set_title(title)
    return ax


def save_first_seams_figure(image: Array, out_path: str | Path) -> None:
    """
    Για την ίδια εικόνα εμφανίζει:
      - το πρώτο horizontal seam,
      - το πρώτο vertical seam.
    """
    img = as_float_rgb(image)
    energy = compute_energy(img, mode="basic")
    v_seam, v_cost, _ = find_vertical_seam(energy)
    h_seam, h_cost, _ = find_horizontal_seam(energy)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    draw_seam(img, h_seam, "horizontal", ax=axes[0], title=f"First horizontal seam, cost={h_cost:.3f}")
    draw_seam(img, v_seam, "vertical", ax=axes[1], title=f"First vertical seam, cost={v_cost:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_energy_and_maps_figure(image: Array, out_path: str | Path) -> None:
    """
    Αποθηκεύει figure με:
      (a) την energy function e1,
      (b) cumulative map για vertical seams,
      (c) cumulative map για horizontal seams.
    """
    img = as_float_rgb(image)
    energy = compute_energy(img, mode="basic")
    M_v, _ = cumulative_map_vertical(energy)
    M_h, _ = cumulative_map_horizontal(energy)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(energy, cmap="gray")
    axes[0].set_title("Energy e1 = |Ix| + |Iy|")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(M_v, cmap="viridis")
    axes[1].set_title("Cumulative map M - vertical seams")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(M_h, cmap="viridis")
    axes[2].set_title("Cumulative map M - horizontal seams")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def simple_resize_like(image: Array, target_shape: Tuple[int, int]) -> Array:
    """
    Κάνει απλό resampling τύπου imresize για σύγκριση.

    target_shape = (target_height, target_width)
    """
    img = as_float_rgb(image)
    target_h, target_w = target_shape
    pil = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    resized = pil.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float64) / 255.0


def save_comparison_figure(
    original: Array,
    carved: Array,
    out_path: str | Path,
    title: str,
    sequence_text: str,
) -> None:
    """
    Αποθηκεύει figure που περιέχει:
      (a) αρχική εικόνα,
      (b) seam-carved εικόνα,
      (c) απλό resampling στο ίδιο τελικό μέγεθος.
    """
    original = as_float_rgb(original)
    carved = as_float_rgb(carved)
    blind = simple_resize_like(original, carved.shape[:2])

    h0, w0 = original.shape[:2]
    h1, w1 = carved.shape[:2]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title(f"Original\n{h0}x{w0}")
    axes[0].axis("off")

    axes[1].imshow(carved)
    axes[1].set_title(f"Seam carving\n{h1}x{w1}")
    axes[1].axis("off")

    axes[2].imshow(blind)
    axes[2].set_title(f"Simple resize / bicubic\n{h1}x{w1}")
    axes[2].axis("off")

    fig.suptitle(f"{title}\nSequence: {sequence_text}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def retarget_by_sequence(
    image: Array,
    operations: Sequence[Tuple[Orientation, int]],
    energy_mode: EnergyMode = "basic",
    sigma: float = 1.0,
    verbose: bool = True,
) -> Array:
    """
    Εκτελεί διαδοχικές αφαιρέσεις seams.

    operations παραδείγματα:
      [("vertical", 80)]
          Μειώνει width κατά 80.

      [("horizontal", 40), ("vertical", 60)]
          Πρώτα μειώνει height κατά 40, μετά width κατά 60.

    Προσοχή:
      Εδώ η λέξη vertical/horizontal αναφέρεται στον τύπο seam που αφαιρείται.
      - vertical seam   => μειώνει width
      - horizontal seam => μειώνει height
    """
    img = as_float_rgb(image)

    for orientation, n in operations:
        if orientation == "vertical":
            img = reduce_width(img, n, energy_mode=energy_mode, sigma=sigma, verbose=verbose)
        elif orientation == "horizontal":
            img = reduce_height(img, n, energy_mode=energy_mode, sigma=sigma, verbose=verbose)
        else:
            raise ValueError("Κάθε operation πρέπει να είναι 'vertical' ή 'horizontal'.")

    return img


# -----------------------------------------------------------------------------
# 8. Demo για τα ζητούμενα της εργασίας
# -----------------------------------------------------------------------------

def run_assignment_demo(
    austin_path: str | Path,
    disney_path: str | Path,
    num_pixels: int = 100,
    out_dir: str | Path = "results",
) -> None:
    """
    Τρέχει ένα πλήρες demo για τα υποχρεωτικά ζητούμενα:
      - reduce_width(austin, 100)
      - reduce_height(disney, 100)
      - energy και cumulative maps για austin
      - πρώτο horizontal και vertical seam για austin
      - σύγκριση βασικής ενέργειας με εναλλακτική Sobel+Gaussian ενέργεια
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    austin = load_image(austin_path)
    disney = load_image(disney_path)

    print("[1/5] Μείωση πλάτους austin με reduce_width...")
    austin_rw = reduce_width(austin, num_pixels, energy_mode="basic", verbose=True)
    save_image(austin_rw, out_dir / f"austin_reduce_width_{num_pixels}.png")
    save_comparison_figure(
        original=austin,
        carved=austin_rw,
        out_path=out_dir / f"austin_reduce_width_{num_pixels}_comparison.png",
        title="Austin: reduce_width with basic energy",
        sequence_text=f"remove {num_pixels} vertical seams",
    )

    print("[2/5] Μείωση ύψους disney με reduce_height...")
    disney_rh = reduce_height(disney, num_pixels, energy_mode="basic", verbose=True)
    save_image(disney_rh, out_dir / f"disney_reduce_height_{num_pixels}.png")
    save_comparison_figure(
        original=disney,
        carved=disney_rh,
        out_path=out_dir / f"disney_reduce_height_{num_pixels}_comparison.png",
        title="Disney: reduce_height with basic energy",
        sequence_text=f"remove {num_pixels} horizontal seams",
    )

    print("[3/5] Αποθήκευση energy και cumulative maps για austin...")
    save_energy_and_maps_figure(austin, out_dir / "austin_energy_and_cumulative_maps.png")

    print("[4/5] Αποθήκευση πρώτου horizontal και vertical seam για austin...")
    save_first_seams_figure(austin, out_dir / "austin_first_horizontal_and_vertical_seams.png")

    print("[5/5] Εναλλακτική ενέργεια Sobel + Gaussian για σύγκριση...")
    try:
        austin_rw_sobel = reduce_width(
            austin,
            num_pixels,
            energy_mode="sobel_smooth",
            sigma=1.2,
            verbose=True,
        )
        save_comparison_figure(
            original=austin,
            carved=austin_rw_sobel,
            out_path=out_dir / f"austin_reduce_width_{num_pixels}_sobel_smooth_comparison.png",
            title="Austin: reduce_width with Sobel + Gaussian energy",
            sequence_text=f"remove {num_pixels} vertical seams, sigma=1.2",
        )
    except ImportError as exc:
        print(f"Παράλειψη Sobel+Gaussian demo: {exc}")

    print(f"\nΈτοιμο. Τα αποτελέσματα αποθηκεύτηκαν στον φάκελο: {out_dir.resolve()}")


# -----------------------------------------------------------------------------
# 9. Command-line interface
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seam carving implementation for content-aware image resizing."
    )

    parser.add_argument("--austin", type=str, default=None, help="Path στο austin.jpg")
    parser.add_argument("--disney", type=str, default=None, help="Path στο disney.jpg")
    parser.add_argument("--num_pixels", type=int, default=100, help="Πόσα seams/pixels να αφαιρεθούν")
    parser.add_argument("--out", type=str, default="results", help="Φάκελος εξόδου")

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Προαιρετικά: path σε μία εικόνα για custom seam carving.",
    )
    parser.add_argument(
        "--reduce_width",
        type=int,
        default=0,
        help="Custom mode: πόσο να μειωθεί το width.",
    )
    parser.add_argument(
        "--reduce_height",
        type=int,
        default=0,
        help="Custom mode: πόσο να μειωθεί το height.",
    )
    parser.add_argument(
        "--energy",
        type=str,
        default="basic",
        choices=["basic", "sobel_smooth"],
        help="Τύπος ενέργειας.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Sigma για Gaussian smoothing στο sobel_smooth.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Custom mode για οποιαδήποτε εικόνα του χρήστη.
    if args.image is not None:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        img = load_image(args.image)
        original = img.copy()

        operations: list[Tuple[Orientation, int]] = []
        if args.reduce_width > 0:
            operations.append(("vertical", args.reduce_width))
        if args.reduce_height > 0:
            operations.append(("horizontal", args.reduce_height))

        if not operations:
            raise ValueError("Δώσε --reduce_width ή/και --reduce_height για custom mode.")

        carved = retarget_by_sequence(
            img,
            operations,
            energy_mode=args.energy,  # type: ignore[arg-type]
            sigma=args.sigma,
            verbose=True,
        )

        stem = Path(args.image).stem
        out_image = out_dir / f"{stem}_carved.png"
        out_compare = out_dir / f"{stem}_comparison.png"

        save_image(carved, out_image)
        sequence = ", ".join(f"remove {n} {ori} seams" for ori, n in operations)
        save_comparison_figure(original, carved, out_compare, title=stem, sequence_text=sequence)

        print(f"Αποθηκεύτηκε: {out_image}")
        print(f"Αποθηκεύτηκε: {out_compare}")
        return

    # Assignment mode για austin/disney.
    if args.austin is None or args.disney is None:
        raise ValueError(
            "Για assignment demo δώσε και --austin και --disney, ή δώσε --image για custom mode."
        )

    run_assignment_demo(
        austin_path=args.austin,
        disney_path=args.disney,
        num_pixels=args.num_pixels,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
