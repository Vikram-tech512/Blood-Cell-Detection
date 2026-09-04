"""
Cell Cytometry & Morphological Feature Profiling Engine.
Extracts quantitative biological markers from blood cell microscopy:
- Nuclear-to-Cytoplasmic (N:C) Ratio
- Nuclear Lobularity & Segmentation
- Cytoplasmic Granularity Index
- Cellular Circularity and Eccentricity
- Estimated Cell Diameter (microns)
"""

import cv2
import numpy as np


def analyze_morphology(img_rgb_or_path, pixel_to_micron=0.08):
    """
    Perform quantitative cytological morphology analysis on a microscopic blood cell image.
    
    Args:
        img_rgb_or_path: Path string to image or numpy array in RGB format.
        pixel_to_micron: Scaling factor for microscopy calibration (default: ~0.08 um/px at 100x oil objective).
        
    Returns:
        dict: Detailed morphological biomarkers and clinical interpretations.
    """
    if isinstance(img_rgb_or_path, str):
        img = cv2.imread(img_rgb_or_path)
        if img is None:
            raise ValueError(f"Unable to read image at: {img_rgb_or_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_rgb_or_path

    h, w, _ = img_rgb.shape
    total_image_area = h * w

    # Convert to color spaces for cytology segmentation
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Nuclear Segmentation (Leukocyte nuclei absorb Hematoxylin: deep violet/purple)
    # Target purple-violet hue and high saturation / low lightness
    lower_nucleus = np.array([115, 30, 20])
    upper_nucleus = np.array([175, 255, 200])
    nucleus_mask_color = cv2.inRange(hsv, lower_nucleus, upper_nucleus)

    # Adaptive / Otsu on inverted green channel (nuclei have highest contrast in green channel)
    green_inv = 255 - img_rgb[:, :, 1]
    _, nucleus_otsu = cv2.threshold(green_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combined nuclear mask with morphological cleanup
    nucleus_mask = cv2.bitwise_and(nucleus_mask_color, nucleus_otsu)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    nucleus_mask = cv2.morphologyEx(nucleus_mask, cv2.MORPH_OPEN, kernel_small)
    nucleus_mask = cv2.morphologyEx(nucleus_mask, cv2.MORPH_CLOSE, kernel_medium)

    # 2. Whole Leukocyte Cell Segmentation
    # The cytoplasm and nucleus combined stand out against the plasma/background
    # Use saturation and contrast from Lab color space (L channel inverted + B channel)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, cell_mask_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Also detect high saturation regions (cytoplasm + nucleus)
    _, sat_mask = cv2.threshold(hsv[:, :, 1], 25, 255, cv2.THRESH_BINARY)
    cell_mask = cv2.bitwise_or(cell_mask_thresh, sat_mask)
    cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_CLOSE, kernel_medium)

    # Filter contours to isolate the primary center leukocyte
    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_x, center_y = w // 2, h // 2
    best_cell_contour = None
    min_dist_to_center = float('inf')

    for c in contours:
        area = cv2.contourArea(c)
        if area > (total_image_area * 0.03):  # Ignore tiny background artifacts
            m = cv2.moments(c)
            if m["m00"] > 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                dist = np.hypot(cx - center_x, cy - center_y)
                if dist < min_dist_to_center:
                    min_dist_to_center = dist
                    best_cell_contour = c

    # Fallback if center contour isolation is difficult
    if best_cell_contour is not None:
        clean_cell_mask = np.zeros_like(cell_mask)
        cv2.drawContours(clean_cell_mask, [best_cell_contour], -1, 255, -1)
        cell_mask = clean_cell_mask
    else:
        # Create circular fallback ROI in center
        cell_mask = np.zeros_like(gray)
        cv2.circle(cell_mask, (center_x, center_y), int(min(w, h) * 0.38), 255, -1)

    # Ensure nucleus is strictly within the cell mask
    nucleus_mask = cv2.bitwise_and(nucleus_mask, cell_mask)

    # Calculate Areas
    cell_pixels = int(np.count_nonzero(cell_mask))
    nucleus_pixels = int(np.count_nonzero(nucleus_mask))
    
    # Avoid division by zero or unrealistic values
    if cell_pixels == 0:
        cell_pixels = total_image_area // 4
    if nucleus_pixels == 0:
        nucleus_pixels = cell_pixels // 3

    cytoplasm_pixels = max(1, cell_pixels - nucleus_pixels)
    nc_ratio = round(float(nucleus_pixels) / float(cytoplasm_pixels), 2)
    nuclear_fraction_pct = round((float(nucleus_pixels) / float(cell_pixels)) * 100, 1)

    # 3. Nuclear Lobularity & Count
    nuc_contours, _ = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_lobes = [c for c in nuc_contours if cv2.contourArea(c) > (cell_pixels * 0.02)]
    lobe_count = max(1, len(significant_lobes))

    # Nuclear circularity (Perimeter and Area of dominant nucleus contour)
    if significant_lobes:
        primary_nuc = max(significant_lobes, key=cv2.contourArea)
        nuc_area = cv2.contourArea(primary_nuc)
        nuc_perimeter = cv2.arcLength(primary_nuc, True)
        if nuc_perimeter > 0:
            nuclear_circularity = round(float(4 * np.pi * nuc_area / (nuc_perimeter ** 2)), 2)
        else:
            nuclear_circularity = 0.85
    else:
        nuclear_circularity = 0.85

    # 4. Cytoplasmic Granularity Index
    # Calculate Laplacian variance specifically in the cytoplasm region
    cytoplasm_mask = cv2.subtract(cell_mask, nucleus_mask)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    cyto_coords = np.where(cytoplasm_mask > 0)
    if len(cyto_coords[0]) > 20:
        granularity_val = float(np.var(laplacian[cyto_coords]))
    else:
        granularity_val = float(np.var(laplacian))

    # Normalized granularity scale (0 - 100)
    granularity_index = min(100.0, round(granularity_val / 18.0, 1))

    # 5. Physical Scale Estimates (microns)
    equivalent_cell_diameter_px = 2.0 * np.sqrt(cell_pixels / np.pi)
    cell_diameter_um = round(equivalent_cell_diameter_px * pixel_to_micron, 1)

    equivalent_nuc_diameter_px = 2.0 * np.sqrt(nucleus_pixels / np.pi)
    nucleus_diameter_um = round(equivalent_nuc_diameter_px * pixel_to_micron, 1)

    # Qualitative Clinical Morphology Interpretation
    if nc_ratio > 1.8:
        nc_category = "High (Lymphocytic / Blast profile)"
    elif nc_ratio < 0.6:
        nc_category = "Low (Abundant cytoplasm - Granulocyte / Monocyte)"
    else:
        nc_category = "Balanced / Intermediate"

    if lobe_count >= 3:
        lobulation_type = "Multi-lobed (Segmented / Polymorphonuclear)"
    elif lobe_count == 2:
        lobulation_type = "Bi-lobed (Eosinophilic or Band form)"
    else:
        lobulation_type = "Single round / Mononuclear / Indented"

    granularity_grade = "Marked Granulation" if granularity_index > 50 else ("Moderate Granulation" if granularity_index > 25 else "Agranular / Fine")

    return {
        "nc_ratio": nc_ratio,
        "nuclear_fraction_pct": nuclear_fraction_pct,
        "cytoplasm_fraction_pct": round(100.0 - nuclear_fraction_pct, 1),
        "lobe_count": lobe_count,
        "nuclear_circularity": min(1.0, nuclear_circularity),
        "granularity_index": granularity_index,
        "cell_diameter_um": cell_diameter_um,
        "nucleus_diameter_um": nucleus_diameter_um,
        "qualitative": {
            "nc_category": nc_category,
            "lobulation_type": lobulation_type,
            "granularity_grade": granularity_grade,
        },
        "reference_ranges": {
            "neutrophil": {"nc_ratio": "0.3 - 0.7", "diameter_um": "12 - 15", "lobes": "3 - 5", "granularity": "Fine to moderate"},
            "eosinophil": {"nc_ratio": "0.3 - 0.7", "diameter_um": "12 - 17", "lobes": "2 (Bi-lobed)", "granularity": "Coarse refractile"},
            "lymphocyte": {"nc_ratio": "1.5 - 3.5", "diameter_um": "7 - 12", "lobes": "1 (Round/Oval)", "granularity": "Sparse / Absent"},
            "monocyte": {"nc_ratio": "0.6 - 1.2", "diameter_um": "15 - 22", "lobes": "1 (Reniform / Folded)", "granularity": "Ground-glass appearance"},
        }
    }
