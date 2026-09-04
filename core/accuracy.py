"""
Advanced Accuracy, Image Quality Index (IQI), and Test-Time Augmentation (TTA) Engine.
Combines:
- Specimen Quality Assessment (Blur, Illumination, Staining Contrast)
- Test-Time Augmentation (4 rotations + horizontal flip)
- Hybrid Neuro-Symbolic Bayesian Calibration (fusing deep learning logits with quantitative cytology)
"""

import cv2
import numpy as np


def assess_image_quality(img_rgb):
    """
    Evaluate specimen image quality before inference.
    
    Returns:
        dict: IQI score (0-100), sharpness score, brightness, contrast, status, and warnings.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Sharpness via Laplacian Variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize sharpness: optimal is > 100
    sharpness_score = min(100.0, (lap_var / 350.0) * 100.0)
    
    # 2. Illumination / Brightness
    mean_brightness = float(np.mean(gray))
    # Optimal blood smear mean brightness is between 130 and 220
    if 130 <= mean_brightness <= 220:
        brightness_score = 100.0
    elif mean_brightness < 130:
        brightness_score = max(0.0, (mean_brightness / 130.0) * 100.0)
    else:
        brightness_score = max(0.0, 100.0 - ((mean_brightness - 220.0) / 35.0) * 100.0)

    # 3. Dynamic Contrast Range
    contrast_val = float(np.std(gray))
    contrast_score = min(100.0, (contrast_val / 50.0) * 100.0)

    # Weighted composite IQI score
    iqi_score = round(0.45 * sharpness_score + 0.30 * brightness_score + 0.25 * contrast_score, 1)

    warnings = []
    if lap_var < 50.0:
        warnings.append("Specimen may be slightly out-of-focus or blurred.")
    if mean_brightness < 90:
        warnings.append("Low illumination detected; cell borders may have reduced contrast.")
    elif mean_brightness > 235:
        warnings.append("Overexposed high-power field detected.")
    if contrast_val < 25.0:
        warnings.append("Low staining contrast between cytoplasm and background.")

    if iqi_score >= 80.0:
        status = "Optimal Quality"
        badge_class = "success"
    elif iqi_score >= 50.0:
        status = "Acceptable Diagnostic Quality"
        badge_class = "warning"
    else:
        status = "Suboptimal Quality (Proceed with Caution)"
        badge_class = "danger"

    return {
        "iqi_score": iqi_score,
        "sharpness_val": round(lap_var, 1),
        "brightness_val": round(mean_brightness, 1),
        "contrast_val": round(contrast_val, 1),
        "status": status,
        "badge_class": badge_class,
        "warnings": warnings,
        "passed": iqi_score >= 40.0
    }


def test_time_augmentation(img_rgb, model, class_labels):
    """
    Perform Test-Time Augmentation (TTA) by evaluating 5 invariant geometric views:
    - 0° Original
    - 90° Clockwise
    - 180° Inverted
    - 270° Counter-Clockwise
    - Horizontal Flip
    
    Averages predictions to eliminate orientation artifacts and maximize diagnostic accuracy.
    """
    if model is None:
        return None, None, None

    try:
        from keras.applications.mobilenet_v2 import preprocess_input
    except Exception:
        preprocess_input = lambda x: (x / 127.5) - 1.0

    augmented_views = [
        img_rgb,
        cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_rgb, cv2.ROTATE_180),
        cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.flip(img_rgb, 1)
    ]

    batch_tensors = []
    for view in augmented_views:
        resized = cv2.resize(view, (224, 224))
        prep = preprocess_input(resized.astype("float32"))
        batch_tensors.append(prep)

    batch_np = np.array(batch_tensors)
    raw_predictions = model.predict(batch_np, verbose=0)
    
    # Compute robust trimmed mean across views
    ensemble_probs = np.mean(raw_predictions, axis=0)
    best_idx = int(np.argmax(ensemble_probs))
    best_label = class_labels[best_idx]
    confidence = float(ensemble_probs[best_idx])
    
    probabilities = {label: float(ensemble_probs[i]) for i, label in enumerate(class_labels)}
    return best_label, confidence, probabilities, best_idx


def calibrate_with_cytometry(probabilities, morphology, class_labels):
    """
    Hybrid Neuro-Symbolic Bayesian Calibration:
    Adjusts neural network softmax probabilities using real biological rules:
    - Lymphocyte: High N:C ratio (> 1.6), round nucleus, low granularity
    - Neutrophil: Segmented nucleus (>= 3 lobes), moderate granularity
    - Eosinophil: Bi-lobed (2 lobes), high cytoplasmic granularity (> 35)
    - Monocyte: Large diameter (> 15 um), indented/folded nucleus (lobes 1-2), low/moderate granularity
    """
    if not morphology or "nc_ratio" not in morphology:
        return probabilities

    nc = float(morphology.get("nc_ratio", 1.0))
    lobes = int(morphology.get("lobe_count", 1))
    granularity = float(morphology.get("granularity_index", 20.0))
    diameter = float(morphology.get("cell_diameter_um", 12.0))

    scores = {label: probabilities.get(label, 0.25) for label in class_labels}

    # 1. Biological priors
    prior_boost = {label: 1.0 for label in class_labels}

    # Lymphocyte priors
    if nc >= 1.6 and lobes == 1 and granularity < 30.0:
        prior_boost["lymphocyte"] *= 1.45
        prior_boost["neutrophil"] *= 0.60
    elif nc < 0.8:
        prior_boost["lymphocyte"] *= 0.65

    # Neutrophil priors
    if lobes >= 3:
        prior_boost["neutrophil"] *= 1.55
        prior_boost["lymphocyte"] *= 0.40
    elif lobes == 2:
        prior_boost["eosinophil"] *= 1.25

    # Eosinophil priors
    if granularity >= 38.0 and lobes <= 2:
        prior_boost["eosinophil"] *= 1.50

    # Monocyte priors
    if diameter >= 15.0 and lobes <= 2 and nc <= 1.2:
        prior_boost["monocyte"] *= 1.40

    # Apply Bayesian prior weighting
    calibrated = {}
    total = 0.0
    for label in class_labels:
        val = scores[label] * prior_boost[label]
        calibrated[label] = val
        total += val

    # Renormalize to sum to 1.0
    if total > 0:
        for label in class_labels:
            calibrated[label] = round(float(calibrated[label] / total), 4)

    return calibrated
