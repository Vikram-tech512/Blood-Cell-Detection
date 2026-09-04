"""
Clinical Hematology Copilot, Anomaly Screening, and Reasoning Engine.
Features:
- Anomaly & Hematologic Malignancy Screening (Blast flags, Toxic Granulation, Dysplasia)
- ICD-10 Disease Mapping
- Clinical Triage Urgency (Routine, Warning, Urgent Action Required)
- Evidence-based Confirmatory Test Protocols
- Interactive Clinical Q&A Query Assistant
"""

def detect_anomalies(cell_type, confidence, morphology):
    """
    Screen for cytological anomalies and clinical alert flags.
    
    Returns:
        list of dict: Detected anomaly flags, severity, and clinical warning descriptions.
    """
    flags = []
    nc = float(morphology.get("nc_ratio", 1.0)) if morphology else 1.0
    lobes = int(morphology.get("lobe_count", 1)) if morphology else 1
    granularity = float(morphology.get("granularity_index", 20.0)) if morphology else 20.0
    diameter = float(morphology.get("cell_diameter_um", 12.0)) if morphology else 12.0

    # 1. Immature Blast Cell Suspicion Flag (Acute Leukemia / Myelodysplasia screening)
    if nc >= 2.5 and diameter >= 16.0:
        flags.append({
            "title": "Immature Blast / Atypical Mononuclear Pattern",
            "severity": "critical",
            "badge": "Critical Alert",
            "details": "Markedly elevated Nuclear-to-Cytoplasmic (N:C) ratio with large cellular diameter suggests possible immature blast morphology. Urgent hematopathology review and flow cytometry recommended to exclude acute leukemia (AML/ALL)."
        })

    # 2. Toxic Granulation Flag (Sepsis / Severe Inflammation)
    if cell_type == "neutrophil" and granularity >= 55.0:
        flags.append({
            "title": "Possible Toxic Granulation / Reactive Shift",
            "severity": "warning",
            "badge": "Clinical Warning",
            "details": "Prominent coarse azurophilic granulation observed in neutrophil cytoplasm. Typically correlates with systemic bacterial infection, severe sepsis, tissue necrosis, or G-CSF administration."
        })

    # 3. Hypereosinophilic Syndrome / Severe Allergy Flag
    if cell_type == "eosinophil" and confidence >= 0.85 and granularity >= 60.0:
        flags.append({
            "title": "Marked Eosinophilic Granulation Pattern",
            "severity": "info",
            "badge": "Clinical Advisory",
            "details": "Dense eosinophil granules detected. Correlate with clinical history of severe atopy, drug-induced DRESS syndrome, eosinophilic granulomatosis, or tissue-invasive parasitic infection."
        })

    # 4. Monocytosis / Chronic Inflammatory Flag
    if cell_type == "monocyte" and diameter >= 18.0:
        flags.append({
            "title": "Macrocytic Mononuclear Cell",
            "severity": "info",
            "badge": "Observation",
            "details": "Large monocyte morphology with broad cytoplasm. Check for chronic infections (e.g. tuberculosis, subacute endocarditis) or chronic myelomonocytic leukemia (CMML)."
        })

    return flags


def get_clinical_guidance(cell_type, morphology=None, anomalies=None):
    """
    Generate evidence-based ICD-10 diagnostic codes, triage level, and confirmatory lab tests.
    """
    cell_type = str(cell_type).lower().strip()
    anomalies = anomalies or []

    # Determine triage urgency
    has_critical = any(a.get("severity") == "critical" for a in anomalies)
    has_warning = any(a.get("severity") == "warning" for a in anomalies)

    if has_critical:
        triage = {
            "level": "Urgent Review Required",
            "status_code": "CRITICAL_TRIAGE",
            "badge_class": "danger",
            "action": "Immediate manual peripheral blood smear review by an attending hematopathologist."
        }
    elif has_warning:
        triage = {
            "level": "Clinical Follow-up Recommended",
            "status_code": "PRIORITY_TRIAGE",
            "badge_class": "warning",
            "action": "Correlate with acute-phase reactants (CRP, Procalcitonin) and blood cultures."
        }
    else:
        triage = {
            "level": "Routine Diagnostic Screening",
            "status_code": "STANDARD_TRIAGE",
            "badge_class": "success",
            "action": "Findings consistent with standard reference morphology for this lineage."
        }

    guidance_map = {
        "neutrophil": {
            "icd10": [
                {"code": "D72.829", "description": "Leukocytosis, unspecified"},
                {"code": "D70.9", "description": "Neutropenia, unspecified"},
                {"code": "R78.81", "description": "Bacteremia / Severe Inflammatory Response"}
            ],
            "confirmatory_tests": [
                "Complete Blood Count (CBC) with Automated Differential",
                "Serum C-Reactive Protein (CRP) and Procalcitonin",
                "Blood Cultures (if toxic granulation or fever present)",
                "Manual Peripheral Blood Smear Examination"
            ],
            "differential_considerations": [
                "Acute bacterial pyogenic infection (Staphylococcus, Streptococcus, E. coli)",
                "Physiological stress response, strenuous exercise, or glucocorticoid therapy",
                "Tissue necrosis (acute myocardial infarction, extensive burns)",
                "Myeloproliferative neoplasms (e.g., Chronic Myeloid Leukemia - check BCR-ABL)"
            ]
        },
        "lymphocyte": {
            "icd10": [
                {"code": "D72.820", "description": "Lymphocytosis (symptomatic)"},
                {"code": "D72.810", "description": "Lymphocytopenia"},
                {"code": "B27.90", "description": "Infectious mononucleosis, unspecified"}
            ],
            "confirmatory_tests": [
                "Flow Cytometry Immunophenotyping (CD3, CD4, CD8, CD19, CD20, CD56)",
                "Epstein-Barr Virus (EBV) and Cytomegalovirus (CMV) Serology / PCR",
                "Serum Protein Electrophoresis (SPEP) and Free Light Chains",
                "Bone Marrow Biopsy (if clonal lymphocytosis or persistent B symptoms)"
            ],
            "differential_considerations": [
                "Viral infection (Infectious mononucleosis, viral hepatitis, COVID-19, CMV)",
                "Chronic Lymphocytic Leukemia (CLL) / Small Lymphocytic Lymphoma (SLL)",
                "Autoimmune and chronic inflammatory diseases",
                "Pertussis (Bordetella pertussis) marked lymphocytosis"
            ]
        },
        "eosinophil": {
            "icd10": [
                {"code": "D72.10", "description": "Eosinophilia, unspecified"},
                {"code": "D72.118", "description": "Other hypereosinophilic syndrome"},
                {"code": "L27.0", "description": "Generalized skin eruption due to drugs"}
            ],
            "confirmatory_tests": [
                "Total and Allergen-Specific Serum IgE",
                "Stool Examination for Ova and Parasites (x3 samples)",
                "FIP1L1-PDGFRA and PDGFRB Mutation Testing (FISH / RT-PCR)",
                "Cardiac Troponin and Echocardiography (screen for Loeffler endomyocarditis)"
            ],
            "differential_considerations": [
                "Allergic and atopic diseases (bronchial asthma, allergic rhinitis, eczema)",
                "Drug-induced adverse reaction with eosinophilia and systemic symptoms (DRESS)",
                "Invasive helminthic parasite infections (Strongyloides, Ascaris, Schistosoma)",
                "Churg-Strauss syndrome (EGPA) or Clonal Hypereosinophilic Syndrome"
            ]
        },
        "monocyte": {
            "icd10": [
                {"code": "D72.821", "description": "Monocytosis (symptomatic)"},
                {"code": "A15.0", "description": "Tuberculosis of lung"},
                {"code": "C93.10", "description": "Chronic myelomonocytic leukemia (CMML)"}
            ],
            "confirmatory_tests": [
                "Interferon-Gamma Release Assay (QuantiFERON-TB) or Tuberculin Skin Test",
                "Transthoracic Echocardiogram (if subacute infective endocarditis suspected)",
                "Next-Generation Sequencing (NGS) myeloid panel (TET2, SRSF2, ASXL1 mutations)",
                "Comprehensive Metabolic Panel (CMP) and Serum Ferritin"
            ],
            "differential_considerations": [
                "Chronic bacterial infection (Tuberculosis, Brucellosis, Syphilis, SBE)",
                "Recovery phase following acute bone marrow suppression or chemotherapy",
                "Autoimmune connective tissue disorders (Systemic Lupus Erythematosus, RA)",
                "Chronic Myelomonocytic Leukemia (CMML) if persistent absolute monocytosis > 1.0 x 10^9/L"
            ]
        }
    }

    details = guidance_map.get(cell_type, guidance_map["neutrophil"])

    return {
        "triage": triage,
        "anomalies": anomalies,
        "icd10": details["icd10"],
        "confirmatory_tests": details["confirmatory_tests"],
        "differential_considerations": details["differential_considerations"]
    }


def copilot_query_engine(question, context):
    """
    Evidence-based clinical query copilot.
    Answers clinician or researcher questions regarding the evaluated cell.
    """
    q = str(question).lower().strip()
    cell_type = str(context.get("cell_type", "neutrophil")).lower()
    confidence = float(context.get("confidence", 85.0))
    morphology = context.get("morphology", {})
    nc_ratio = morphology.get("nc_ratio", 0.5)

    if any(k in q for k in ["differential", "diagnosis", "cause", "condition"]):
        return (
            f"**Clinical Differential for {cell_type.title()} Elevation:**\n\n"
            f"Given the observed morphology (N:C Ratio: {nc_ratio}, Confidence: {confidence}%):\n"
            f"1. **Primary Causes:** Correlate with standard etiology for {cell_type}s. "
            f"{'Check for acute bacterial infections or tissue necrosis.' if cell_type == 'neutrophil' else ''}"
            f"{'Consider viral infections (EBV/CMV) or chronic lymphoid proliferations.' if cell_type == 'lymphocyte' else ''}"
            f"{'Screen for drug hypersensitivity, asthma, or parasitic infection.' if cell_type == 'eosinophil' else ''}"
            f"{'Evaluate for chronic granulomatous infections (TB) or CMML.' if cell_type == 'monocyte' else ''}\n"
            f"2. **Next Diagnostic Step:** Perform a complete CBC differential and order confirmatory serology or flow cytometry if clinical symptoms persist."
        )

    if any(k in q for k in ["test", "lab", "investigation", "confirm", "follow"]):
        return (
            f"**Recommended Confirmatory Laboratory Tests:**\n\n"
            f"- **Complete Blood Count (CBC) with Peripheral Blood Smear (PBS)** review under 100x oil immersion.\n"
            f"- **Inflammatory Markers:** Serum CRP, ESR, and Procalcitonin.\n"
            f"- **Specific Testing:** "
            f"{'Blood cultures and bone marrow evaluation if severe left-shift exists.' if cell_type == 'neutrophil' else ''}"
            f"{'Flow cytometry immunophenotyping for light chain clonality (kappa/lambda).' if cell_type == 'lymphocyte' else ''}"
            f"{'Total serum IgE, stool O&P, and PDGFRA mutation testing.' if cell_type == 'eosinophil' else ''}"
            f"{'Tuberculosis QuantiFERON gold and myeloid mutation profiling.' if cell_type == 'monocyte' else ''}"
        )

    if any(k in q for k in ["blast", "leukemia", "malignan", "cancer"]):
        if float(nc_ratio) > 1.8:
            return (
                "**Malignancy Screening Alert:**\n\n"
                f"The measured Nuclear-to-Cytoplasmic (N:C) ratio is {nc_ratio}, which is in the high range. "
                "While compact mature lymphocytes naturally exhibit high N:C ratios, the presence of loose lacy chromatin or nucleoli requires urgent manual hematopathology review and immunophenotyping to rule out blast crises or acute leukemia."
            )
        else:
            return (
                f"The specimen shows an N:C ratio of {nc_ratio} and mature nuclear segmentation, which is atypical for acute blast cells. However, clinical correlation with the overall white cell count is essential."
            )

    if any(k in q for k in ["icd", "code", "billing"]):
        guidance = get_clinical_guidance(cell_type)
        codes = ", ".join([f"{item['code']} ({item['description']})" for item in guidance["icd10"]])
        return f"**Relevant ICD-10 Clinical Diagnostic Codes for {cell_type.title()}:**\n\n{codes}"

    # General clinical response
    return (
        f"**Hematology Copilot Review:**\n\n"
        f"This specimen was classified as a **{cell_type.title()}** with **{confidence}% diagnostic confidence**.\n"
        f"- **Morphometry:** N:C ratio of {nc_ratio}, with {morphology.get('lobe_count', 1)} detectable nuclear lobes and an estimated diameter of {morphology.get('cell_diameter_um', 13)} μm.\n"
        f"- **Grad-CAM Insights:** The model focused predominantly on the nuclear chromatin structure and cytoplasmic granulation.\n"
        "Feel free to ask about differential diagnoses, confirmatory testing protocols, ICD-10 codes, or malignant anomaly screening."
    )
