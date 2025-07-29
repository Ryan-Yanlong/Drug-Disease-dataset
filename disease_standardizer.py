# FILE: disease_standardizer.py
#
# DESCRIPTION:
# This is the final, most advanced version of the script. It integrates a
# curated set of high-quality synonyms confirmed by the user, combining this
# expert review with our existing semantic map and fuzzy matching logic.

import pandas as pd
import re
from thefuzz import process
from tqdm import tqdm

def create_expanded_semantic_map():
    """
    Returns the master dictionary mapping clinical terms to a standard name.
    This version includes all curated mappings from the user-led review.
    """
    mapping = {
        # === USER-CONFIRMED MAPPINGS START ===
        "peptic ulcer disease, dyspepsia [l1308]": "peptic ulcer",
        "peptic ulcer disease and the relief of smooth muscle spasms in gastrointestinal disorder": "peptic ulcer",
        "reflux oesophagitis and peptic ulcer disease": "peptic ulcer",
        "peptic ulcer disease and in acquired nystagmu": "peptic ulcer",
        "tuberculosi": "pulmonary tuberculosi",
        "alcohol dependence in conjunction with a behavioural modification program": "alcoholism",
        "migraine headache": "migraine with or without aura in adult",
        "iron deficiency anemia in adults and children": "used in preventing and treating iron-deficiency anemia",
        "hypertension and congestive heart failure, diabetic nephropathy, and improvement of prognosis for coronary artery diseases (including acute myocardial infarction)": "hypertension and heart failure",
        "pulmonary arterial hypertension (pah) to delay disease progression and reduce risk of hospitalization": "hypertension and heart failure",
        "treatment of anxiety, tension, irritability and similar stress related symptom": "anxiety disorder",
        # === USER-CONFIRMED MAPPINGS END ===
        
        # === Previously curated mappings ===
        'alcohol dependence': 'alcoholism',
        'parkinsonism': "parkinson's disease",
        'erectile dysfunction': 'impotence and vasospasm',
        'sleep disorder': "used mainly for sedation and hypnosi",
        'bipolar disorder': "depression and mania",
        'unresponsive to stimuli': "depressed level of consciousnes",
        'neuropathy peripheral': "neuropathic pain associated with post-herpetic neuralgia",
        'deep vein thrombosi': "deep venous thrombosis (dvt) and pulmonary embolism (pe) in patients who have been treated with a parenteral anticoagulant for 5-10 day",

        # === Base semantic map ===
        'neoplasm': 'cancer', 'benign neoplasm': 'cancer', 'malignant neoplasm': 'cancer',
        'carcinoma': 'cancer', 'tumor': 'cancer', 'leukaemia': 'leukemia',
        'myocardial infarction': 'heart attack', 'cerebrovascular accident': 'stroke',
        'transient ischaemic attack': 'transient ischemic attack',
        'deep vein thrombosis': 'deep vein thrombosis', 'pulmonary embolism': 'pulmonary embolism',
        'blood pressure increased': 'hypertension',
        'blood pressure decreased': 'hypotension',
        'cardiac failure': 'heart failure', 'cardiac failure congestive': 'heart failure',
        'angina pectoris': 'angina', 'arrhythmia': 'arrhythmia',
        'renal failure': 'kidney failure', 'renal impairment': 'kidney failure',
        'diabetes mellitus': 'diabetes', 'type 2 diabetes mellitus': 'diabetes',
        'blood glucose increased': 'hyperglycemia', 'hyperglycaemia': 'hyperglycemia',
        'blood glucose decreased': 'hypoglycemia', 'hypoglycaemia': 'hypoglycemia',
        'suicidal ideation': 'suicidal thoughts', 'major depression': 'depression',
        'generalized anxiety disorder': 'anxiety', 'confusional state': 'confusion',
        'epilepsy': 'seizure', 'convulsion': 'seizure',
        'gastrooesophageal reflux disease': 'acid reflux', 'diarrhoea': 'diarrhea',
        'haemorrhage': 'hemorrhage', 'gastrointestinal haemorrhage': 'gastrointestinal bleeding',
        'pyrexia': 'fever', 'oedema': 'edema', 'dyspnoea': 'shortness of breath',
        'anaemia': 'anemia', 'vertigo': 'dizziness',
    }
    return mapping

def normalize_disease_series(series: pd.Series, mapping: dict) -> pd.Series:
    """Applies a series of normalization rules to a pandas Series."""
    s = series.str.lower().str.strip()
    # Apply the powerful semantic map first
    s = s.replace(mapping)
    # Apply general cleaning rules
    s = s.str.replace(r'\s+\(disease or disorder\)', '', regex=True)
    s = s.str.replace(r's$', '', regex=True) # Basic plural to singular
    s = s.str.strip()
    return s

def create_fuzzy_map(set1, set2, score_cutoff=90):
    """Creates a mapping from terms in set2 to the best match in set1."""
    fuzzy_map = {}
    for term_from in tqdm(set2, desc="Building Fuzzy Map"):
        best_match, score = process.extractOne(term_from, set1)
        if score >= score_cutoff:
            fuzzy_map[term_from] = best_match
    return fuzzy_map

def enhance_disease_mapping(side_effects_df, therapeutic_df):
    """Standardizes disease names using a hybrid semantic and fuzzy matching approach."""
    print("Standardizing disease names with final curated map and fuzzy matching...")
    
    # Get the expanded semantic map
    semantic_map = create_expanded_semantic_map()

    # Step 1: Apply initial normalization using the expanded map
    therapeutic_df['disease_normalized'] = normalize_disease_series(therapeutic_df['disease_name'], semantic_map)
    side_effects_df['disease_normalized'] = normalize_disease_series(side_effects_df['disease_name'], semantic_map)
    
    therapeutic_df.dropna(subset=['disease_normalized'], inplace=True)
    side_effects_df.dropna(subset=['disease_normalized'], inplace=True)
    
    # Step 2: Create unique sets for fuzzy matching on the already improved names
    therapeutic_diseases = set(therapeutic_df['disease_normalized'].unique())
    side_effect_diseases = set(side_effects_df['disease_normalized'].unique())

    # Step 3: Create and apply the fuzzy map for remaining textual variations
    print(f"\nComparing {len(side_effect_diseases)} side effect terms against {len(therapeutic_diseases)} therapeutic terms for final fuzzy match.")
    fuzzy_map = create_fuzzy_map(therapeutic_diseases, side_effect_diseases, score_cutoff=90)
    print(f"Created a fuzzy map with {len(fuzzy_map)} additional synonym pairs.")
    side_effects_df['disease_normalized'] = side_effects_df['disease_normalized'].map(fuzzy_map).fillna(side_effects_df['disease_normalized'])

    # Step 4: Report the final, improved overlap
    final_therapeutic_diseases = set(therapeutic_df['disease_normalized'].unique())
    final_side_effect_diseases = set(side_effects_df['disease_normalized'].unique())
    overlap = final_therapeutic_diseases.intersection(final_side_effect_diseases)
    
    print("\nDisease overlap statistics after FINAL standardization:")
    print(f"  - Overlapping diseases found: {len(overlap)}")
    if len(final_therapeutic_diseases) > 0:
        print(f"  - Overlap Percentage: {(len(overlap) / len(final_therapeutic_diseases)) * 100:.1f}%")

    # Finalize the dataframes for output
    side_effects_df['disease_name'] = side_effects_df['disease_normalized']
    therapeutic_df['disease_name'] = therapeutic_df['disease_normalized']
    
    return side_effects_df.drop(columns=['disease_normalized']), therapeutic_df.drop(columns=['disease_normalized'])