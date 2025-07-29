# FILE: drugbank_parser_enhanced.py
#
# DESCRIPTION:
# Enhanced version that extracts more therapeutic relationships from DrugBank
# by being more aggressive in parsing indication text and adding more sources.

import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import re
from collections import defaultdict

def parse_drugbank_xml(xml_file_path):
    """
    Parse DrugBank XML to extract drug info and therapeutic relationships.
    This enhanced version extracts more therapeutic relationships.
    """
    print("Starting Enhanced DrugBank XML parsing...")
    
    try:
        context = ET.iterparse(xml_file_path, events=('start', 'end'))
        _, root = next(context)
    except ET.ParseError as e:
        print(f"XML parsing failed! Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

    namespace = root.tag.split('}')[0][1:] if '}' in root.tag else ''
    ns_map = {'db': namespace}
    
    drugs_data = []
    therapeutic_data = []
    
    for event, elem in tqdm(context, desc="Parsing DrugBank XML"):
        if event == 'end' and elem.tag == f"{{{namespace}}}drug":
            if elem.attrib.get('type') != 'small molecule':
                elem.clear()
                root.clear()
                continue
            
            db_id_node = elem.find('db:drugbank-id[@primary="true"]', ns_map)
            name_node = elem.find('db:name', ns_map)
            smiles_node = elem.find("db:calculated-properties/db:property[db:kind='SMILES']/db:value", ns_map)

            if db_id_node is None or name_node is None or smiles_node is None or not smiles_node.text:
                elem.clear()
                root.clear()
                continue

            drugbank_id = db_id_node.text
            name = name_node.text
            smiles = smiles_node.text
            
            drugs_data.append({'drug_id': drugbank_id, 'name': name, 'smiles': smiles})
            
            # Extract therapeutic indications from various sources
            indications = _extract_indications_enhanced(elem, ns_map, drugbank_id)
            therapeutic_data.extend(indications)
            
            elem.clear()
            root.clear()
    
    df_drugs = pd.DataFrame(drugs_data)
    df_therapeutic = pd.DataFrame(therapeutic_data)
    
    # Deduplicate therapeutic relationships
    df_therapeutic = df_therapeutic.drop_duplicates(subset=['drug_id', 'disease_name'])
    
    print("\nParsing complete!")
    print(f"Found {len(df_drugs)} small molecule drugs with SMILES.")
    print(f"Found {len(df_therapeutic)} therapeutic relationships.")
    print(f"Unique diseases in therapeutic data: {df_therapeutic['disease_name'].nunique()}")
    
    return df_drugs, df_therapeutic

def _extract_indications_enhanced(elem, ns_map, drugbank_id):
    """Enhanced extraction of therapeutic indications."""
    drug_therapeutic_info = []
    indications_found = set()

    # 1. Primary Indications (highest confidence: 1.0)
    indication_tags = elem.findall('db:indication', ns_map)
    indications_wrapper = elem.find('db:indications', ns_map)
    if indications_wrapper is not None:
        indication_tags.extend(indications_wrapper.findall('db:indication', ns_map))

    for ind in indication_tags:
        if ind.text and ind.text.strip():
            diseases = _text_to_diseases_enhanced(ind.text.strip())
            for disease in diseases:
                if disease.lower() not in indications_found:
                    indications_found.add(disease.lower())
                    drug_therapeutic_info.append({
                        'drug_id': drugbank_id, 'disease_name': disease,
                        'source': 'indication', 'confidence': 1.0
                    })
    
    # 2. Categories (high confidence: 0.9)
    categories_elem = elem.find('db:categories', ns_map)
    if categories_elem is not None:
        for cat in categories_elem.findall('db:category/db:category', ns_map):
            if cat.text:
                diseases = _category_to_diseases_enhanced(cat.text.strip())
                for disease in diseases:
                    if disease.lower() not in indications_found:
                        drug_therapeutic_info.append({
                            'drug_id': drugbank_id, 'disease_name': disease,
                            'source': 'category', 'confidence': 0.9
                        })

    # 3. ATC Codes (high confidence: 0.85)
    atc_codes_elem = elem.find('db:atc-codes', ns_map)
    if atc_codes_elem is not None:
        for level in atc_codes_elem.findall('.//db:level', ns_map):
            if level.text:
                diseases = _atc_to_diseases_enhanced(level.text.strip())
                for disease in diseases:
                    if disease.lower() not in indications_found:
                        drug_therapeutic_info.append({
                            'drug_id': drugbank_id, 'disease_name': disease,
                            'source': 'atc', 'confidence': 0.85
                        })
    
    # 4. Pharmacology description (medium confidence: 0.7)
    pharmacology = elem.find('db:pharmacodynamics', ns_map)
    if pharmacology is not None and pharmacology.text:
        diseases = _text_to_diseases_enhanced(pharmacology.text[:500])  # First 500 chars
        for disease in diseases:
            if disease.lower() not in indications_found:
                drug_therapeutic_info.append({
                    'drug_id': drugbank_id, 'disease_name': disease,
                    'source': 'pharmacology', 'confidence': 0.7
                })
    
    # 5. Mechanism of action (medium confidence: 0.65)
    mechanism = elem.find('db:mechanism-of-action', ns_map)
    if mechanism is not None and mechanism.text:
        diseases = _mechanism_to_diseases(mechanism.text[:300])
        for disease in diseases:
            if disease.lower() not in indications_found:
                drug_therapeutic_info.append({
                    'drug_id': drugbank_id, 'disease_name': disease,
                    'source': 'mechanism', 'confidence': 0.65
                })
                
    return drug_therapeutic_info

def _text_to_diseases_enhanced(text):
    """Enhanced extraction of disease names from free text."""
    diseases = []
    text_lower = text.lower()
    
    # Expanded patterns for disease extraction
    patterns = [
        r'treatment of (.+?)(?:\.|,|;|and|$)',
        r'indicated for (?:the )?(.+?)(?:\.|,|;|and|$)',
        r'management of (.+?)(?:\.|,|;|and|$)',
        r'therapy (?:of|for) (.+?)(?:\.|,|;|and|$)',
        r'used (?:in|for) (?:the )?(.+?)(?:\.|,|;|and|$)',
        r'relief of (.+?)(?:\.|,|;|and|$)',
        r'prophylaxis of (.+?)(?:\.|,|;|and|$)',
        r'prevention of (.+?)(?:\.|,|;|and|$)',
        r'effective against (.+?)(?:\.|,|;|and|$)',
        r'treats (.+?)(?:\.|,|;|and|$)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Split on common conjunctions
            sub_diseases = re.split(r'\s+(?:and|or)\s+', match)
            for sub_disease in sub_diseases:
                # Clean up the disease name
                disease = re.sub(r'^(the|acute|chronic|severe|mild|moderate)\s+', '', sub_disease).strip()
                disease = re.sub(r'\s+in\s+(?:adults|children|patients).*$', '', disease)
                disease = re.sub(r'\s+\(.*?\)', '', disease)  # Remove parenthetical info
                disease = disease.rstrip('.,;')
                
                if disease and len(disease) > 3 and not any(skip in disease for skip in ['patient', 'dose', 'mg', 'ml']):
                    diseases.append(disease)
    
    # Also look for specific disease mentions
    disease_keywords = [
        'hypertension', 'diabetes', 'cancer', 'infection', 'inflammation',
        'pain', 'fever', 'anxiety', 'depression', 'epilepsy', 'asthma',
        'arthritis', 'ulcer', 'migraine', 'insomnia', 'obesity', 'allergy',
        'pneumonia', 'bronchitis', 'hepatitis', 'gastritis', 'dermatitis',
        'psoriasis', 'eczema', 'acne', 'glaucoma', 'cataract',
        'osteoporosis', 'anemia', 'thrombosis', 'embolism', 'stroke',
        'angina', 'arrhythmia', 'heart failure', 'kidney disease',
        'liver disease', 'thyroid disorder', 'parkinson', 'alzheimer',
        'schizophrenia', 'bipolar disorder', 'adhd', 'autism'
    ]
    
    for keyword in disease_keywords:
        if keyword in text_lower and keyword not in [d.lower() for d in diseases]:
            diseases.append(keyword)
    
    return list(set(d for d in diseases if len(d) > 3))[:5]  # Max 5 per text

def _category_to_diseases_enhanced(category):
    """Enhanced conversion of drug categories to disease names."""
    category_lower = category.lower()
    diseases = []
    
    # Expanded category mappings
    category_map = {
        'antihypertensive': ['hypertension', 'high blood pressure'],
        'antidepressive': ['depression', 'major depressive disorder'],
        'anti-anxiety': ['anxiety', 'anxiety disorder'],
        'anxiolytic': ['anxiety', 'anxiety disorder'],
        'anti-bacterial': ['bacterial infection', 'infection'],
        'antibiotic': ['bacterial infection', 'infection'],
        'antineoplastic': ['cancer', 'neoplasm', 'tumor'],
        'antidiabetic': ['diabetes', 'diabetes mellitus'],
        'anti-inflammatory': ['inflammation', 'inflammatory disease'],
        'analgesic': ['pain', 'chronic pain'],
        'antipyretic': ['fever', 'pyrexia'],
        'anticonvulsant': ['epilepsy', 'seizure'],
        'antiepileptic': ['epilepsy', 'seizure disorder'],
        'antipsychotic': ['schizophrenia', 'psychosis'],
        'antihistamine': ['allergy', 'allergic reaction'],
        'antiviral': ['viral infection', 'virus infection'],
        'antifungal': ['fungal infection', 'mycosis'],
        'antiemetic': ['nausea', 'vomiting'],
        'antacid': ['acid reflux', 'heartburn', 'peptic ulcer'],
        'bronchodilator': ['asthma', 'copd', 'bronchospasm'],
        'diuretic': ['hypertension', 'edema', 'fluid retention'],
        'immunosuppressant': ['autoimmune disease', 'organ rejection'],
        'anticoagulant': ['thrombosis', 'blood clot', 'embolism'],
        'antiarrhythmic': ['arrhythmia', 'irregular heartbeat'],
        'antianginal': ['angina', 'chest pain'],
        'antispasmodic': ['muscle spasm', 'cramp'],
        'antimigraine': ['migraine', 'headache'],
        'hypnotic': ['insomnia', 'sleep disorder'],
        'sedative': ['anxiety', 'insomnia', 'agitation'],
        'muscle relaxant': ['muscle spasm', 'muscle pain'],
        'antiparkinson': ['parkinson disease', 'parkinsonism'],
        'antithyroid': ['hyperthyroidism', 'thyroid disorder'],
        'antiulcer': ['peptic ulcer', 'gastric ulcer'],
        'laxative': ['constipation', 'bowel disorder'],
        'antidiarrheal': ['diarrhea', 'bowel disorder'],
        'antiglaucoma': ['glaucoma', 'ocular hypertension'],
        'antiobesity': ['obesity', 'weight gain'],
        'antilipemic': ['hyperlipidemia', 'high cholesterol'],
        'vasodilator': ['hypertension', 'peripheral vascular disease'],
    }
    
    for key, values in category_map.items():
        if key in category_lower:
            diseases.extend(values)
    
    return list(set(diseases))

def _atc_to_diseases_enhanced(atc_text):
    """Enhanced conversion of ATC descriptions to diseases."""
    atc_lower = atc_text.lower()
    diseases = []
    
    # Expanded ATC mappings
    atc_map = {
        'antihypertensives': ['hypertension', 'high blood pressure'],
        'antidepressants': ['depression', 'depressive disorder'],
        'antineoplastic': ['cancer', 'neoplasm'],
        'drugs used in diabetes': ['diabetes', 'diabetes mellitus'],
        'cardiac therapy': ['heart disease', 'cardiac disorder'],
        'analgesics': ['pain', 'chronic pain'],
        'psycholeptics': ['anxiety', 'psychosis', 'insomnia'],
        'psychoanaleptics': ['depression', 'adhd', 'dementia'],
        'antiepileptics': ['epilepsy', 'seizure disorder'],
        'anti-inflammatory': ['inflammation', 'arthritis'],
        'antibacterials': ['bacterial infection', 'infection'],
        'antivirals': ['viral infection', 'virus infection'],
        'antimycotics': ['fungal infection', 'mycosis'],
        'antiprotozoals': ['protozoal infection', 'parasitic infection'],
        'immunosuppressants': ['autoimmune disease', 'transplant rejection'],
        'antithrombotic': ['thrombosis', 'embolism'],
        'diuretics': ['hypertension', 'edema'],
        'beta blocking': ['hypertension', 'arrhythmia', 'angina'],
        'calcium channel': ['hypertension', 'angina'],
        'lipid modifying': ['hyperlipidemia', 'high cholesterol'],
        'antacids': ['acid reflux', 'peptic ulcer'],
        'antiemetics': ['nausea', 'vomiting'],
        'laxatives': ['constipation'],
        'antidiarrheals': ['diarrhea'],
        'antihistamines': ['allergy', 'allergic rhinitis'],
        'corticosteroids': ['inflammation', 'autoimmune disease', 'asthma'],
        'sex hormones': ['hormone deficiency', 'menopause', 'contraception'],
        'thyroid': ['thyroid disorder', 'hypothyroidism', 'hyperthyroidism'],
        'antidiabetics': ['diabetes', 'hyperglycemia'],
        'ophthalmological': ['glaucoma', 'eye infection', 'dry eye'],
        'dermatological': ['skin disease', 'dermatitis', 'psoriasis'],
    }
    
    for key, values in atc_map.items():
        if key in atc_lower:
            diseases.extend(values)
    
    return list(set(diseases))

def _mechanism_to_diseases(text):
    """Extract diseases from mechanism of action text."""
    diseases = []
    text_lower = text.lower()
    
    # Look for receptor/target mentions that imply diseases
    mechanism_map = {
        'dopamine': ['parkinson disease', 'schizophrenia'],
        'serotonin': ['depression', 'anxiety'],
        'norepinephrine': ['depression', 'adhd'],
        'gaba': ['epilepsy', 'anxiety'],
        'histamine': ['allergy', 'allergic reaction'],
        'prostaglandin': ['inflammation', 'pain', 'fever'],
        'insulin': ['diabetes'],
        'glucagon': ['diabetes', 'hypoglycemia'],
        'thyroid': ['thyroid disorder'],
        'estrogen': ['menopause', 'osteoporosis'],
        'testosterone': ['hypogonadism'],
        'acetylcholine': ['alzheimer disease', 'myasthenia gravis'],
        'angiotensin': ['hypertension', 'heart failure'],
        'beta-adrenergic': ['hypertension', 'heart failure', 'asthma'],
        'calcium channel': ['hypertension', 'angina'],
        'sodium channel': ['epilepsy', 'arrhythmia'],
        'potassium channel': ['arrhythmia', 'epilepsy'],
        'opioid': ['pain'],
        'cannabinoid': ['pain', 'nausea'],
        'benzodiazepine': ['anxiety', 'insomnia', 'seizure'],
    }
    
    for key, values in mechanism_map.items():
        if key in text_lower:
            diseases.extend(values)
    
    return list(set(diseases))

if __name__ == '__main__':
    import os
    print("Running enhanced drugbank_parser as a standalone script...")
    if os.path.exists('drugbank.xml'):
        df_drugs, df_therapeutic = parse_drugbank_xml('drugbank.xml')
        
        # Save results
        df_drugs.to_csv('drugbank_drugs_enhanced.csv', index=False)
        df_therapeutic.to_csv('drugbank_therapeutic_relations_enhanced.csv', index=False)
        
        print("\nStand-alone run complete. Files saved:")
        print("  - drugbank_drugs_enhanced.csv")
        print("  - drugbank_therapeutic_relations_enhanced.csv")
    else:
        print("\n'drugbank.xml' not found. Skipping stand-alone execution.")
