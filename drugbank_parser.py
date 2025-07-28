"""
DrugBank XML Parser
Extracts drug information and therapeutic relationships from DrugBank XML files
"""

import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import re
from collections import defaultdict
import logging
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DrugBankParser:
    """Parser for DrugBank XML files to extract drug and therapeutic information"""
    
    def __init__(self):
        self.namespace = ''
        self.ns_map = {}
        self.stats = defaultdict(int)
        
    def parse_drugbank_xml(self, xml_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse DrugBank XML file and extract drug information and therapeutic relationships
        
        Args:
            xml_file_path: Path to DrugBank XML file
            
        Returns:
            Tuple of (drugs_df, therapeutic_df)
        """
        logger.info(f"Starting to parse DrugBank XML file: {xml_file_path}")
        
        try:
            context = ET.iterparse(xml_file_path, events=('start', 'end'))
            _, root = next(context)
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML file: {e}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Extract namespace
        if '}' in root.tag:
            self.namespace = root.tag.split('}')[0][1:]
        self.ns_map = {'db': self.namespace}
        
        drugs_data = []
        therapeutic_data = []
        drugs_without_info = []
        
        # Parse drug entries
        for event, elem in tqdm(context, desc="Parsing DrugBank"):
            if event == 'end' and elem.tag == f"{{{self.namespace}}}drug":
                drug_info = self._extract_drug_info(elem)
                if drug_info:
                    drugs_data.append(drug_info['drug'])
                    therapeutic_data.extend(drug_info['therapeutic'])
                    if not drug_info['therapeutic']:
                        drugs_without_info.append((drug_info['drug']['drug_id'], 
                                                 drug_info['drug']['name']))
                
                # Clear memory
                elem.clear()
                root.clear()
        
        # Create dataframes
        df_drugs = pd.DataFrame(drugs_data)
        df_therapeutic = pd.DataFrame(therapeutic_data)
        
        # Log statistics
        self._log_statistics(df_drugs, df_therapeutic, drugs_without_info)
        
        return df_drugs, df_therapeutic
    
    def _extract_drug_info(self, elem) -> Dict:
        """Extract drug information from XML element"""
        # Only process small molecule drugs
        if elem.attrib.get('type') != 'small molecule':
            return None
        
        # Extract basic info
        db_id_node = elem.find('db:drugbank-id[@primary="true"]', self.ns_map)
        name_node = elem.find('db:name', self.ns_map)
        
        if db_id_node is None or name_node is None:
            return None
        
        drugbank_id = db_id_node.text
        name = name_node.text
        
        # Extract SMILES
        smiles_node = elem.find("db:calculated-properties/db:property[db:kind='SMILES']/db:value", 
                               self.ns_map)
        smiles = smiles_node.text if smiles_node is not None else None
        
        if not smiles:
            return None
        
        # Extract description
        descr_node = elem.find('db:description', self.ns_map)
        description = (
            ET.tostring(descr_node, method='text', encoding='unicode').strip()
            if descr_node is not None else ""
        )
        
        drug_data = {
            'drug_id': drugbank_id,
            'name': name,
            'smiles': smiles,
            'description': description
        }
        
        # Extract therapeutic information
        therapeutic_data = self._extract_therapeutic_info(elem, drugbank_id)
        
        return {
            'drug': drug_data,
            'therapeutic': therapeutic_data
        }
    
    def _extract_therapeutic_info(self, elem, drug_id: str) -> List[Dict]:
        """Extract all therapeutic information for a drug"""
        therapeutic_info = []
        indications_found = set()
        
        # 1. Extract from indications (highest priority)
        therapeutic_info.extend(self._extract_indications(elem, drug_id, indications_found))
        
        # 2. Extract from categories
        therapeutic_info.extend(self._extract_from_categories(elem, drug_id, indications_found))
        
        # 3. Extract from ATC codes
        therapeutic_info.extend(self._extract_from_atc(elem, drug_id, indications_found))
        
        # 4. Extract from description if no other info found
        if not therapeutic_info:
            descr_node = elem.find('db:description', self.ns_map)
            if descr_node is not None:
                description = ET.tostring(descr_node, method='text', encoding='unicode').strip()
                therapeutic_info.extend(self._extract_from_description(description, drug_id))
        
        # 5. Add generic classification for approved drugs if still no info
        if not therapeutic_info:
            therapeutic_info.extend(self._extract_from_groups(elem, drug_id))
        
        return therapeutic_info
    
    def _extract_indications(self, elem, drug_id: str, indications_found: set) -> List[Dict]:
        """Extract therapeutic relationships from indications"""
        therapeutic_info = []
        
        # Method 1: db:indications/db:indication
        indications_wrapper = elem.find('db:indications', self.ns_map)
        if indications_wrapper is not None:
            for ind in indications_wrapper.findall('db:indication', self.ns_map):
                if ind.text and ind.text.strip():
                    diseases = self._extract_diseases_from_indication(ind.text.strip())
                    for disease in diseases:
                        indications_found.add(disease.lower())
                        therapeutic_info.append({
                            'drug_id': drug_id,
                            'disease_name': disease,
                            'source': 'indication',
                            'confidence': 1.0
                        })
                        self.stats['indication'] += 1
        
        # Method 2: Direct db:indication elements
        for ind in elem.findall('db:indication', self.ns_map):
            if ind.text and ind.text.strip():
                diseases = self._extract_diseases_from_indication(ind.text.strip())
                for disease in diseases:
                    if disease.lower() not in indications_found:
                        indications_found.add(disease.lower())
                        therapeutic_info.append({
                            'drug_id': drug_id,
                            'disease_name': disease,
                            'source': 'indication',
                            'confidence': 1.0
                        })
                        self.stats['indication'] += 1
        
        return therapeutic_info
    
    def _extract_from_categories(self, elem, drug_id: str, indications_found: set) -> List[Dict]:
        """Extract therapeutic relationships from drug categories"""
        therapeutic_info = []
        categories_elem = elem.find('db:categories', self.ns_map)
        
        if categories_elem is not None:
            for cat in categories_elem.findall('db:category', self.ns_map):
                cat_name = cat.find('db:category', self.ns_map)
                if cat_name is not None and cat_name.text:
                    diseases = self._convert_category_to_diseases(cat_name.text.strip())
                    for disease in diseases:
                        if disease and disease.lower() not in indications_found:
                            therapeutic_info.append({
                                'drug_id': drug_id,
                                'disease_name': disease,
                                'source': 'category',
                                'confidence': 0.8
                            })
                            self.stats['category'] += 1
        
        return therapeutic_info
    
    def _extract_from_atc(self, elem, drug_id: str, indications_found: set) -> List[Dict]:
        """Extract therapeutic relationships from ATC codes"""
        therapeutic_info = []
        atc_codes_elem = elem.find('db:atc-codes', self.ns_map)
        
        if atc_codes_elem is not None:
            for atc in atc_codes_elem.findall('db:atc-code', self.ns_map):
                for level in atc:
                    level_tag = level.tag.replace(f"{{{self.namespace}}}", "")
                    if level.text and 'level' in level_tag:
                        diseases = self._convert_atc_to_diseases(level.text.strip())
                        for disease in diseases:
                            if disease and disease.lower() not in indications_found:
                                therapeutic_info.append({
                                    'drug_id': drug_id,
                                    'disease_name': disease,
                                    'source': 'atc',
                                    'confidence': 0.7
                                })
                                self.stats['atc'] += 1
        
        return therapeutic_info
    
    def _extract_from_description(self, description: str, drug_id: str) -> List[Dict]:
        """Extract therapeutic relationships from drug description"""
        therapeutic_info = []
        diseases = self._extract_diseases_from_text(description[:500])  # Only check first 500 chars
        
        for disease in diseases:
            therapeutic_info.append({
                'drug_id': drug_id,
                'disease_name': disease,
                'source': 'description',
                'confidence': 0.5
            })
            self.stats['description'] += 1
        
        return therapeutic_info
    
    def _extract_from_groups(self, elem, drug_id: str) -> List[Dict]:
        """Extract generic classification for approved drugs"""
        therapeutic_info = []
        groups_elem = elem.find('db:groups', self.ns_map)
        
        if groups_elem is not None:
            for group in groups_elem.findall('db:group', self.ns_map):
                if group.text and group.text.strip() == 'approved':
                    therapeutic_info.append({
                        'drug_id': drug_id,
                        'disease_name': 'general therapeutic use',
                        'source': 'group',
                        'confidence': 0.3
                    })
                    self.stats['group'] += 1
                    break
        
        return therapeutic_info
    
    def _extract_diseases_from_indication(self, indication_text: str) -> List[str]:
        """Extract specific disease names from indication text"""
        diseases = []
        text_lower = indication_text.lower()
        
        # Common indication patterns
        patterns = [
            r'treatment of (.+?)(?:\.|,|;|$)',
            r'indicated for (.+?)(?:\.|,|;|$)',
            r'used for (.+?)(?:\.|,|;|$)',
            r'management of (.+?)(?:\.|,|;|$)',
            r'therapy for (.+?)(?:\.|,|;|$)',
            r'prevention of (.+?)(?:\.|,|;|$)',
            r'prophylaxis of (.+?)(?:\.|,|;|$)',
            r'relief of (.+?)(?:\.|,|;|$)',
        ]
        
        # Extract diseases using patterns
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Clean and split multiple diseases
                diseases_in_match = re.split(r' and | or |, ', match)
                for disease in diseases_in_match:
                    disease = disease.strip()
                    # Remove common modifiers
                    disease = re.sub(r'^(the |acute |chronic |severe |mild |moderate )', '', disease)
                    disease = re.sub(r'( in patients| in adults| in children).*$', '', disease)
                    if len(disease) > 3 and disease not in ['use', 'users', 'patient', 'patients']:
                        diseases.append(disease)
        
        # If no patterns matched, return simplified full text
        if not diseases and len(indication_text) < 100:
            simplified = re.sub(r'^(For |Used for |Indicated for |Treatment of )', '', indication_text)
            simplified = simplified.strip().rstrip('.')
            if len(simplified) > 3:
                diseases.append(simplified)
        
        return diseases[:3]  # Return at most 3 diseases
    
    def _convert_category_to_diseases(self, category: str) -> List[str]:
        """Convert drug category to disease list"""
        # Category to disease mapping
        category_mapping = {
            # Anti-infective agents
            'Anti-Bacterial Agents': ['bacterial infections'],
            'Antibiotics': ['bacterial infections'],
            'Antiviral Agents': ['viral infections'],
            'Antifungal Agents': ['fungal infections'],
            'Antiparasitic Products': ['parasitic infections'],
            'Anti-Infective Agents': ['infections'],
            'Antimalarials': ['malaria'],
            'Antitubercular Agents': ['tuberculosis'],
            
            # Antineoplastic agents
            'Antineoplastic Agents': ['cancer'],
            'Antimetabolites': ['cancer'],
            'Antineoplastic and Immunomodulating Agents': ['cancer', 'immune disorders'],
            
            # Cardiovascular agents
            'Antihypertensive Agents': ['hypertension'],
            'Antiarrhythmic Agents': ['arrhythmia'],
            'Antianginal Agents': ['angina'],
            'Cardiovascular Agents': ['cardiovascular diseases'],
            'Vasodilator Agents': ['vascular disorders'],
            'Platelet Aggregation Inhibitors': ['thrombosis'],
            'Anticoagulants': ['blood clots', 'thrombosis'],
            'Diuretics': ['fluid retention', 'hypertension'],
            'Beta Blocking Agents': ['hypertension', 'heart diseases'],
            'Calcium Channel Blockers': ['hypertension', 'angina'],
            'ACE Inhibitors': ['hypertension', 'heart failure'],
            
            # Neurological/Psychiatric agents
            'Antidepressive Agents': ['depression'],
            'Antipsychotic Agents': ['psychosis', 'schizophrenia'],
            'Anti-Anxiety Agents': ['anxiety'],
            'Anxiolytics': ['anxiety'],
            'Hypnotics and Sedatives': ['insomnia'],
            'Anticonvulsants': ['epilepsy', 'seizures'],
            'Antiparkinsonian Agents': ['parkinson disease'],
            'Analgesics': ['pain'],
            'Antimigraine Agents': ['migraine'],
            
            # Metabolic/Endocrine agents
            'Antidiabetic Agents': ['diabetes'],
            'Hypoglycemic Agents': ['diabetes'],
            'Thyroid Agents': ['thyroid disorders'],
            'Lipid Regulating Agents': ['dyslipidemia', 'high cholesterol'],
            'Anti-Obesity Agents': ['obesity'],
            
            # Gastrointestinal agents
            'Gastrointestinal Agents': ['gastrointestinal disorders'],
            'Antiemetics': ['nausea', 'vomiting'],
            'Antacids': ['acid reflux', 'heartburn'],
            'Proton Pump Inhibitors': ['acid reflux', 'peptic ulcers'],
            'Antiulcer Agents': ['ulcers'],
            'Laxatives': ['constipation'],
            'Antidiarrheals': ['diarrhea'],
            
            # Respiratory agents
            'Respiratory System Agents': ['respiratory diseases'],
            'Bronchodilator Agents': ['asthma', 'COPD'],
            'Antitussive Agents': ['cough'],
            'Antihistamines': ['allergies', 'allergic rhinitis'],
            
            # Anti-inflammatory/Immune agents
            'Anti-Inflammatory Agents': ['inflammation'],
            'Immunosuppressive Agents': ['autoimmune diseases', 'organ rejection'],
            'Antirheumatic Agents': ['rheumatoid arthritis'],
            
            # Other
            'Dermatologic Agents': ['skin conditions'],
            'Hematologic Agents': ['blood disorders'],
            'Antianemic Agents': ['anemia'],
        }
        
        # Try exact match first
        if category in category_mapping:
            return category_mapping[category]
        
        # Try partial match
        category_lower = category.lower()
        diseases = []
        
        for key, value in category_mapping.items():
            key_lower = key.lower()
            if key_lower in category_lower or category_lower in key_lower:
                diseases.extend(value)
        
        # If no match found, try generic rules
        if not diseases:
            if 'agents' in category_lower and 'anti' in category_lower:
                match = re.search(r'anti[- ]?(\w+)', category_lower)
                if match:
                    target = match.group(1)
                    if target not in ['agents', 'drug', 'drugs']:
                        diseases.append(f"{target}")
        
        return diseases
    
    def _convert_atc_to_diseases(self, atc_text: str) -> List[str]:
        """Convert ATC classification to disease list"""
        atc_text_lower = atc_text.lower()
        
        # ATC to disease mapping
        atc_mapping = {
            # Main systems
            'alimentary tract and metabolism': ['gastrointestinal disorders', 'metabolic disorders'],
            'blood and blood forming organs': ['blood disorders', 'anemia'],
            'cardiovascular system': ['cardiovascular diseases'],
            'dermatologicals': ['skin diseases'],
            'genito urinary system and sex hormones': ['genitourinary disorders', 'hormonal disorders'],
            'systemic hormonal preparations': ['hormonal disorders'],
            'antiinfectives for systemic use': ['infections'],
            'antineoplastic and immunomodulating agents': ['cancer', 'immune disorders'],
            'musculo-skeletal system': ['musculoskeletal disorders'],
            'nervous system': ['neurological disorders'],
            'antiparasitic products': ['parasitic infections'],
            'respiratory system': ['respiratory diseases'],
            'sensory organs': ['eye diseases', 'ear disorders'],
            
            # Specific subcategories
            'antibacterials for systemic use': ['bacterial infections'],
            'antimycotics for systemic use': ['fungal infections'],
            'antivirals for systemic use': ['viral infections'],
            'antidiabetes drugs': ['diabetes'],
            'psycholeptics': ['psychiatric disorders', 'anxiety', 'insomnia'],
            'psychoanaleptics': ['depression', 'ADHD'],
            'analgesics': ['pain'],
            'antiepileptics': ['epilepsy'],
            'anti-parkinson drugs': ['parkinson disease'],
            'cardiac therapy': ['heart diseases'],
            'antihypertensives': ['hypertension'],
            'diuretics': ['fluid retention', 'hypertension'],
            'lipid modifying agents': ['dyslipidemia', 'high cholesterol'],
            'antithrombotic agents': ['thrombosis', 'stroke prevention'],
            'drugs for obstructive airway diseases': ['asthma', 'COPD'],
            'antihistamines for systemic use': ['allergies'],
        }
        
        diseases = []
        
        # Try matching
        for key, value in atc_mapping.items():
            if key in atc_text_lower:
                diseases.extend(value)
        
        # If no match, try pattern-based extraction
        if not diseases:
            if 'therapy' in atc_text_lower:
                match = re.search(r'(\w+)\s+therapy', atc_text_lower)
                if match:
                    organ = match.group(1)
                    if organ not in ['drug', 'drugs']:
                        diseases.append(f"{organ} disorders")
            elif 'drugs for' in atc_text_lower:
                match = re.search(r'drugs for (.+)', atc_text_lower)
                if match:
                    diseases.append(match.group(1).strip())
        
        return list(set(diseases))
    
    def _extract_diseases_from_text(self, text: str) -> List[str]:
        """Extract disease names from free text"""
        diseases = []
        text_lower = text.lower()
        
        # Disease extraction patterns
        patterns = [
            r'treatment of ([^,\.;]+)',
            r'indicated for ([^,\.;]+)',
            r'used for ([^,\.;]+)',
            r'effective against ([^,\.;]+)',
            r'management of ([^,\.;]+)',
            r'therapy for ([^,\.;]+)',
            r'prevention of ([^,\.;]+)',
            r'relief of ([^,\.;]+)',
        ]
        
        # Common disease keywords
        disease_keywords = [
            'infection', 'cancer', 'tumor', 'carcinoma', 'lymphoma', 'leukemia',
            'disease', 'disorder', 'syndrome', 'condition',
            'pain', 'inflammation', 'fever', 'allergy', 'allergies',
            'diabetes', 'hypertension', 'depression', 'anxiety', 'psychosis',
            'epilepsy', 'seizure', 'asthma', 'arthritis', 'osteoporosis',
            'migraine', 'headache', 'insomnia', 'obesity',
            'anemia', 'pneumonia', 'bronchitis', 'hepatitis',
            'thrombosis', 'ulcer', 'nausea', 'vomiting', 'diarrhea',
        ]
        
        # Extract using patterns
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                match = match.strip()
                # Clean up
                match = re.sub(r'^(the |acute |chronic |severe |mild |moderate )', '', match)
                match = re.sub(r'( in patients| in adults| in children).*$', '', match)
                
                # Check if contains disease keyword
                if any(keyword in match for keyword in disease_keywords):
                    if len(match) > 3 and match not in diseases:
                        diseases.append(match)
        
        # Deduplicate and limit
        unique_diseases = []
        seen = set()
        for disease in diseases:
            if disease.lower() not in seen:
                seen.add(disease.lower())
                unique_diseases.append(disease)
        
        return unique_diseases[:3]
    
    def _log_statistics(self, df_drugs: pd.DataFrame, df_therapeutic: pd.DataFrame, 
                       drugs_without_info: List[Tuple[str, str]]):
        """Log parsing statistics"""
        logger.info(f"Parsing completed!")
        logger.info(f"Found {len(df_drugs)} small molecule drugs")
        logger.info(f"Found {len(df_therapeutic)} therapeutic relationships")
        
        logger.info("Therapeutic information sources:")
        for source, count in self.stats.items():
            logger.info(f"  {source}: {count}")
        
        # Statistics on drug coverage
        drugs_with_info = df_therapeutic['drug_id'].nunique()
        coverage = drugs_with_info / len(df_drugs) * 100 if len(df_drugs) > 0 else 0
        logger.info(f"Drugs with therapeutic information: {drugs_with_info} ({coverage:.1f}%)")
        
        # Average relationships per drug
        avg_relations = len(df_therapeutic) / drugs_with_info if drugs_with_info > 0 else 0
        logger.info(f"Average therapeutic relationships per drug: {avg_relations:.2f}")
        
        # Log some drugs without info
        if drugs_without_info:
            logger.info(f"Drugs without therapeutic information: {len(drugs_without_info)}")
            logger.info("Examples (first 5):")
            for drug_id, name in drugs_without_info[:5]:
                logger.info(f"  {drug_id}: {name}")


def main():
    """Main function to run the parser"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse DrugBank XML file')
    parser.add_argument('xml_file', help='Path to DrugBank XML file')
    parser.add_argument('--output-dir', default='.', help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    # Create parser instance
    drugbank_parser = DrugBankParser()
    
    # Parse XML
    df_drugs, df_therapeutic = drugbank_parser.parse_drugbank_xml(args.xml_file)
    
    # Save results
    import os
    drugs_output = os.path.join(args.output_dir, 'drugbank_drugs.csv')
    therapeutic_output = os.path.join(args.output_dir, 'drugbank_therapeutic_relations.csv')
    
    df_drugs.to_csv(drugs_output, index=False)
    df_therapeutic.to_csv(therapeutic_output, index=False)
    
    logger.info(f"Data saved to:")
    logger.info(f"  - {drugs_output}")
    logger.info(f"  - {therapeutic_output}")
    
    # Display some examples
    logger.info("\nTherapeutic relationship examples (first 10):")
    sample = df_therapeutic.head(10)
    for _, row in sample.iterrows():
        logger.info(f"  {row['drug_id']} -> {row['disease_name']} "
                   f"(source: {row['source']}, confidence: {row['confidence']})")


if __name__ == '__main__':
    main()
