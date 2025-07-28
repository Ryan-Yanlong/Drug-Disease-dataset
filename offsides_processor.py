"""
OFFSIDES Side Effects Processor
Processes and standardizes side effect data from the OFFSIDES dataset
"""

import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm
import logging
from typing import Dict, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedDRAMapper:
    """Maps MedDRA terms to common disease names"""
    
    def __init__(self):
        self.mapping = self._create_meddra_mapping()
        self.pattern_rules = self._create_pattern_rules()
    
    def _create_meddra_mapping(self) -> Dict[str, str]:
        """Create MedDRA term to common disease name mapping"""
        return {
            # Blood pressure related
            'blood pressure increased': 'hypertension',
            'blood pressure decreased': 'hypotension',
            'blood pressure abnormal': 'blood pressure disorder',
            'hypertension': 'hypertension',
            'hypotension': 'hypotension',
            'orthostatic hypotension': 'orthostatic hypotension',
            
            # Blood glucose related
            'blood glucose increased': 'hyperglycemia',
            'blood glucose decreased': 'hypoglycemia',
            'diabetes mellitus': 'diabetes',
            'type 2 diabetes mellitus': 'diabetes',
            'type 1 diabetes mellitus': 'diabetes',
            'hyperglycaemia': 'hyperglycemia',
            'hypoglycaemia': 'hypoglycemia',
            
            # Cardiac related
            'cardiac failure': 'heart failure',
            'cardiac failure congestive': 'heart failure',
            'myocardial infarction': 'myocardial infarction',
            'angina pectoris': 'angina',
            'arrhythmia': 'arrhythmia',
            'atrial fibrillation': 'atrial fibrillation',
            'tachycardia': 'tachycardia',
            'bradycardia': 'bradycardia',
            'palpitations': 'palpitations',
            
            # Respiratory system
            'dyspnoea': 'dyspnea',
            'dyspnoea exertional': 'dyspnea',
            'asthma': 'asthma',
            'pneumonia': 'pneumonia',
            'cough': 'cough',
            'respiratory distress': 'respiratory distress',
            'bronchitis': 'bronchitis',
            
            # Gastrointestinal
            'nausea': 'nausea',
            'vomiting': 'vomiting',
            'diarrhoea': 'diarrhea',
            'constipation': 'constipation',
            'abdominal pain': 'abdominal pain',
            'abdominal pain upper': 'abdominal pain',
            'dyspepsia': 'dyspepsia',
            'gastrointestinal haemorrhage': 'gastrointestinal bleeding',
            
            # Nervous system
            'headache': 'headache',
            'dizziness': 'dizziness',
            'somnolence': 'somnolence',
            'insomnia': 'insomnia',
            'tremor': 'tremor',
            'convulsion': 'seizures',
            'paraesthesia': 'paresthesia',
            'neuropathy peripheral': 'peripheral neuropathy',
            
            # Psychiatric
            'depression': 'depression',
            'anxiety': 'anxiety',
            'confusional state': 'confusion',
            'hallucination': 'hallucinations',
            'psychotic disorder': 'psychosis',
            'mood swings': 'mood disorders',
            'agitation': 'agitation',
            
            # Skin related
            'rash': 'rash',
            'pruritus': 'pruritus',
            'urticaria': 'urticaria',
            'dermatitis': 'dermatitis',
            'eczema': 'eczema',
            'psoriasis': 'psoriasis',
            'alopecia': 'alopecia',
            'hyperhidrosis': 'hyperhidrosis',
            
            # Blood/Hematologic
            'anaemia': 'anemia',
            'thrombocytopenia': 'thrombocytopenia',
            'neutropenia': 'neutropenia',
            'leukopenia': 'leukopenia',
            'haemorrhage': 'hemorrhage',
            'epistaxis': 'epistaxis',
            
            # Hepatic/Renal
            'hepatic function abnormal': 'liver dysfunction',
            'renal failure': 'renal failure',
            'renal impairment': 'renal impairment',
            'proteinuria': 'proteinuria',
            
            # Musculoskeletal
            'arthralgia': 'joint pain',
            'myalgia': 'muscle pain',
            'back pain': 'back pain',
            'pain in extremity': 'limb pain',
            'muscle spasms': 'muscle spasms',
            'osteoporosis': 'osteoporosis',
            
            # Infections
            'infection': 'infection',
            'urinary tract infection': 'urinary tract infection',
            'nasopharyngitis': 'nasopharyngitis',
            'influenza': 'influenza',
            
            # General
            'fatigue': 'fatigue',
            'asthenia': 'asthenia',
            'pyrexia': 'fever',
            'oedema': 'edema',
            'oedema peripheral': 'peripheral edema',
            'weight increased': 'weight gain',
            'weight decreased': 'weight loss',
            'appetite decreased': 'decreased appetite',
            'malaise': 'malaise',
            'pain': 'pain',
            'fall': 'falls',
            'death': 'death',
            
            # Laboratory abnormalities
            'alanine aminotransferase increased': 'elevated ALT',
            'aspartate aminotransferase increased': 'elevated AST',
            'blood creatinine increased': 'elevated creatinine',
            'white blood cell count decreased': 'leukopenia',
            'platelet count decreased': 'thrombocytopenia',
            'haemoglobin decreased': 'anemia',
        }
    
    def _create_pattern_rules(self) -> list:
        """Create pattern-based transformation rules"""
        return [
            # Increase/decrease patterns
            (r'(.+) increased$', r'\1 elevation'),
            (r'(.+) decreased$', r'\1 deficiency'),
            (r'blood (.+) increased$', r'hyper\1emia'),
            (r'blood (.+) decreased$', r'hypo\1emia'),
            
            # Abnormal patterns
            (r'(.+) abnormal$', r'\1 disorder'),
            (r'(.+) disorder$', r'\1 disorder'),
            
            # Pain patterns
            (r'(.+) pain$', r'\1 pain'),
            (r'pain in (.+)$', r'\1 pain'),
            
            # Pluralization
            (r'(.+)s$', r'\1'),
        ]
    
    def map_term(self, term: str) -> str:
        """Map a MedDRA term to common disease name"""
        term_lower = term.lower().strip()
        
        # First try direct mapping
        if term_lower in self.mapping:
            return self.mapping[term_lower]
        
        # Then try pattern rules
        for pattern, replacement in self.pattern_rules:
            new_term = re.sub(pattern, replacement, term_lower)
            if new_term != term_lower:
                return new_term
        
        # Return original if no mapping found
        return term_lower


class OFFSIDESProcessor:
    """Process OFFSIDES side effects data"""
    
    def __init__(self):
        self.meddra_mapper = MedDRAMapper()
    
    def enhance_side_effects(self, side_effects_df: pd.DataFrame, 
                           therapeutic_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance side effects data with standardized disease names
        
        Args:
            side_effects_df: DataFrame with side effects data
            therapeutic_df: DataFrame with therapeutic relationships
            
        Returns:
            Enhanced side effects DataFrame
        """
        logger.info("Starting side effects enhancement...")
        
        # Create a copy to avoid modifying original
        enhanced_df = side_effects_df.copy()
        
        # Standardize disease names
        logger.info("Standardizing disease names...")
        enhanced_df['disease_name_normalized'] = enhanced_df['disease_name'].str.lower().str.strip()
        
        # Apply MedDRA mapping
        logger.info("Applying MedDRA mapping...")
        mapped_count = 0
        for idx, row in tqdm(enhanced_df.iterrows(), total=len(enhanced_df), 
                           desc="Mapping MedDRA terms"):
            original = row['disease_name_normalized']
            mapped = self.meddra_mapper.map_term(original)
            if mapped != original:
                enhanced_df.at[idx, 'disease_name_normalized'] = mapped
                mapped_count += 1
        
        logger.info(f"MedDRA mapping applied to {mapped_count} terms")
        
        # Calculate overlap with therapeutic diseases
        self._calculate_overlap(enhanced_df, therapeutic_df)
        
        # Replace original disease names with normalized ones
        enhanced_df['disease_name'] = enhanced_df['disease_name_normalized']
        enhanced_df = enhanced_df.drop(columns=['disease_name_normalized'])
        
        return enhanced_df
    
    def _calculate_overlap(self, side_effects_df: pd.DataFrame, 
                          therapeutic_df: pd.DataFrame):
        """Calculate and log overlap between side effects and therapeutic diseases"""
        therapeutic_diseases = set(therapeutic_df['disease_name'].str.lower().str.strip().unique())
        side_effect_diseases = set(side_effects_df['disease_name_normalized'].unique())
        
        overlap = therapeutic_diseases & side_effect_diseases
        
        logger.info("Disease overlap statistics:")
        logger.info(f"  Therapeutic diseases: {len(therapeutic_diseases)}")
        logger.info(f"  Side effect diseases (normalized): {len(side_effect_diseases)}")
        logger.info(f"  Overlapping diseases: {len(overlap)} ({len(overlap)/len(therapeutic_diseases)*100:.1f}%)")
        
        # Log some examples of overlapping diseases
        if overlap:
            logger.info("Examples of overlapping diseases:")
            for disease in list(overlap)[:10]:
                logger.info(f"  - {disease}")


class DatasetCreator:
    """Create final drug repositioning dataset"""
    
    def __init__(self):
        pass
    
    def create_final_dataset(self, drugs_df: pd.DataFrame, therapeutic_df: pd.DataFrame,
                           side_effects_df: pd.DataFrame, output_dir: str = 'drug_repositioning_dataset'):
        """
        Create final drug repositioning dataset
        
        Args:
            drugs_df: DataFrame with drug information
            therapeutic_df: DataFrame with therapeutic relationships
            side_effects_df: DataFrame with side effects
            output_dir: Output directory for dataset files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Creating final dataset...")
        
        # Collect all unique drugs and diseases
        all_drugs = set()
        all_drugs.update(therapeutic_df['drug_id'].unique())
        all_drugs.update(side_effects_df['drug_id'].unique())
        
        all_diseases = set()
        all_diseases.update(therapeutic_df['disease_name'].str.lower().unique())
        all_diseases.update(side_effects_df['disease_name'].str.lower().unique())
        
        # Only keep drugs that exist in drugs_df
        valid_drugs = all_drugs & set(drugs_df['drug_id'])
        
        logger.info(f"Dataset scale:")
        logger.info(f"  Total drugs: {len(valid_drugs)}")
        logger.info(f"  Total diseases: {len(all_diseases)}")
        logger.info(f"  Potential relationships: {len(valid_drugs) * len(all_diseases):,}")
        
        # Create final drug list
        final_drugs = drugs_df[drugs_df['drug_id'].isin(valid_drugs)].copy()
        
        # Create disease list with IDs
        final_diseases = pd.DataFrame({
            'disease_id': range(len(all_diseases)),
            'disease_name': sorted(list(all_diseases))
        })
        
        # Create disease name to ID mapping
        disease_to_id = dict(zip(final_diseases['disease_name'], final_diseases['disease_id']))
        
        # Create interaction matrix
        logger.info("Creating interaction matrix...")
        interactions = []
        
        # Add therapeutic relationships (label = 1)
        treat_count = 0
        for _, row in therapeutic_df.iterrows():
            if row['drug_id'] in valid_drugs:
                disease_lower = row['disease_name'].lower()
                if disease_lower in disease_to_id:
                    interactions.append({
                        'drug_id': row['drug_id'],
                        'disease_id': disease_to_id[disease_lower],
                        'label': 1,
                        'confidence': row.get('confidence', 1.0),
                        'source': 'therapeutic'
                    })
                    treat_count += 1
        
        # Add side effect relationships (label = -1)
        se_count = 0
        existing_pairs = set((i['drug_id'], i['disease_id']) for i in interactions)
        
        for _, row in side_effects_df.iterrows():
            if row['drug_id'] in valid_drugs:
                disease_lower = row['disease_name'].lower()
                if disease_lower in disease_to_id:
                    pair = (row['drug_id'], disease_to_id[disease_lower])
                    if pair not in existing_pairs:
                        interactions.append({
                            'drug_id': row['drug_id'],
                            'disease_id': disease_to_id[disease_lower],
                            'label': -1,
                            'confidence': 0.9,
                            'source': 'side_effect'
                        })
                        se_count += 1
        
        df_interactions = pd.DataFrame(interactions)
        
        # Remove conflicts (keep therapeutic if both exist)
        logger.info("Handling conflicts...")
        df_interactions = df_interactions.sort_values(['drug_id', 'disease_id', 'label'], 
                                                    ascending=[True, True, False])
        df_interactions = df_interactions.drop_duplicates(subset=['drug_id', 'disease_id'], 
                                                        keep='first')
        
        # Calculate statistics
        self._log_statistics(final_drugs, final_diseases, df_interactions)
        
        # Save files
        logger.info(f"Saving files to {output_dir}/...")
        final_drugs.to_csv(os.path.join(output_dir, 'drugs.csv'), index=False)
        final_diseases.to_csv(os.path.join(output_dir, 'diseases.csv'), index=False)
        df_interactions.to_csv(os.path.join(output_dir, 'drug_disease_interactions.csv'), index=False)
        
        # Save sparse format
        sparse_df = df_interactions[['drug_id', 'disease_id', 'label', 'confidence']].copy()
        sparse_df.to_csv(os.path.join(output_dir, 'interactions_sparse.csv'), index=False)
        
        # Save metadata
        import json
        metadata = {
            'n_drugs': len(final_drugs),
            'n_diseases': len(final_diseases),
            'n_interactions': len(df_interactions),
            'n_therapeutic': len(df_interactions[df_interactions['label'] == 1]),
            'n_side_effects': len(df_interactions[df_interactions['label'] == -1]),
            'sparsity': 1 - len(df_interactions) / (len(final_drugs) * len(final_diseases))
        }
        
        with open(os.path.join(output_dir, 'dataset_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return final_drugs, final_diseases, df_interactions
    
    def _log_statistics(self, drugs_df: pd.DataFrame, diseases_df: pd.DataFrame, 
                       interactions_df: pd.DataFrame):
        """Log dataset statistics"""
        logger.info("Final dataset statistics:")
        logger.info(f"  Drugs: {len(drugs_df)}")
        logger.info(f"  Diseases: {len(diseases_df)}")
        logger.info(f"  Total relationships: {len(interactions_df)}")
        logger.info(f"    - Therapeutic (label=1): {len(interactions_df[interactions_df['label']==1])}")
        logger.info(f"    - Side effects (label=-1): {len(interactions_df[interactions_df['label']==-1])}")
        
        # Calculate sparsity
        total_possible = len(drugs_df) * len(diseases_df)
        sparsity = 1 - len(interactions_df) / total_possible
        logger.info(f"  Matrix sparsity: {sparsity:.2%}")
        
        # Drug coverage
        drugs_with_treatment = interactions_df[interactions_df['label']==1]['drug_id'].nunique()
        drugs_with_sideeffect = interactions_df[interactions_df['label']==-1]['drug_id'].nunique()
        drugs_with_both = len(
            set(interactions_df[interactions_df['label']==1]['drug_id']) & 
            set(interactions_df[interactions_df['label']==-1]['drug_id'])
        )
        
        logger.info("Drug coverage:")
        logger.info(f"  With therapeutic info: {drugs_with_treatment} ({drugs_with_treatment/len(drugs_df)*100:.1f}%)")
        logger.info(f"  With side effect info: {drugs_with_sideeffect} ({drugs_with_sideeffect/len(drugs_df)*100:.1f}%)")
        logger.info(f"  With both: {drugs_with_both} ({drugs_with_both/len(drugs_df)*100:.1f}%)")
        
        # Disease coverage
        diseases_as_treatment = interactions_df[interactions_df['label']==1]['disease_id'].nunique()
        diseases_as_sideeffect = interactions_df[interactions_df['label']==-1]['disease_id'].nunique()
        diseases_both = len(
            set(interactions_df[interactions_df['label']==1]['disease_id']) & 
            set(interactions_df[interactions_df['label']==-1]['disease_id'])
        )
        
        logger.info("Disease coverage:")
        logger.info(f"  As therapeutic target: {diseases_as_treatment} ({diseases_as_treatment/len(diseases_df)*100:.1f}%)")
        logger.info(f"  As side effect: {diseases_as_sideeffect} ({diseases_as_sideeffect/len(diseases_df)*100:.1f}%)")
        logger.info(f"  As both: {diseases_both} ({diseases_both/len(diseases_df)*100:.1f}%)")


def main():
    """Main function to process OFFSIDES data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process OFFSIDES side effects data')
    parser.add_argument('--drugs-file', required=True, help='Path to drugs CSV file')
    parser.add_argument('--therapeutic-file', required=True, help='Path to therapeutic relations CSV file')
    parser.add_argument('--side-effects-file', required=True, help='Path to side effects CSV file')
    parser.add_argument('--output-dir', default='drug_repositioning_dataset', 
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # Read data
    logger.info("Reading input files...")
    drugs_df = pd.read_csv(args.drugs_file)
    therapeutic_df = pd.read_csv(args.therapeutic_file)
    side_effects_df = pd.read_csv(args.side_effects_file)
    
    # Process side effects
    processor = OFFSIDESProcessor()
    enhanced_side_effects = processor.enhance_side_effects(side_effects_df, therapeutic_df)
    
    # Save enhanced side effects
    enhanced_output = args.side_effects_file.replace('.csv', '_enhanced.csv')
    enhanced_side_effects.to_csv(enhanced_output, index=False)
    logger.info(f"Enhanced side effects saved to: {enhanced_output}")
    
    # Create final dataset
    creator = DatasetCreator()
    creator.create_final_dataset(drugs_df, therapeutic_df, enhanced_side_effects, args.output_dir)


if __name__ == '__main__':
    main()
