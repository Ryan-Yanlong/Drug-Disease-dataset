"""
Dataset Optimizer
Creates balanced drug-disease interaction matrices for drug repositioning research
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import os
import json
import time
import random
import logging
from typing import Dict, List, Tuple, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BalancedDatasetOptimizer:
    """Create balanced drug-disease relationship matrix"""
    
    def __init__(self, n_drugs: int = 1000, n_diseases: int = 1000, 
                 target_sparsity: float = 0.98, treatment_ratio: float = 0.2):
        """
        Initialize optimizer
        
        Args:
            n_drugs: Target number of drugs
            n_diseases: Target number of diseases  
            target_sparsity: Target sparsity (0.98 means 2% of positions have data)
            treatment_ratio: Ratio of treatment relationships (0.2 means 20% treatment, 80% side effects)
        """
        self.n_drugs = n_drugs
        self.n_diseases = n_diseases
        self.target_sparsity = target_sparsity
        self.treatment_ratio = treatment_ratio
        
        # Calculate target relationship counts
        total_possible = n_drugs * n_diseases
        self.target_relations = int(total_possible * (1 - target_sparsity))
        self.target_treatments = int(self.target_relations * treatment_ratio)
        self.target_side_effects = self.target_relations - self.target_treatments
        
        logger.info("Target parameters:")
        logger.info(f"  Matrix size: {n_drugs} × {n_diseases}")
        logger.info(f"  Target sparsity: {target_sparsity:.1%}")
        logger.info(f"  Target total relationships: {self.target_relations:,}")
        logger.info(f"  Target treatment relationships: {self.target_treatments:,} ({treatment_ratio:.0%})")
        logger.info(f"  Target side effect relationships: {self.target_side_effects:,} ({1-treatment_ratio:.0%})")
        
        self.disease_mapping = self._create_disease_mapping()
    
    def _create_disease_mapping(self) -> Dict[str, str]:
        """Create disease name standardization mapping"""
        return {
            'blood pressure increased': 'hypertension',
            'blood pressure decreased': 'hypotension',
            'diabetes mellitus': 'diabetes',
            'type 2 diabetes mellitus': 'diabetes',
            'type 1 diabetes mellitus': 'diabetes',
            'cardiac failure': 'heart failure',
            'cardiac failure congestive': 'heart failure',
            'bacterial infections': 'bacterial infection',
            'viral infections': 'viral infection',
            'diarrhoea': 'diarrhea',
            'anaemia': 'anemia',
            'pyrexia': 'fever',
            'oedema': 'edema',
            'dyspnoea': 'dyspnea',
            'blood glucose increased': 'hyperglycemia',
            'blood glucose decreased': 'hypoglycemia',
        }
    
    def create_balanced_dataset(self, drugs_file: str, therapeutic_file: str, 
                               side_effects_file: str, output_dir: str = 'balanced_dataset'):
        """
        Create balanced drug-disease dataset
        
        Args:
            drugs_file: Path to drugs CSV file
            therapeutic_file: Path to therapeutic relationships CSV file
            side_effects_file: Path to side effects CSV file
            output_dir: Output directory for balanced dataset
        """
        logger.info("="*80)
        logger.info(f"Creating balanced {self.n_drugs}×{self.n_diseases} drug-disease relationship matrix")
        logger.info("="*80)
        
        start_time = time.time()
        
        # 1. Load data
        logger.info("1. Loading data...")
        drugs_df = pd.read_csv(drugs_file)
        therapeutic_df = pd.read_csv(therapeutic_file)
        side_effects_df = pd.read_csv(side_effects_file)
        
        logger.info(f"  Total drugs: {len(drugs_df):,}")
        logger.info(f"  Therapeutic relationships: {len(therapeutic_df):,}")
        logger.info(f"  Side effect relationships: {len(side_effects_df):,}")
        
        # 2. Standardize disease names
        logger.info("\n2. Standardizing disease names...")
        therapeutic_df['disease_std'] = therapeutic_df['disease_name'].str.lower().str.strip()
        side_effects_df['disease_std'] = side_effects_df['disease_name'].str.lower().str.strip()
        
        # Apply mapping
        therapeutic_df['disease_std'] = therapeutic_df['disease_std'].map(
            lambda x: self.disease_mapping.get(x, x)
        )
        side_effects_df['disease_std'] = side_effects_df['disease_std'].map(
            lambda x: self.disease_mapping.get(x, x)
        )
        
        # 3. Calculate importance scores
        logger.info("\n3. Calculating importance scores...")
        drug_scores, disease_scores = self._calculate_importance_scores(
            therapeutic_df, side_effects_df
        )
        
        # 4. Select top drugs and diseases
        logger.info(f"\n4. Selecting top {self.n_drugs} drugs and {self.n_diseases} diseases...")
        selected_drugs, selected_diseases = self._select_top_entities(
            drug_scores, disease_scores, therapeutic_df, side_effects_df
        )
        
        # 5. Filter data
        logger.info("\n5. Filtering data to selected entities...")
        filtered_therapeutic, filtered_side_effects = self._filter_data(
            therapeutic_df, side_effects_df, selected_drugs, selected_diseases
        )
        
        # 6. Sample to reach target counts
        logger.info("\n6. Sampling to reach target relationship counts...")
        sampled_treatments, sampled_side_effects = self._sample_relationships(
            filtered_therapeutic, filtered_side_effects
        )
        
        # 7. Create final dataset
        logger.info("\n7. Creating final dataset...")
        final_drugs, final_diseases, interactions_df = self._create_final_dataset(
            drugs_df, selected_drugs, selected_diseases, 
            sampled_treatments, sampled_side_effects
        )
        
        # 8. Save dataset
        self._save_dataset(final_drugs, final_diseases, interactions_df, output_dir)
        
        logger.info(f"\nTotal processing time: {time.time() - start_time:.1f} seconds")
        logger.info(f"Dataset saved to: {output_dir}/")
        
        return final_drugs, final_diseases, interactions_df
    
    def _calculate_importance_scores(self, therapeutic_df: pd.DataFrame, 
                                   side_effects_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Calculate importance scores for drugs and diseases"""
        drug_scores = defaultdict(float)
        disease_scores = defaultdict(float)
        
        # Score based on therapeutic relationships (higher weight)
        for _, row in therapeutic_df.iterrows():
            drug_scores[row['drug_id']] += 10
            disease_scores[row['disease_std']] += 10
        
        # Score based on side effects (lower weight)
        for _, row in side_effects_df.iterrows():
            drug_scores[row['drug_id']] += 1
            disease_scores[row['disease_std']] += 1
        
        # Extra score for entities that appear in both
        drugs_with_treatment = set(therapeutic_df['drug_id'])
        drugs_with_sideeffect = set(side_effects_df['drug_id'])
        drugs_overlap = drugs_with_treatment & drugs_with_sideeffect
        
        diseases_as_treatment = set(therapeutic_df['disease_std'])
        diseases_as_sideeffect = set(side_effects_df['disease_std'])
        diseases_overlap = diseases_as_treatment & diseases_as_sideeffect
        
        # Bonus for overlap
        for drug in drugs_overlap:
            drug_scores[drug] += 20
        for disease in diseases_overlap:
            disease_scores[disease] += 20
        
        return drug_scores, disease_scores
    
    def _select_top_entities(self, drug_scores: Dict, disease_scores: Dict,
                           therapeutic_df: pd.DataFrame, 
                           side_effects_df: pd.DataFrame) -> Tuple[List, List]:
        """Select top drugs and diseases based on scores"""
        # Sort by score
        sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N
        selected_drugs = [drug for drug, _ in sorted_drugs[:self.n_drugs]]
        selected_diseases = [disease for disease, _ in sorted_diseases[:self.n_diseases]]
        
        logger.info(f"  Selected drugs: {len(selected_drugs)}")
        logger.info(f"  Selected diseases: {len(selected_diseases)}")
        
        # Calculate overlap statistics
        drugs_with_treatment = set(therapeutic_df['drug_id'])
        drugs_with_sideeffect = set(side_effects_df['drug_id'])
        drugs_overlap = drugs_with_treatment & drugs_with_sideeffect
        
        diseases_as_treatment = set(therapeutic_df['disease_std'])
        diseases_as_sideeffect = set(side_effects_df['disease_std'])
        diseases_overlap = diseases_as_treatment & diseases_as_sideeffect
        
        selected_drug_overlap = len(set(selected_drugs) & drugs_overlap)
        selected_disease_overlap = len(set(selected_diseases) & diseases_overlap)
        
        logger.info(f"  Selected overlapping drugs: {selected_drug_overlap} "
                   f"({selected_drug_overlap/len(selected_drugs)*100:.1f}%)")
        logger.info(f"  Selected overlapping diseases: {selected_disease_overlap} "
                   f"({selected_disease_overlap/len(selected_diseases)*100:.1f}%)")
        
        return selected_drugs, selected_diseases
    
    def _filter_data(self, therapeutic_df: pd.DataFrame, side_effects_df: pd.DataFrame,
                    selected_drugs: List, selected_diseases: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter data to selected drugs and diseases"""
        filtered_therapeutic = therapeutic_df[
            (therapeutic_df['drug_id'].isin(selected_drugs)) &
            (therapeutic_df['disease_std'].isin(selected_diseases))
        ].copy()
        
        filtered_side_effects = side_effects_df[
            (side_effects_df['drug_id'].isin(selected_drugs)) &
            (side_effects_df['disease_std'].isin(selected_diseases))
        ].copy()
        
        logger.info(f"  Filtered therapeutic relationships: {len(filtered_therapeutic):,}")
        logger.info(f"  Filtered side effect relationships: {len(filtered_side_effects):,}")
        
        return filtered_therapeutic, filtered_side_effects
    
    def _sample_relationships(self, filtered_therapeutic: pd.DataFrame,
                            filtered_side_effects: pd.DataFrame) -> Tuple[Set, Set]:
        """Sample relationships to reach target counts"""
        # Collect all possible treatment pairs
        treatment_pairs = set()
        for _, row in filtered_therapeutic.iterrows():
            pair = (row['drug_id'], row['disease_std'])
            treatment_pairs.add(pair)
        
        # Collect all possible side effect pairs (excluding treatment pairs)
        side_effect_pairs = set()
        for _, row in filtered_side_effects.iterrows():
            pair = (row['drug_id'], row['disease_std'])
            if pair not in treatment_pairs:
                side_effect_pairs.add(pair)
        
        logger.info(f"  Available treatment pairs: {len(treatment_pairs):,}")
        logger.info(f"  Available side effect pairs: {len(side_effect_pairs):,}")
        
        # Sample treatments
        if len(treatment_pairs) > self.target_treatments:
            sampled_treatments = set(random.sample(list(treatment_pairs), self.target_treatments))
        else:
            sampled_treatments = treatment_pairs
            logger.warning(f"  Warning: Not enough treatment relationships. "
                         f"Only {len(sampled_treatments)} available")
        
        # Sample side effects
        if len(side_effect_pairs) > self.target_side_effects:
            sampled_side_effects = set(random.sample(list(side_effect_pairs), self.target_side_effects))
        else:
            sampled_side_effects = side_effect_pairs
            logger.warning(f"  Warning: Not enough side effect relationships. "
                         f"Only {len(sampled_side_effects)} available")
        
        return sampled_treatments, sampled_side_effects
    
    def _create_final_dataset(self, drugs_df: pd.DataFrame, selected_drugs: List,
                            selected_diseases: List, sampled_treatments: Set,
                            sampled_side_effects: Set) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create final dataset components"""
        # Create drug table
        final_drugs = drugs_df[drugs_df['drug_id'].isin(selected_drugs)].copy()
        final_drugs = final_drugs.sort_values('drug_id').reset_index(drop=True)
        
        # Create disease table
        diseases_df = pd.DataFrame({
            'disease_id': range(len(selected_diseases)),
            'disease_name': selected_diseases
        })
        
        # Create disease mapping
        disease_to_id = {disease: idx for idx, disease in enumerate(selected_diseases)}
        
        # Create interactions
        interactions = []
        
        # Add treatment relationships
        for drug_id, disease_name in sampled_treatments:
            interactions.append({
                'drug_id': drug_id,
                'disease_id': disease_to_id[disease_name],
                'disease_name': disease_name,
                'relation_type': 'treatment',
                'label': 1,
                'confidence': 1.0
            })
        
        # Add side effect relationships
        for drug_id, disease_name in sampled_side_effects:
            interactions.append({
                'drug_id': drug_id,
                'disease_id': disease_to_id[disease_name],
                'disease_name': disease_name,
                'relation_type': 'side_effect',
                'label': -1,
                'confidence': 0.9
            })
        
        interactions_df = pd.DataFrame(interactions)
        
        # Ensure no duplicates
        interactions_df = interactions_df.drop_duplicates(subset=['drug_id', 'disease_id'])
        
        # Log statistics
        self._log_final_statistics(final_drugs, diseases_df, interactions_df)
        
        return final_drugs, diseases_df, interactions_df
    
    def _log_final_statistics(self, drugs_df: pd.DataFrame, diseases_df: pd.DataFrame,
                            interactions_df: pd.DataFrame):
        """Log final dataset statistics"""
        logger.info("\n" + "="*80)
        logger.info("Final Dataset Statistics")
        logger.info("="*80)
        
        n_drugs = len(drugs_df)
        n_diseases = len(diseases_df)
        n_interactions = len(interactions_df)
        n_treatments = len(interactions_df[interactions_df['label'] == 1])
        n_side_effects = len(interactions_df[interactions_df['label'] == -1])
        
        logger.info(f"\nMatrix dimensions: {n_drugs} × {n_diseases}")
        logger.info(f"Total relationships: {n_interactions:,}")
        logger.info(f"  - Treatment relationships: {n_treatments:,} ({n_treatments/n_interactions*100:.1f}%)")
        logger.info(f"  - Side effect relationships: {n_side_effects:,} ({n_side_effects/n_interactions*100:.1f}%)")
        
        actual_sparsity = 1 - n_interactions / (n_drugs * n_diseases)
        logger.info(f"\nActual sparsity: {actual_sparsity:.2%} (target: {self.target_sparsity:.2%})")
        
        # Degree distribution
        drug_degrees = interactions_df.groupby('drug_id').size()
        disease_degrees = interactions_df.groupby('disease_id').size()
        
        logger.info(f"\nDegree distribution:")
        logger.info(f"  Drug average degree: {drug_degrees.mean():.1f} "
                   f"(range: {drug_degrees.min()}-{drug_degrees.max()})")
        logger.info(f"  Disease average degree: {disease_degrees.mean():.1f} "
                   f"(range: {disease_degrees.min()}-{disease_degrees.max()})")
        
        # Overlap statistics
        drug_treatments = set(interactions_df[interactions_df['label'] == 1]['drug_id'])
        drug_sideeffects = set(interactions_df[interactions_df['label'] == -1]['drug_id'])
        drug_overlap = len(drug_treatments & drug_sideeffects)
        
        disease_treatments = set(interactions_df[interactions_df['label'] == 1]['disease_id'])
        disease_sideeffects = set(interactions_df[interactions_df['label'] == -1]['disease_id'])
        disease_overlap = len(disease_treatments & disease_sideeffects)
        
        logger.info(f"\nOverlap statistics:")
        logger.info(f"  Drugs with both treatment and side effects: {drug_overlap} "
                   f"({drug_overlap/n_drugs*100:.1f}%)")
        logger.info(f"  Diseases as both treatment target and side effect: {disease_overlap} "
                   f"({disease_overlap/n_diseases*100:.1f}%)")
    
    def _save_dataset(self, drugs_df: pd.DataFrame, diseases_df: pd.DataFrame,
                     interactions_df: pd.DataFrame, output_dir: str):
        """Save dataset files"""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"\nSaving dataset to {output_dir}/...")
        
        # Save main files
        drugs_df.to_csv(os.path.join(output_dir, 'drugs.csv'), index=False)
        diseases_df.to_csv(os.path.join(output_dir, 'diseases.csv'), index=False)
        interactions_df.to_csv(os.path.join(output_dir, 'drug_disease_interactions.csv'), index=False)
        
        # Save sparse format
        sparse_df = interactions_df[['drug_id', 'disease_id', 'label', 'confidence']].copy()
        sparse_df.to_csv(os.path.join(output_dir, 'interactions_sparse.csv'), index=False)
        
        # Create and save matrix view (for validation)
        n_drugs = len(drugs_df)
        n_diseases = len(diseases_df)
        matrix = np.zeros((n_drugs, n_diseases))
        
        drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(drugs_df['drug_id'])}
        
        for _, row in interactions_df.iterrows():
            drug_idx = drug_to_idx[row['drug_id']]
            disease_idx = row['disease_id']
            matrix[drug_idx, disease_idx] = row['label']
        
        # Save summary
        summary = {
            'matrix_size': f"{n_drugs} × {n_diseases}",
            'target_sparsity': float(self.target_sparsity),
            'actual_sparsity': float(1 - len(interactions_df) / (n_drugs * n_diseases)),
            'n_interactions': int(len(interactions_df)),
            'n_treatments': int(len(interactions_df[interactions_df['label'] == 1])),
            'n_side_effects': int(len(interactions_df[interactions_df['label'] == -1])),
            'treatment_ratio': float(len(interactions_df[interactions_df['label'] == 1]) / len(interactions_df)),
            'drug_degree_stats': {
                'mean': float(interactions_df.groupby('drug_id').size().mean()),
                'min': int(interactions_df.groupby('drug_id').size().min()),
                'max': int(interactions_df.groupby('drug_id').size().max())
            },
            'disease_degree_stats': {
                'mean': float(interactions_df.groupby('disease_id').size().mean()),
                'min': int(interactions_df.groupby('disease_id').size().min()),
                'max': int(interactions_df.groupby('disease_id').size().max())
            }
        }
        
        with open(os.path.join(output_dir, 'dataset_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    """Main function to create balanced dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create balanced drug-disease dataset')
    parser.add_argument('--drugs-file', required=True, help='Path to drugs CSV file')
    parser.add_argument('--therapeutic-file', required=True, help='Path to therapeutic relations CSV file')
    parser.add_argument('--side-effects-file', required=True, help='Path to side effects CSV file')
    parser.add_argument('--output-dir', default='balanced_dataset', help='Output directory')
    parser.add_argument('--n-drugs', type=int, default=1000, help='Number of drugs to select')
    parser.add_argument('--n-diseases', type=int, default=1000, help='Number of diseases to select')
    parser.add_argument('--sparsity', type=float, default=0.98, help='Target sparsity (0-1)')
    parser.add_argument('--treatment-ratio', type=float, default=0.2, 
                       help='Ratio of treatment relationships (0-1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create optimizer
    optimizer = BalancedDatasetOptimizer(
        n_drugs=args.n_drugs,
        n_diseases=args.n_diseases,
        target_sparsity=args.sparsity,
        treatment_ratio=args.treatment_ratio
    )
    
    # Run optimization
    optimizer.create_balanced_dataset(
        drugs_file=args.drugs_file,
        therapeutic_file=args.therapeutic_file,
        side_effects_file=args.side_effects_file,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
