# FILE: dataset_builder_v2.py
#
# DESCRIPTION:
# Improved dataset builder that ensures better balance between treatment
# and side effect diseases, and achieves higher overlap.

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import json
import random

class BalancedDatasetBuilderV2:
    """Enhanced builder for balanced drug-disease relationship dataset."""
    
    def __init__(self, n_drugs=1000, n_diseases=1000, target_sparsity=0.98, 
                 treatment_ratio=0.2, target_disease_overlap=0.3,
                 min_treatment_diseases_ratio=0.4):
        """
        Args:
            n_drugs: Target number of drugs
            n_diseases: Target number of diseases  
            target_sparsity: Desired sparsity of the final matrix
            treatment_ratio: Desired ratio of treatments to all relations
            target_disease_overlap: Minimum desired ratio of overlapping diseases
            min_treatment_diseases_ratio: Minimum ratio of diseases that should
                                          appear in treatment context (new param)
        """
        self.n_drugs = n_drugs
        self.n_diseases = n_diseases
        self.target_sparsity = target_sparsity
        self.treatment_ratio = treatment_ratio
        self.target_disease_overlap = target_disease_overlap
        self.min_treatment_diseases_ratio = min_treatment_diseases_ratio
        
        total_possible = n_drugs * n_diseases
        self.target_relations = int(total_possible * (1 - target_sparsity))
        self.target_treatments = int(self.target_relations * treatment_ratio)
        self.target_side_effects = self.target_relations - self.target_treatments
        
        print("Enhanced dataset parameters:")
        print(f"  - Matrix size: {n_drugs} x {n_diseases}")
        print(f"  - Target Disease Overlap: {self.target_disease_overlap:.0%}")
        print(f"  - Min Treatment Diseases Ratio: {self.min_treatment_diseases_ratio:.0%}")
        print(f"  - Target total relationships: {self.target_relations:,}")
        print(f"  - Target treatments: {self.target_treatments:,}")
        print(f"  - Target side effects: {self.target_side_effects:,}")

    def create_balanced_dataset(self, drugs_file, therapeutic_file, side_effects_file, output_dir):
        """Creates balanced dataset with improved overlap and balance."""
        print("\nStarting enhanced balanced dataset creation...")
        
        # Load data
        drugs_df, therapeutic_df, side_effects_df = self._load_data(
            drugs_file, therapeutic_file, side_effects_file
        )
        
        # Analyze original data distribution
        self._analyze_original_distribution(therapeutic_df, side_effects_df)
        
        # Select entities with enhanced strategy
        selected_drugs, selected_diseases = self._select_entities_enhanced(
            therapeutic_df, side_effects_df
        )
        
        # Filter to selected entities
        filtered_therapeutic = therapeutic_df[
            therapeutic_df['drug_id'].isin(selected_drugs) & 
            therapeutic_df['disease_name'].isin(selected_diseases)
        ]
        filtered_side_effects = side_effects_df[
            side_effects_df['drug_id'].isin(selected_drugs) & 
            side_effects_df['disease_name'].isin(selected_diseases)
        ]
        
        # Sample interactions with balance enforcement
        interactions_df = self._sample_interactions_balanced(
            filtered_therapeutic, filtered_side_effects
        )
        
        # Save the dataset
        self._save_dataset(
            interactions_df, drugs_df, selected_drugs, selected_diseases, output_dir
        )
        
        # Analyze final dataset
        self._analyze_final_dataset(interactions_df)

    def _load_data(self, drugs_file, therapeutic_file, side_effects_file):
        """Loads and standardizes data."""
        print("Loading and standardizing data...")
        
        drugs_df = pd.read_csv(drugs_file)
        therapeutic_df = pd.read_csv(therapeutic_file)
        side_effects_df = pd.read_csv(side_effects_file)
        
        # Map side effect drug names to IDs
        name_to_id_map = pd.Series(
            drugs_df.drug_id.values, 
            index=drugs_df.name.str.lower()
        ).to_dict()
        
        side_effects_df['drug_id'] = (
            side_effects_df['drug_id'].str.lower().map(name_to_id_map)
        )
        side_effects_df.dropna(subset=['drug_id'], inplace=True)
        
        # Clean disease names
        therapeutic_df.dropna(subset=['disease_name'], inplace=True)
        side_effects_df.dropna(subset=['disease_name'], inplace=True)
        
        therapeutic_df['disease_name'] = (
            therapeutic_df['disease_name'].str.lower().str.strip()
        )
        side_effects_df['disease_name'] = (
            side_effects_df['disease_name'].str.lower().str.strip()
        )
        
        print(f"Loaded {len(drugs_df)} drugs")
        print(f"Loaded {len(therapeutic_df)} therapeutic relations")
        print(f"Loaded {len(side_effects_df)} side effect relations")
        
        return drugs_df, therapeutic_df, side_effects_df

    def _analyze_original_distribution(self, therapeutic_df, side_effects_df):
        """Analyzes the original data distribution."""
        print("\nAnalyzing original data distribution...")
        
        therapeutic_diseases = set(therapeutic_df['disease_name'].unique())
        side_effect_diseases = set(side_effects_df['disease_name'].unique())
        overlap = therapeutic_diseases.intersection(side_effect_diseases)
        
        print(f"Original therapeutic diseases: {len(therapeutic_diseases)}")
        print(f"Original side effect diseases: {len(side_effect_diseases)}")
        print(f"Original overlap: {len(overlap)} ({len(overlap)/len(therapeutic_diseases.union(side_effect_diseases))*100:.1f}%)")

    def _select_entities_enhanced(self, therapeutic_df, side_effects_df):
        """Enhanced entity selection with better balance and overlap."""
        print("\nSelecting entities with enhanced strategy...")
        
        # Calculate drug scores (unchanged)
        drug_scores = defaultdict(float)
        for drug in therapeutic_df['drug_id']: 
            drug_scores[drug] += 5
        for drug in side_effects_df['drug_id']: 
            drug_scores[drug] += 1
            
        # Select top drugs
        sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
        selected_drugs = [drug for drug, _ in sorted_drugs[:self.n_drugs]]
        
        # Enhanced disease selection
        therapeutic_diseases = set(therapeutic_df['disease_name'])
        side_effect_diseases = set(side_effects_df['disease_name'])
        overlap_diseases = therapeutic_diseases.intersection(side_effect_diseases)
        
        # Calculate disease importance scores
        disease_scores = defaultdict(float)
        
        # Score based on frequency
        for disease in therapeutic_df['disease_name']:
            disease_scores[disease] += 3  # Boost therapeutic diseases more
        for disease in side_effects_df['disease_name']:
            disease_scores[disease] += 1
            
        # Bonus for overlap diseases
        for disease in overlap_diseases:
            disease_scores[disease] += 10  # Strong bonus for overlap
        
        # Ensure minimum treatment diseases
        min_treatment_diseases = int(self.n_diseases * self.min_treatment_diseases_ratio)
        target_overlap_count = int(self.n_diseases * self.target_disease_overlap)
        
        # Step 1: Select overlap diseases first (up to target)
        overlap_candidates = sorted(
            [(d, disease_scores[d]) for d in overlap_diseases],
            key=lambda x: x[1], 
            reverse=True
        )
        selected_overlap = [d for d, _ in overlap_candidates[:target_overlap_count]]
        selected_diseases = set(selected_overlap)
        
        print(f"Selected {len(selected_overlap)} overlap diseases")
        
        # Step 2: Ensure we have enough treatment diseases
        treatment_only = therapeutic_diseases - overlap_diseases
        treatment_candidates = sorted(
            [(d, disease_scores[d]) for d in treatment_only],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate how many more treatment diseases we need
        current_treatment_diseases = len(selected_overlap)  # All overlap diseases are treatment diseases
        additional_treatment_needed = max(
            0, 
            min_treatment_diseases - current_treatment_diseases
        )
        
        # Add treatment-only diseases
        for disease, _ in treatment_candidates[:additional_treatment_needed]:
            if len(selected_diseases) < self.n_diseases:
                selected_diseases.add(disease)
        
        print(f"Added {len(selected_diseases) - len(selected_overlap)} treatment-only diseases")
        
        # Step 3: Fill remaining slots with highest scoring diseases
        remaining_needed = self.n_diseases - len(selected_diseases)
        all_diseases = therapeutic_diseases.union(side_effect_diseases)
        remaining_candidates = sorted(
            [(d, disease_scores[d]) for d in all_diseases if d not in selected_diseases],
            key=lambda x: x[1],
            reverse=True
        )
        
        for disease, _ in remaining_candidates[:remaining_needed]:
            selected_diseases.add(disease)
        
        selected_diseases = list(selected_diseases)[:self.n_diseases]
        
        print(f"Final selection: {len(selected_diseases)} diseases")
        
        return selected_drugs, selected_diseases

    def _sample_interactions_balanced(self, therapeutic_df, side_effects_df):
        """Sample interactions with balance enforcement."""
        print("\nSampling balanced interactions...")
        
        # Get all possible pairs
        treatment_pairs = set(zip(
            therapeutic_df['drug_id'], 
            therapeutic_df['disease_name']
        ))
        side_effect_pairs = set(zip(
            side_effects_df['drug_id'], 
            side_effects_df['disease_name']
        ))
        
        # Remove conflicts (same pair can't be both treatment and side effect)
        side_effect_pairs -= treatment_pairs
        
        # Analyze disease coverage
        treatment_diseases = set(pair[1] for pair in treatment_pairs)
        side_effect_diseases = set(pair[1] for pair in side_effect_pairs)
        
        print(f"Available treatment diseases: {len(treatment_diseases)}")
        print(f"Available side effect diseases: {len(side_effect_diseases)}")
        print(f"Disease overlap in filtered data: {len(treatment_diseases.intersection(side_effect_diseases))}")
        
        # Sample with stratification to ensure disease diversity
        sampled_treatments = self._stratified_sample(
            list(treatment_pairs), 
            self.target_treatments,
            key_func=lambda x: x[1]  # Group by disease
        )
        
        sampled_side_effects = self._stratified_sample(
            list(side_effect_pairs),
            self.target_side_effects,
            key_func=lambda x: x[1]  # Group by disease
        )
        
        # Create interaction dataframe
        interactions = []
        for drug_id, disease_name in sampled_treatments:
            interactions.append({
                'drug_id': drug_id, 
                'disease_name': disease_name, 
                'label': 1
            })
        for drug_id, disease_name in sampled_side_effects:
            interactions.append({
                'drug_id': drug_id, 
                'disease_name': disease_name, 
                'label': -1
            })
            
        return pd.DataFrame(interactions)

    def _stratified_sample(self, pairs, n_samples, key_func):
        """Stratified sampling to ensure diversity."""
        if len(pairs) <= n_samples:
            return pairs
            
        # Group by key
        groups = defaultdict(list)
        for pair in pairs:
            key = key_func(pair)
            groups[key].append(pair)
        
        # Calculate samples per group
        samples_per_group = max(1, n_samples // len(groups))
        sampled = []
        
        # Sample from each group
        for key, group_pairs in groups.items():
            n = min(len(group_pairs), samples_per_group)
            sampled.extend(random.sample(group_pairs, n))
        
        # Fill remaining quota randomly
        remaining = n_samples - len(sampled)
        if remaining > 0:
            unsampled = [p for p in pairs if p not in sampled]
            if unsampled:
                additional = random.sample(
                    unsampled, 
                    min(remaining, len(unsampled))
                )
                sampled.extend(additional)
        
        return sampled[:n_samples]

    def _save_dataset(self, interactions_df, drugs_df, selected_drugs, 
                      selected_diseases, output_dir):
        """Save the dataset files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare drug data
        final_drugs = drugs_df[drugs_df['drug_id'].isin(selected_drugs)].copy()
        
        # Prepare disease data
        disease_to_id = {name: i for i, name in enumerate(selected_diseases)}
        final_diseases = pd.DataFrame({
            'disease_id': range(len(selected_diseases)), 
            'disease_name': selected_diseases
        })
        
        # Map disease names to IDs in interactions
        interactions_df['disease_id'] = interactions_df['disease_name'].map(disease_to_id)
        interactions_df.dropna(subset=['disease_id'], inplace=True)
        interactions_df['disease_id'] = interactions_df['disease_id'].astype(int)
        
        # Save files
        final_drugs.to_csv(os.path.join(output_dir, 'drugs.csv'), index=False)
        final_diseases.to_csv(os.path.join(output_dir, 'diseases.csv'), index=False)
        interactions_df.to_csv(
            os.path.join(output_dir, 'drug_disease_interactions.csv'), 
            index=False
        )
        
        print(f"\nDataset saved to '{output_dir}/'")
        print(f"  - Drugs: {len(final_drugs)}")
        print(f"  - Diseases: {len(final_diseases)}")
        print(f"  - Interactions: {len(interactions_df)}")

    def _analyze_final_dataset(self, interactions_df):
        """Analyze the final dataset balance and overlap."""
        print("\nFinal dataset analysis:")
        
        treatments = interactions_df[interactions_df['label'] == 1]
        side_effects = interactions_df[interactions_df['label'] == -1]
        
        treatment_diseases = set(treatments['disease_name'].unique())
        side_effect_diseases = set(side_effects['disease_name'].unique())
        overlap = treatment_diseases.intersection(side_effect_diseases)
        all_diseases = treatment_diseases.union(side_effect_diseases)
        
        print(f"Treatment interactions: {len(treatments)}")
        print(f"Side effect interactions: {len(side_effects)}")
        print(f"Treatment diseases: {len(treatment_diseases)}")
        print(f"Side effect diseases: {len(side_effect_diseases)}")
        print(f"Overlapping diseases: {len(overlap)}")
        print(f"Disease overlap percentage: {len(overlap)/len(all_diseases)*100:.1f}%")

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    
    # Create dataset with enhanced parameters
    builder = BalancedDatasetBuilderV2(
        n_drugs=1000,
        n_diseases=1000,
        target_sparsity=0.98,
        treatment_ratio=0.2,
        target_disease_overlap=0.3,
        min_treatment_diseases_ratio=0.4  # At least 40% diseases should have treatment info
    )
    
    builder.create_balanced_dataset(
        drugs_file='drugbank_drugs.csv',
        therapeutic_file='drugbank_therapeutic_relations_enhanced.csv',
        side_effects_file='offside_enhanced_side_effects.csv',
        output_dir='balanced_drug_disease_dataset_v2'
    )