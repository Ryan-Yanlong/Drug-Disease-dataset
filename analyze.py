# FILE: analyze_enhanced.py
#
# DESCRIPTION:
# Enhanced analysis script that provides detailed statistics about dataset
# balance, overlap, and distribution.

import pandas as pd
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset_comprehensive(dataset_dir='balanced_drug_disease_dataset_v2'):
    """
    Comprehensive analysis of the drug-disease dataset.
    
    Args:
        dataset_dir: Directory containing the dataset files
    """
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE DATASET ANALYSIS")
    print(f"Dataset location: {dataset_dir}")
    print(f"{'='*60}\n")

    # Define file paths
    interactions_file = os.path.join(dataset_dir, 'drug_disease_interactions.csv')
    drugs_file = os.path.join(dataset_dir, 'drugs.csv')
    diseases_file = os.path.join(dataset_dir, 'diseases.csv')

    # Check if files exist
    if not all(os.path.exists(f) for f in [interactions_file, drugs_file, diseases_file]):
        print("Error: Dataset files not found. Please run the pipeline first.")
        return

    # Load data
    interactions_df = pd.read_csv(interactions_file)
    drugs_df = pd.read_csv(drugs_file)
    diseases_df = pd.read_csv(diseases_file)

    # Basic statistics
    print("1. BASIC STATISTICS")
    print("-" * 40)
    print(f"Total drugs: {len(drugs_df):,}")
    print(f"Total diseases: {len(diseases_df):,}")
    print(f"Total interactions: {len(interactions_df):,}")
    print(f"Matrix density: {len(interactions_df) / (len(drugs_df) * len(diseases_df)) * 100:.2f}%")
    
    # Interaction breakdown
    treatments = interactions_df[interactions_df['label'] == 1]
    side_effects = interactions_df[interactions_df['label'] == -1]
    
    print(f"\nTreatment relations: {len(treatments):,} ({len(treatments)/len(interactions_df)*100:.1f}%)")
    print(f"Side effect relations: {len(side_effects):,} ({len(side_effects)/len(interactions_df)*100:.1f}%)")

    # Drug analysis
    print("\n2. DRUG OVERLAP ANALYSIS")
    print("-" * 40)
    
    drugs_treatment = set(treatments['drug_id'])
    drugs_side_effect = set(side_effects['drug_id'])
    drug_overlap = drugs_treatment.intersection(drugs_side_effect)
    
    print(f"Drugs with treatment info: {len(drugs_treatment):,}")
    print(f"Drugs with side effect info: {len(drugs_side_effect):,}")
    print(f"Drugs with BOTH: {len(drug_overlap):,}")
    print(f"Drug overlap rate: {len(drug_overlap)/len(drugs_df)*100:.1f}%")
    
    # Drugs exclusive to each category
    treatment_only_drugs = drugs_treatment - drugs_side_effect
    side_effect_only_drugs = drugs_side_effect - drugs_treatment
    print(f"\nDrugs with ONLY treatment info: {len(treatment_only_drugs):,}")
    print(f"Drugs with ONLY side effect info: {len(side_effect_only_drugs):,}")

    # Disease analysis
    print("\n3. DISEASE OVERLAP ANALYSIS")
    print("-" * 40)
    
    diseases_treatment = set(treatments['disease_id'])
    diseases_side_effect = set(side_effects['disease_id'])
    disease_overlap = diseases_treatment.intersection(diseases_side_effect)
    
    print(f"Diseases as treatment targets: {len(diseases_treatment):,}")
    print(f"Diseases as side effects: {len(diseases_side_effect):,}")
    print(f"Diseases appearing as BOTH: {len(disease_overlap):,}")
    print(f"Disease overlap rate: {len(disease_overlap)/len(diseases_df)*100:.1f}%")
    
    # Diseases exclusive to each category
    treatment_only_diseases = diseases_treatment - diseases_side_effect
    side_effect_only_diseases = diseases_side_effect - diseases_treatment
    print(f"\nDiseases ONLY as treatment targets: {len(treatment_only_diseases):,}")
    print(f"Diseases ONLY as side effects: {len(side_effect_only_diseases):,}")

    # Distribution analysis
    print("\n4. DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    # Drug degree distribution
    drug_treatment_counts = treatments['drug_id'].value_counts()
    drug_side_effect_counts = side_effects['drug_id'].value_counts()
    
    print(f"\nDrug-Treatment Distribution:")
    print(f"  Mean treatments per drug: {drug_treatment_counts.mean():.2f}")
    print(f"  Median treatments per drug: {drug_treatment_counts.median():.0f}")
    print(f"  Max treatments per drug: {drug_treatment_counts.max()}")
    
    print(f"\nDrug-Side Effect Distribution:")
    print(f"  Mean side effects per drug: {drug_side_effect_counts.mean():.2f}")
    print(f"  Median side effects per drug: {drug_side_effect_counts.median():.0f}")
    print(f"  Max side effects per drug: {drug_side_effect_counts.max()}")
    
    # Disease degree distribution
    disease_treatment_counts = treatments['disease_id'].value_counts()
    disease_side_effect_counts = side_effects['disease_id'].value_counts()
    
    print(f"\nDisease-Treatment Distribution:")
    print(f"  Mean drugs per treatment target: {disease_treatment_counts.mean():.2f}")
    print(f"  Median drugs per treatment target: {disease_treatment_counts.median():.0f}")
    print(f"  Max drugs per treatment target: {disease_treatment_counts.max()}")
    
    print(f"\nDisease-Side Effect Distribution:")
    print(f"  Mean drugs causing disease as side effect: {disease_side_effect_counts.mean():.2f}")
    print(f"  Median drugs causing disease as side effect: {disease_side_effect_counts.median():.0f}")
    print(f"  Max drugs causing disease as side effect: {disease_side_effect_counts.max()}")

    # Top diseases analysis
    print("\n5. TOP DISEASES ANALYSIS")
    print("-" * 40)
    
    # Convert value_counts to DataFrame with proper column names
    disease_treatment_counts_df = disease_treatment_counts.reset_index()
    disease_treatment_counts_df.columns = ['disease_id', 'treatment_count']
    
    disease_side_effect_counts_df = disease_side_effect_counts.reset_index()
    disease_side_effect_counts_df.columns = ['disease_id', 'side_effect_count']
    
    # Merge with disease names
    disease_treatment_df = pd.merge(
        disease_treatment_counts_df,
        diseases_df,
        on='disease_id'
    )[['disease_name', 'disease_id', 'treatment_count']]
    
    disease_side_effect_df = pd.merge(
        disease_side_effect_counts_df,
        diseases_df,
        on='disease_id'
    )[['disease_name', 'disease_id', 'side_effect_count']]
    
    print("\nTop 10 Treatment Targets:")
    for _, row in disease_treatment_df.nlargest(10, 'treatment_count').iterrows():
        print(f"  {row['disease_name']}: {row['treatment_count']} drugs")
    
    print("\nTop 10 Side Effects:")
    for _, row in disease_side_effect_df.nlargest(10, 'side_effect_count').iterrows():
        print(f"  {row['disease_name']}: {row['side_effect_count']} drugs")

    # Overlap diseases details
    if len(disease_overlap) > 0:
        print("\n6. OVERLAP DISEASES DETAILS")
        print("-" * 40)
        print(f"Total diseases appearing in both contexts: {len(disease_overlap)}")
        
        # Get details for overlap diseases
        overlap_df = diseases_df[diseases_df['disease_id'].isin(disease_overlap)].copy()
        
        # Add counts
        overlap_df['treatment_count'] = overlap_df['disease_id'].map(
            disease_treatment_counts.to_dict()
        ).fillna(0).astype(int)
        overlap_df['side_effect_count'] = overlap_df['disease_id'].map(
            disease_side_effect_counts.to_dict()
        ).fillna(0).astype(int)
        overlap_df['total_count'] = overlap_df['treatment_count'] + overlap_df['side_effect_count']
        
        print("\nTop 10 Overlap Diseases (by total connections):")
        for _, row in overlap_df.nlargest(10, 'total_count').iterrows():
            print(f"  {row['disease_name']}:")
            print(f"    - As treatment: {row['treatment_count']} drugs")
            print(f"    - As side effect: {row['side_effect_count']} drugs")

    # Save summary report
    report_file = os.path.join(dataset_dir, 'dataset_analysis_report.txt')
    with open(report_file, 'w') as f:
        f.write(f"Dataset Analysis Report\n")
        f.write(f"Generated from: {dataset_dir}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Summary Statistics:\n")
        f.write(f"- Total drugs: {len(drugs_df)}\n")
        f.write(f"- Total diseases: {len(diseases_df)}\n")
        f.write(f"- Total interactions: {len(interactions_df)}\n")
        f.write(f"- Drug overlap rate: {len(drug_overlap)/len(drugs_df)*100:.1f}%\n")
        f.write(f"- Disease overlap rate: {len(disease_overlap)/len(diseases_df)*100:.1f}%\n")
        f.write(f"- Treatment/Side-effect ratio: {len(treatments)}/{len(side_effects)}\n")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Report saved to: {report_file}")
    print(f"{'='*60}")

    return {
        'n_drugs': len(drugs_df),
        'n_diseases': len(diseases_df),
        'n_interactions': len(interactions_df),
        'drug_overlap_rate': len(drug_overlap)/len(drugs_df)*100,
        'disease_overlap_rate': len(disease_overlap)/len(diseases_df)*100,
        'treatment_ratio': len(treatments)/len(interactions_df)*100
    }


def plot_distributions(dataset_dir='balanced_drug_disease_dataset_v2'):
    """Create visualization plots for the dataset."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/Seaborn not installed. Skipping visualizations.")
        return
    
    # Load data
    interactions_df = pd.read_csv(os.path.join(dataset_dir, 'drug_disease_interactions.csv'))
    treatments = interactions_df[interactions_df['label'] == 1]
    side_effects = interactions_df[interactions_df['label'] == -1]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Dataset Distribution Analysis', fontsize=16)
    
    # Drug degree distributions
    ax = axes[0, 0]
    drug_treatment_counts = treatments['drug_id'].value_counts()
    drug_side_effect_counts = side_effects['drug_id'].value_counts()
    
    ax.hist([drug_treatment_counts, drug_side_effect_counts], 
            bins=30, alpha=0.7, label=['Treatments', 'Side Effects'])
    ax.set_xlabel('Number of relations per drug')
    ax.set_ylabel('Count')
    ax.set_title('Drug Degree Distribution')
    ax.legend()
    
    # Disease degree distributions
    ax = axes[0, 1]
    disease_treatment_counts = treatments['disease_id'].value_counts()
    disease_side_effect_counts = side_effects['disease_id'].value_counts()
    
    ax.hist([disease_treatment_counts, disease_side_effect_counts], 
            bins=30, alpha=0.7, label=['As Treatment', 'As Side Effect'])
    ax.set_xlabel('Number of drugs per disease')
    ax.set_ylabel('Count')
    ax.set_title('Disease Degree Distribution')
    ax.legend()
    
    # Overlap visualization
    ax = axes[1, 0]
    diseases_treatment = set(treatments['disease_id'])
    diseases_side_effect = set(side_effects['disease_id'])
    
    venn_data = [
        len(diseases_treatment - diseases_side_effect),
        len(diseases_side_effect - diseases_treatment),
        len(diseases_treatment.intersection(diseases_side_effect))
    ]
    
    ax.bar(['Treatment Only', 'Side Effect Only', 'Both'], venn_data)
    ax.set_ylabel('Number of Diseases')
    ax.set_title('Disease Category Distribution')
    
    # Summary stats
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""Dataset Summary:
    
Total Interactions: {len(interactions_df):,}
Treatment Relations: {len(treatments):,}
Side Effect Relations: {len(side_effects):,}

Disease Overlap: {venn_data[2]/sum(venn_data)*100:.1f}%
Treatment Ratio: {len(treatments)/len(interactions_df)*100:.1f}%
"""
    ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plot_file = os.path.join(dataset_dir, 'dataset_distributions.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {plot_file}")
    plt.close()


if __name__ == '__main__':
    # Check both possible dataset directories
    if os.path.exists('balanced_drug_disease_dataset_v2'):
        dataset_dir = 'balanced_drug_disease_dataset_v2'
    elif os.path.exists('balanced_drug_disease_dataset'):
        dataset_dir = 'balanced_drug_disease_dataset'
    else:
        print("No dataset directory found. Please run the pipeline first.")
        exit(1)
    
    # Run comprehensive analysis
    results = analyze_dataset_comprehensive(dataset_dir)
    
    # Create visualizations if possible
    plot_distributions(dataset_dir)
