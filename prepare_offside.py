# FILE: prepare_offside.py
#
# DESCRIPTION:
# This script serves as a one-time preliminary step for the pipeline.
# It reads the raw OFFSIDES data file (e.g., 'OFFSIDES.csv'), performs
# initial cleaning, filtering, and formatting, and saves the result as
# 'offside_raw_side_effects.csv'. This output file is then used as the
# starting point for side effect processing in the main pipeline.
#
# USAGE:
# Run this script directly from your terminal before starting the main pipeline:
# python prepare_offside.py

import pandas as pd
import os
import logging

def prepare_offside_data():
    """
    Reads, processes, and formats the raw OFFSIDES data file.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Preparing OFFSIDE data...")

    # Look for common OFFSIDES file names
    possible_files = [
        'OFFSIDES.csv'
    ]

    offside_file = None
    for filename in possible_files:
        if os.path.exists(filename):
            offside_file = filename
            logger.info(f"Found OFFSIDE file: {filename}")
            break

    if not offside_file:
        logger.error("OFFSIDE data file not found!")
        logger.info("Please ensure your OFFSIDE CSV file is in the current directory.")
        logger.info("Expected file names: " + ", ".join(possible_files))
        return False

    # Read OFFSIDE data
    logger.info(f"Reading OFFSIDE data from {offside_file}...")
    df = pd.read_csv(offside_file, low_memory=False)

    logger.info(f"Original OFFSIDE data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Check for expected columns
    expected_columns = ['drug_concept_name', 'condition_concept_name', 'mean_reporting_frequency']
    if not all(col in df.columns for col in expected_columns):
        logger.error(f"Missing expected columns. Found: {df.columns.tolist()}")
        return False

    # Process the data
    logger.info("Processing OFFSIDE data...")
    processed_df = pd.DataFrame({
        'drug_id': df['drug_concept_name'].str.strip(),
        'disease_name': df['condition_concept_name'].str.strip(),
        'frequency': df['mean_reporting_frequency']
    })

    # **FIX:** Convert frequency to a numeric type. Values that can't be converted become NaN.
    processed_df['frequency'] = pd.to_numeric(processed_df['frequency'], errors='coerce')

    # Remove rows with missing values (this now also removes non-numeric frequencies).
    processed_df.dropna(inplace=True)

    # Filter by a minimum frequency threshold
    min_frequency = 0.01  # 1% reporting frequency
    processed_df = processed_df[processed_df['frequency'] >= min_frequency]

    # Remove duplicates
    processed_df.drop_duplicates(subset=['drug_id', 'disease_name'], inplace=True)

    # Clean drug names to improve matching with DrugBank
    logger.info("Cleaning drug names...")
    drug_name_replacements = {
        ', USP': '', ' USP': '', ' HCl': '', ' HCL': '',
        ' hydrochloride': '', ' maleate': '', ' tartrate': '',
        ' sulfate': '', ' sodium': '', ' potassium': '', ' calcium': '',
    }
    for pattern, replacement in drug_name_replacements.items():
        processed_df['drug_id'] = processed_df['drug_id'].str.replace(
            pattern, replacement, case=False, regex=False
        )
    processed_df['drug_id'] = processed_df['drug_id'].str.strip()

    # Show statistics
    logger.info("\nProcessed OFFSIDE statistics:")
    logger.info(f"  Total side effects: {len(processed_df)}")
    logger.info(f"  Unique drugs: {processed_df['drug_id'].nunique()}")
    logger.info(f"  Unique conditions: {processed_df['disease_name'].nunique()}")

    # Save the processed file, which will be used by the main pipeline
    output_file = 'offside_raw_side_effects.csv'
    processed_df.to_csv(output_file, index=False)
    logger.info(f"\nSaved processed OFFSIDE data to: {output_file}")
    
    return True

if __name__ == '__main__':
    prepare_offside_data()