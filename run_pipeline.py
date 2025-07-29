# FILE: run_pipeline_v2.py
#
# DESCRIPTION:
# Enhanced pipeline runner that uses the improved DrugBank parser and
# dataset builder for better disease balance and overlap.

import os
import sys
import time
import json
import logging
from datetime import datetime
import pandas as pd
import random
import numpy as np

# Import the enhanced modules
from drugbank_parser import parse_drugbank_xml
from disease_standardizer import enhance_disease_mapping
from dataset_builder import BalancedDatasetBuilderV2

class EnhancedDrugRepositioningPipeline:
    """Enhanced pipeline with better disease balance and overlap."""

    def __init__(self, config_file='config_v2.json'):
        """Initialize enhanced pipeline."""
        self.start_time = time.time()
        self._setup_logging()
        self.config = self._load_config(config_file)
        self.status = {}

    def _setup_logging(self):
        """Configure logging."""
        log_dir = 'pipeline_logs'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'enhanced_pipeline_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file), 
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_file):
        """Load pipeline configuration."""
        if not os.path.exists(config_file):
            # Create default config if not exists
            default_config = {
                "drugbank_xml": "drugbank.xml",
                "drugbank_drugs": "drugbank_drugs.csv",
                "drugbank_therapeutic": "drugbank_therapeutic_relations.csv",
                "drugbank_therapeutic_enhanced": "drugbank_therapeutic_enhanced.csv",
                "offside_raw": "offside_raw_side_effects.csv",
                "offside_enhanced": "offside_enhanced_side_effects.csv",
                "output_dir": "balanced_drug_disease_dataset_v2",
                "n_drugs": 1000,
                "n_diseases": 1000,
                "target_sparsity": 0.98,
                "treatment_ratio": 0.2,
                "target_disease_overlap": 0.3,
                "min_treatment_diseases_ratio": 0.4
            }
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            self.logger.info(f"Created default config file: {config_file}")
            return default_config
        
        with open(config_file, 'r') as f:
            return json.load(f)

    def _run_step(self, step_name, function, *args, **kwargs):
        """Run a pipeline step with logging."""
        self.logger.info(f"\n{'='*80}\n>>> STARTING: {step_name}\n{'='*80}")
        step_start_time = time.time()
        try:
            result = function(*args, **kwargs)
            elapsed = time.time() - step_start_time
            self.logger.info(
                f"\n{'='*80}\n>>> SUCCESS: {step_name} "
                f"(Completed in {elapsed:.2f}s)\n{'='*80}"
            )
            self.status[step_name] = 'Completed'
            return True
        except Exception as e:
            self.logger.error(
                f"!!! FAILED: {step_name} with error: {e}", 
                exc_info=True
            )
            self.status[step_name] = f'Failed: {e}'
            return False

    def run(self):
        """Execute the enhanced pipeline."""
        self.logger.info("--- Enhanced Drug Repositioning Pipeline: START ---")

        # Step 1: Parse DrugBank with enhanced extraction
        if not self._run_step(
            "Step 1: Enhanced DrugBank Parsing", 
            self.step1_parse_drugbank_enhanced
        ):
            self._finalize_run()
            return

        # Step 2: Standardize disease names
        if not self._run_step(
            "Step 2: Standardize Diseases", 
            self.step2_standardize_diseases
        ):
            self._finalize_run()
            return

        # Step 3: Build balanced dataset with enhanced algorithm
        if not self._run_step(
            "Step 3: Build Enhanced Balanced Dataset", 
            self.step3_build_dataset_enhanced
        ):
            self._finalize_run()
            return
            
        self._finalize_run(success=True)

    def step1_parse_drugbank_enhanced(self):
        """Parse DrugBank with enhanced extraction."""
        self.logger.info("Using enhanced DrugBank parser...")
        
        df_drugs, df_therapeutic = parse_drugbank_xml(self.config['drugbank_xml'])
        
        # Log statistics
        self.logger.info(f"Extracted {len(df_drugs)} drugs")
        self.logger.info(f"Extracted {len(df_therapeutic)} therapeutic relations")
        self.logger.info(
            f"Unique diseases in therapeutic data: "
            f"{df_therapeutic['disease_name'].nunique()}"
        )
        
        # Save files
        df_drugs.to_csv(self.config['drugbank_drugs'], index=False)
        df_therapeutic.to_csv(self.config['drugbank_therapeutic'], index=False)

    def step2_standardize_diseases(self):
        """Standardize disease names."""
        side_effects_df = pd.read_csv(self.config['offside_raw'])
        therapeutic_df = pd.read_csv(self.config['drugbank_therapeutic'])
        
        # Apply standardization
        side_effects_enhanced, therapeutic_enhanced = enhance_disease_mapping(
            side_effects_df, therapeutic_df
        )
        
        # Save enhanced files
        side_effects_enhanced.to_csv(self.config['offside_enhanced'], index=False)
        therapeutic_enhanced.to_csv(
            self.config['drugbank_therapeutic_enhanced'], 
            index=False
        )
        
        # Log statistics
        self.logger.info(
            f"Standardized {len(side_effects_enhanced)} side effects"
        )
        self.logger.info(
            f"Standardized {len(therapeutic_enhanced)} therapeutic relations"
        )

    def step3_build_dataset_enhanced(self):
        """Build dataset with enhanced balancing."""
        self.logger.info("Building dataset with enhanced balance algorithm...")
        
        builder = BalancedDatasetBuilderV2(
            n_drugs=self.config['n_drugs'],
            n_diseases=self.config['n_diseases'],
            target_sparsity=self.config['target_sparsity'],
            treatment_ratio=self.config['treatment_ratio'],
            target_disease_overlap=self.config['target_disease_overlap'],
            min_treatment_diseases_ratio=self.config.get(
                'min_treatment_diseases_ratio', 0.4
            )
        )
        
        builder.create_balanced_dataset(
            drugs_file=self.config['drugbank_drugs'],
            therapeutic_file=self.config['drugbank_therapeutic_enhanced'],
            side_effects_file=self.config['offside_enhanced'],
            output_dir=self.config['output_dir']
        )

    def _finalize_run(self, success=False):
        """Finalize the pipeline run."""
        total_time = time.time() - self.start_time
        self.logger.info("\n--- Enhanced Drug Repositioning Pipeline: END ---")
        self.logger.info(f"Final Status: {'SUCCESS' if success else 'FAILED'}")
        self.logger.info(f"Total execution time: {total_time:.2f} seconds")
        self.logger.info(f"Step summary: {json.dumps(self.status, indent=2)}")
        
        if success:
            self.logger.info(
                f"\nRun analysis script to check final dataset statistics:\n"
                f"python analyze.py"
            )

if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run enhanced pipeline
    pipeline = EnhancedDrugRepositioningPipeline()
    pipeline.run()
