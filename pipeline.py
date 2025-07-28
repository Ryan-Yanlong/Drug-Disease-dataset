"""
Drug Repositioning Dataset Pipeline
Orchestrates the complete dataset generation process from raw data to balanced matrices
"""

import os
import argparse
import logging
from datetime import datetime

# Import our modules
from drugbank_parser import DrugBankParser
from offsides_processor import OFFSIDESProcessor, DatasetCreator
from dataset_optimizer import BalancedDatasetOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatasetPipeline:
    """Main pipeline for drug repositioning dataset generation"""
    
    def __init__(self, config):
        self.config = config
        self.intermediate_dir = config.get('intermediate_dir', 'intermediate_data')
        self.output_dir = config.get('output_dir', 'final_dataset')
        
        # Create directories
        os.makedirs(self.intermediate_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run(self):
        """Run the complete pipeline"""
        logger.info("="*80)
        logger.info("Starting Drug Repositioning Dataset Generation Pipeline")
        logger.info("="*80)
        
        # Step 1: Parse DrugBank
        if self.config.get('parse_drugbank', True):
            self._parse_drugbank()
        
        # Step 2: Process OFFSIDES
        if self.config.get('process_offsides', True):
            self._process_offsides()
        
        # Step 3: Create full dataset
        if self.config.get('create_full_dataset', True):
            self._create_full_dataset()
        
        # Step 4: Create balanced dataset
        if self.config.get('create_balanced_dataset', True):
            self._create_balanced_dataset()
        
        logger.info("\n" + "="*80)
        logger.info("Pipeline completed successfully!")
        logger.info("="*80)
    
    def _parse_drugbank(self):
        """Parse DrugBank XML file"""
        logger.info("\n" + "-"*60)
        logger.info("STEP 1: Parsing DrugBank XML")
        logger.info("-"*60)
        
        parser = DrugBankParser()
        drugs_df, therapeutic_df = parser.parse_drugbank_xml(self.config['drugbank_xml'])
        
        # Save results
        drugs_file = os.path.join(self.intermediate_dir, 'drugbank_drugs.csv')
        therapeutic_file = os.path.join(self.intermediate_dir, 'drugbank_therapeutic.csv')
        
        drugs_df.to_csv(drugs_file, index=False)
        therapeutic_df.to_csv(therapeutic_file, index=False)
        
        logger.info(f"DrugBank data saved to {self.intermediate_dir}")
        
        # Update config for next steps
        self.config['drugs_file'] = drugs_file
        self.config['therapeutic_file'] = therapeutic_file
    
    def _process_offsides(self):
        """Process OFFSIDES side effects data"""
        logger.info("\n" + "-"*60)
        logger.info("STEP 2: Processing OFFSIDES Data")
        logger.info("-"*60)
        
        # Load data
        import pandas as pd
        therapeutic_df = pd.read_csv(self.config['therapeutic_file'])
        side_effects_df = pd.read_csv(self.config['offsides_file'])
        
        # Process side effects
        processor = OFFSIDESProcessor()
        enhanced_side_effects = processor.enhance_side_effects(side_effects_df, therapeutic_df)
        
        # Save enhanced side effects
        enhanced_file = os.path.join(self.intermediate_dir, 'offsides_enhanced.csv')
        enhanced_side_effects.to_csv(enhanced_file, index=False)
        
        logger.info(f"Enhanced side effects saved to {enhanced_file}")
        
        # Update config
        self.config['side_effects_file'] = enhanced_file
    
    def _create_full_dataset(self):
        """Create full drug repositioning dataset"""
        logger.info("\n" + "-"*60)
        logger.info("STEP 3: Creating Full Dataset")
        logger.info("-"*60)
        
        # Load data
        import pandas as pd
        drugs_df = pd.read_csv(self.config['drugs_file'])
        therapeutic_df = pd.read_csv(self.config['therapeutic_file'])
        side_effects_df = pd.read_csv(self.config['side_effects_file'])
        
        # Create dataset
        creator = DatasetCreator()
        full_output_dir = os.path.join(self.output_dir, 'full_dataset')
        
        creator.create_final_dataset(
            drugs_df, therapeutic_df, side_effects_df, full_output_dir
        )
        
        logger.info(f"Full dataset saved to {full_output_dir}")
    
    def _create_balanced_dataset(self):
        """Create balanced subset of the dataset"""
        logger.info("\n" + "-"*60)
        logger.info("STEP 4: Creating Balanced Dataset")
        logger.info("-"*60)
        
        # Create optimizer
        optimizer = BalancedDatasetOptimizer(
            n_drugs=self.config.get('n_drugs', 1000),
            n_diseases=self.config.get('n_diseases', 1000),
            target_sparsity=self.config.get('target_sparsity', 0.98),
            treatment_ratio=self.config.get('treatment_ratio', 0.2)
        )
        
        # Create balanced dataset
        balanced_output_dir = os.path.join(self.output_dir, 'balanced_dataset')
        
        optimizer.create_balanced_dataset(
            drugs_file=self.config['drugs_file'],
            therapeutic_file=self.config['therapeutic_file'],
            side_effects_file=self.config['side_effects_file'],
            output_dir=balanced_output_dir
        )
        
        logger.info(f"Balanced dataset saved to {balanced_output_dir}")


def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(
        description='Generate drug repositioning dataset from DrugBank and OFFSIDES'
    )
    
    # Input files
    parser.add_argument('--drugbank-xml', required=True, 
                       help='Path to DrugBank XML file')
    parser.add_argument('--offsides-file', required=True, 
                       help='Path to OFFSIDES CSV file')
    
    # Output directories
    parser.add_argument('--output-dir', default='drug_repositioning_datasets',
                       help='Output directory for final datasets')
    parser.add_argument('--intermediate-dir', default='intermediate_data',
                       help='Directory for intermediate files')
    
    # Pipeline options
    parser.add_argument('--skip-drugbank', action='store_true',
                       help='Skip DrugBank parsing (use existing files)')
    parser.add_argument('--skip-offsides', action='store_true',
                       help='Skip OFFSIDES processing (use existing files)')
    parser.add_argument('--skip-full', action='store_true',
                       help='Skip full dataset creation')
    parser.add_argument('--skip-balanced', action='store_true',
                       help='Skip balanced dataset creation')
    
    # Balanced dataset parameters
    parser.add_argument('--n-drugs', type=int, default=1000,
                       help='Number of drugs in balanced dataset')
    parser.add_argument('--n-diseases', type=int, default=1000,
                       help='Number of diseases in balanced dataset')
    parser.add_argument('--sparsity', type=float, default=0.98,
                       help='Target sparsity for balanced dataset')
    parser.add_argument('--treatment-ratio', type=float, default=0.2,
                       help='Ratio of treatment relationships in balanced dataset')
    
    # If using existing files
    parser.add_argument('--drugs-file',
                       help='Path to existing drugs CSV (if skipping DrugBank)')
    parser.add_argument('--therapeutic-file',
                       help='Path to existing therapeutic CSV (if skipping DrugBank)')
    parser.add_argument('--side-effects-file',
                       help='Path to existing side effects CSV (if skipping OFFSIDES)')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'drugbank_xml': args.drugbank_xml,
        'offsides_file': args.offsides_file,
        'output_dir': args.output_dir,
        'intermediate_dir': args.intermediate_dir,
        'parse_drugbank': not args.skip_drugbank,
        'process_offsides': not args.skip_offsides,
        'create_full_dataset': not args.skip_full,
        'create_balanced_dataset': not args.skip_balanced,
        'n_drugs': args.n_drugs,
        'n_diseases': args.n_diseases,
        'target_sparsity': args.sparsity,
        'treatment_ratio': args.treatment_ratio,
    }
    
    # Add existing files if provided
    if args.drugs_file:
        config['drugs_file'] = args.drugs_file
    if args.therapeutic_file:
        config['therapeutic_file'] = args.therapeutic_file
    if args.side_effects_file:
        config['side_effects_file'] = args.side_effects_file
    
    # Validate configuration
    if args.skip_drugbank and (not args.drugs_file or not args.therapeutic_file):
        parser.error("When skipping DrugBank parsing, must provide --drugs-file and --therapeutic-file")
    
    if args.skip_offsides and not args.side_effects_file:
        parser.error("When skipping OFFSIDES processing, must provide --side-effects-file")
    
    # Run pipeline
    pipeline = DatasetPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
