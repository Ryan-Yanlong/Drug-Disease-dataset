# Drug Repositioning Dataset Generator

A comprehensive pipeline for generating drug-disease interaction datasets from DrugBank and OFFSIDES for drug repositioning research.

## Overview

This toolkit processes raw drug data from multiple sources to create structured datasets suitable for machine learning applications in drug repositioning. It handles:

- **DrugBank XML parsing** to extract drug information and therapeutic relationships
- **OFFSIDES processing** to standardize side effect data
- **Disease name standardization** using MedDRA mapping
- **Balanced dataset creation** with configurable drug/disease counts and sparsity

## Features

- Modular architecture with separate components for each processing step
- Comprehensive logging and error handling
- Disease name standardization and mapping
- Importance scoring for drug/disease selection
- Configurable balanced dataset generation
- Support for both full and subset datasets

## Requirements

```bash
pip install pandas numpy tqdm lxml
```

## Project Structure

```
drug_repositioning_dataset/
├── drugbank_parser.py       # DrugBank XML parser
├── offsides_processor.py    # OFFSIDES data processor
├── dataset_optimizer.py     # Balanced dataset creator
├── run_pipeline.py         # Main pipeline orchestrator
└── README.md              # This file
```

## Usage

### Quick Start

Run the complete pipeline:

```bash
python run_pipeline.py \
    --drugbank-xml drugbank.xml \
    --offsides-file offsides.csv \
    --output-dir datasets
```

### Detailed Options

```bash
python run_pipeline.py \
    --drugbank-xml drugbank.xml \          # Path to DrugBank XML
    --offsides-file offsides.csv \         # Path to OFFSIDES CSV
    --output-dir datasets \                 # Output directory
    --n-drugs 1000 \                       # Number of drugs in balanced dataset
    --n-diseases 1000 \                    # Number of diseases in balanced dataset
    --sparsity 0.98 \                      # Target sparsity (0.98 = 2% filled)
    --treatment-ratio 0.2                   # Ratio of treatments vs side effects
```

### Running Individual Components

#### 1. Parse DrugBank Only

```bash
python drugbank_parser.py drugbank.xml --output-dir data/
```

#### 2. Process OFFSIDES Only

```bash
python offsides_processor.py \
    --drugs-file data/drugbank_drugs.csv \
    --therapeutic-file data/drugbank_therapeutic.csv \
    --side-effects-file offsides.csv \
    --output-dir data/
```

#### 3. Create Balanced Dataset Only

```bash
python dataset_optimizer.py \
    --drugs-file data/drugbank_drugs.csv \
    --therapeutic-file data/drugbank_therapeutic.csv \
    --side-effects-file data/offsides_enhanced.csv \
    --n-drugs 1000 \
    --n-diseases 1000
```

## Output Files

### Full Dataset (`full_dataset/`)
- `drugs.csv` - All drugs with SMILES and descriptions
- `diseases.csv` - All diseases with unique IDs
- `drug_disease_interactions.csv` - All drug-disease relationships
- `interactions_sparse.csv` - Sparse matrix format
- `dataset_metadata.json` - Dataset statistics

### Balanced Dataset (`balanced_dataset/`)
- Same structure as full dataset but with selected top drugs/diseases
- `dataset_summary.json` - Detailed statistics including degree distributions

## Data Format

### drugs.csv
```csv
drug_id,name,smiles,description
DB00001,Lepirudin,CC[C@H](C)...,"Lepirudin is a..."
```

### diseases.csv
```csv
disease_id,disease_name
0,hypertension
1,diabetes
```

### drug_disease_interactions.csv
```csv
drug_id,disease_id,disease_name,relation_type,label,confidence
DB00001,0,hypertension,treatment,1,1.0
DB00001,5,nausea,side_effect,-1,0.9
```

### interactions_sparse.csv
```csv
drug_id,disease_id,label,confidence
DB00001,0,1,1.0
DB00001,5,-1,0.9
```

## Dataset Statistics

The pipeline provides comprehensive statistics including:

- Matrix dimensions and sparsity
- Number of therapeutic vs side effect relationships
- Drug/disease coverage percentages
- Degree distributions
- Overlap statistics (entities appearing in both categories)

## Disease Mapping

The toolkit includes sophisticated disease name standardization:

- MedDRA term mapping (e.g., "blood pressure increased" → "hypertension")
- Pattern-based transformations
- Pluralization handling
- Synonym resolution

## Customization

### Adding New Disease Mappings

Edit the `_create_meddra_mapping()` method in `offsides_processor.py`:

```python
def _create_meddra_mapping(self) -> Dict[str, str]:
    return {
        'your_meddra_term': 'standardized_name',
        # Add more mappings...
    }
```

### Adjusting Importance Scoring

Modify the `_calculate_importance_scores()` method in `dataset_optimizer.py` to change how drugs/diseases are prioritized:

```python
# Default weights
therapeutic_weight = 10  # Higher priority
side_effect_weight = 1   # Lower priority
overlap_bonus = 20       # Bonus for appearing in both
```

## Logging

All operations are logged with timestamps. Log files are created as:
- `pipeline_YYYYMMDD_HHMMSS.log` - Complete pipeline log
- Console output for real-time monitoring

## Performance Considerations

- DrugBank parsing is memory-intensive for large XML files
- Use iterative parsing to handle large datasets
- Balanced dataset creation uses sampling for efficiency
- All operations include progress bars for monitoring

## Troubleshooting

### Common Issues

1. **Memory errors during XML parsing**
   - The parser uses iterative parsing to minimize memory usage
   - Ensure sufficient RAM for your DrugBank version

2. **Low disease overlap**
   - Check OFFSIDES disease names match DrugBank terminology
   - Adjust disease mapping rules as needed

3. **Insufficient relationships for balanced dataset**
   - Lower the target drug/disease counts
   - Adjust the treatment ratio parameter

## Citation

If you use this dataset generator in your research, please cite:

```bibtex
@software{drug_repositioning_dataset,
  title = {Drug Repositioning Dataset Generator},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/drug-repositioning-dataset}
}
```

## License

This project is licensed under the MIT License. Note that DrugBank and OFFSIDES data may have their own licensing requirements.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments

- DrugBank for comprehensive drug information
- OFFSIDES/FAERS for side effect data
- MedDRA for medical terminology standardization
