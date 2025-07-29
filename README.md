# Drug–Disease Repositioning Dataset Builder

This repository provides a fully automated pipeline to build a balanced drug–disease relationship dataset for drug repositioning research. It integrates data from DrugBank and OFFSIDES, applies advanced semantic standardization of disease names, samples with controlled sparsity and overlap, and offers comprehensive analysis and visualization tools.

## 🚀 Features

* **DrugBank Parsing**: Extract small‑molecule drugs and therapeutic indications from DrugBank XML (`drugbank_parser.py`).
* **Side‑Effect Preparation**: Clean and filter raw OFFSIDES data to match DrugBank drug names (`prepare_offside.py`).
* **Disease Standardization**: Hybrid semantic and fuzzy‑matching to normalize disease names across datasets (`disease_standardizer.py`).
* **Balanced Dataset Builder**: Stratified sampling to enforce desired sparsity, treatment‑side‑effect ratio, and disease overlap (`dataset_builder.py`).
* **Pipeline Orchestration**: Single‑entrypoint runner that ties all steps together with logging and configuration (`run_pipeline.py`).
* **Analysis & Visualization**: Generate summary statistics and plots to evaluate dataset properties (`analyze.py`).

## 📁 Repository Structure

```
├── drugbank_parser.py            # Enhanced DrugBank XML parser
├── prepare_offside.py            # OFFSIDES data preprocessing
├── disease_standardizer.py       # Disease name normalization module
├── dataset_builder.py            # Balanced dataset construction class
├── run_pipeline.py               # End‑to‑end pipeline orchestrator
├── analyze.py                    # Dataset analysis and visualization
├── config_v2.json                # Pipeline configuration (auto‑generated)
├── pipeline_logs/                # Automatic logs for each run
└── requirements.txt              # Python dependencies
```

## 📦 Data Sources & Citations

* **DrugBank** (Version 5.1.9): Comprehensive resource for drug data and interactions. Downloaded from [https://go.drugbank.com](https://go.drugbank.com) on July 1, 2025. ⬥ Wishart DS et al., DrugBank 5.0: a major update to the DrugBank database for 2018. *Nucleic Acids Res*. 2018;46(D1)\:D1074–D1082.

* **OFFSIDES** (Side‑effect data): High‑throughput side‑effect dataset by Tatonetti et al., available at [https://nsides.io](https://nsides.io). ⬥ Chandak P, Tatonetti NP. Using Machine Learning to Identify Adverse Drug Effects Posing Increased Risk to Women. Patterns (N Y). 2020 Oct 9;1(7):100108. doi: 10.1016/j.patter.2020.100108. Epub 2020 Sep 22. PMID: 33179017; PMCID: PMC7654817.

## 🏃‍♂️ Quick Start

1. **Prepare OFFSIDES data**

   ```bash
   python prepare_offside.py
   ```

   Outputs `offside_raw_side_effects.csv` for side‑effect relations.

2. **Run the full pipeline**

   ```bash
   python run_pipeline.py
   ```

   * Generates `config_v2.json` (if missing).
   * Steps:

     1. Parse DrugBank → `drugbank_drugs.csv`, `drugbank_therapeutic_relations.csv`
     2. Standardize diseases → `offside_enhanced_side_effects.csv`, `drugbank_therapeutic_enhanced.csv`
     3. Build balanced dataset → `balanced_drug_disease_dataset_v2/`

3. **Analyze the results**

   ```bash
   python analyze.py
   ```

   * Prints summary statistics and saves `dataset_analysis_report.txt` and distribution plots.

## 🔧 Configuration

* All pipeline parameters (e.g., number of drugs/diseases, sparsity, overlap ratios) can be adjusted in `config_v2.json`.
* Key settings:

  ```json
  {
    "n_drugs": 1000,
    "n_diseases": 1000,
    "target_sparsity": 0.98,
    "treatment_ratio": 0.2,
    "target_disease_overlap": 0.3,
    "min_treatment_diseases_ratio": 0.4
  ```



---


