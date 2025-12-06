# Reproducibility Guide

This guide provides step-by-step instructions to reproduce all results from the research on stylometric analysis of human-generated and LLM-generated texts using statistical and fuzzy logic methods.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Data Collection](#step-1-data-collection)
3. [Step 2: Data Preprocessing](#step-2-data-preprocessing)
4. [Step 3: Feature Extraction](#step-3-feature-extraction)
5. [Step 4: Exploratory Data Analysis](#step-4-exploratory-data-analysis)
6. [Step 5: Model Training](#step-5-model-training)
7. [Step 6: Compile Papers](#step-6-compile-papers)
8. [Expected Results](#expected-results)
9. [Troubleshooting](#troubleshooting)
10. [File Structure](#file-structure)

---

## Prerequisites

### System Requirements

- Python 3.8 or higher
- At least 16 GB RAM (32 GB recommended for full dataset processing)
- At least 50 GB free disk space
- LaTeX distribution (TeX Live 2020 or later) for compiling papers

### Python Packages

Install all required packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

**Core packages:**
- pandas >= 2.1.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- scipy >= 1.11.0
- spacy >= 3.7.0
- seaborn >= 0.12.0

**Additional NLP packages:**
- transformers >= 4.35.0
- datasets >= 2.14.0
- nltk >= 3.8.0

**Jupyter environment:**
- jupyter >= 1.0.0
- ipykernel >= 6.25.0
- notebook >= 7.0.0

### Install Portuguese Language Model

For text processing and analysis:

```bash
python -m spacy download pt_core_news_sm
```

### LaTeX Requirements

For compiling the papers, ensure you have the following LaTeX packages installed:

- abntex2cite (Brazilian ABNT citation style)
- babel (with Brazilian Portuguese support)
- amsmath, amssymb
- graphicx
- booktabs
- hyperref

On Ubuntu/Debian:
```bash
sudo apt-get install texlive-full texlive-lang-portuguese
```

On macOS with MacTeX:
```bash
brew install --cask mactex
```

---

## Step 1: Data Collection

### Data Sources

The project uses the following datasets:

1. **ShareGPT Portuguese** (`data/sharegpt-portuguese.json`)
   - Source: LLM-generated conversations in Portuguese
   - Format: JSON
   - Size: ~36 MB

2. **IMDB Reviews (Portuguese Translation)** (`data/imdb-reviews-pt-br.csv`)
   - Source: Machine-translated movie reviews
   - Format: CSV
   - Size: ~122 MB
   - Columns: id, text_pt, text_en, sentiment

3. **BoolQ Dataset** (`data/boolq.csv` and `data/validation_bool.csv`)
   - Source: Human-written passages in Portuguese
   - Format: CSV
   - Combined size: ~8.1 MB
   - Columns: question, passage, answer

4. **BrWaC Corpus** (`data/brwac/*.parquet`)
   - Source: Brazilian Web as Corpus
   - Format: Parquet files (multiple)
   - Contains human-written Portuguese web texts
   - Structured as: {'paragraphs': [['text1'], ['text2'], ...]}

5. **Canarim Dataset** (`data/canarim/*.parquet`)
   - Source: LLM-generated outputs
   - Format: Parquet files (multiple)
   - Contains machine-generated Portuguese texts
   - Columns include 'output' field

### Obtaining the Data

Place all datasets in the `data/` directory with the following structure:

```
data/
├── sharegpt-portuguese.json
├── imdb-reviews-pt-br.csv
├── boolq.csv
├── validation_bool.csv
├── brwac/
│   └── *.parquet
└── canarim/
    └── *.parquet
```

**Note:** The BrWaC and Canarim datasets are large collections stored in Parquet format. Ensure you have sufficient disk space before downloading.

---

## Step 2: Data Preprocessing

Data preprocessing combines all sources into unified datasets with balanced and unbalanced versions.

### Run the Preprocessing Notebook

```bash
jupyter notebook "0. process_data.ipynb"
```

Or execute it non-interactively:

```bash
jupyter nbconvert --to notebook --execute "0. process_data.ipynb"
```

### What This Step Does

1. **Loads all data sources:**
   - Reads JSON, CSV, and Parquet files
   - Extracts text and labels from each source
   - Normalizes labels ('gpt' → 'llm', consistent labeling)

2. **Text filtering and chunking:**
   - Removes texts shorter than 100 characters
   - Splits texts longer than 10,000 characters into chunks
   - Uses intelligent chunking at sentence boundaries
   - Processes data in batches to manage memory

3. **Creates combined dataset:**
   - Concatenates all sources
   - Saves to `combined.csv` (~8.7 GB)

4. **Creates balanced dataset:**
   - Downsamples majority class (human texts)
   - Upsamples minority class (LLM texts) if needed
   - Target ratio: 30% of combined dataset size per class
   - Saves to `balanced.csv` (~3.2 GB)

5. **Additional processed dataset:**
   - Filtered and chunked version
   - Saves to `processed_filtered_chunked_batch.csv` (~8.4 GB)

### Expected Outputs

After this step, you should have:

- `combined.csv` - Full combined dataset
- `balanced.csv` - Balanced version for training
- `processed_filtered_chunked_batch.csv` - Processed with length filtering

### Key Processing Parameters

- **Minimum text length:** 100 characters
- **Maximum text length:** 10,000 characters (before chunking)
- **Chunk overlap:** 0 characters
- **Batch size:** 50,000 rows (adjustable based on available RAM)

---

## Step 3: Feature Extraction

Extract stylometric features from the preprocessed texts using the `features.py` module.

### Run Feature Extraction

For the balanced dataset (recommended for initial experiments):

```bash
python src/features.py \
    --input balanced_sample_100k.csv \
    --output features_100k.csv \
    --text-col text \
    --lang pt
```

For the full balanced dataset:

```bash
python src/features.py \
    --input balanced.csv \
    --output features_balanced.csv \
    --text-col text \
    --lang pt
```

### Command Arguments

- `--input`: Path to input CSV file containing text data
- `--output`: Path to save extracted features
- `--text-col`: Name of column containing text (default: "text")
- `--lang`: Language code - use "pt" for Portuguese, "en" for English

### Extracted Features

The feature extraction computes the following metrics:

1. **Sentence Statistics:**
   - `sent_mean`: Mean sentence length (in tokens)
   - `sent_std`: Standard deviation of sentence lengths
   - `sent_burst`: Burstiness (ratio of std to mean)

2. **Lexical Diversity:**
   - `ttr`: Type-token ratio
   - `herdan_c`: Herdan's C (logarithmic variant of TTR)
   - `hapax_prop`: Proportion of words appearing exactly once

3. **Character-Level:**
   - `char_entropy`: Shannon entropy of character distribution

4. **Linguistic Markers:**
   - `func_word_ratio`: Proportion of function words (language-specific)
   - `first_person_ratio`: Proportion of first-person pronouns
   - `bigram_repeat_ratio`: Proportion of repeated bigrams

5. **Readability:**
   - `fk_grade`: Flesch-Kincaid grade level (English only; 0.0 for Portuguese)

### Expected Output

A CSV file with all original columns plus the extracted feature columns. Each row represents one text sample with its computed features.

---

## Step 4: Exploratory Data Analysis

Perform comprehensive EDA to understand the data characteristics and differences between human and LLM texts.

### Run EDA Notebook

```bash
jupyter notebook EDA.ipynb
```

Or execute non-interactively:

```bash
jupyter nbconvert --to notebook --execute EDA.ipynb
```

### What This Step Does

1. **Dataset Overview:**
   - Sample loading and examination
   - Label distribution analysis
   - Text example display

2. **Text Feature Analysis:**
   - Calculates length, word count, sentence count
   - Computes average word length and punctuation counts
   - Statistical comparison by label

3. **Visualizations:**
   - Box plots comparing features across labels
   - Distribution histograms
   - Correlation heatmaps
   - Word frequency comparisons

4. **Vocabulary Analysis:**
   - Vocabulary size comparison
   - Type-token ratio analysis
   - Most common words (with lemmatization)
   - Stopword removal using spaCy

5. **Statistical Testing:**
   - Mann-Whitney U tests (non-parametric)
   - Welch's t-tests
   - Kolmogorov-Smirnov tests
   - Cohen's d effect size calculation

6. **Large Dataset Processing:**
   - Memory-efficient chunk-based processing
   - Comprehensive statistics from full dataset

### Expected Outputs

- Visualization plots saved as PNG files
- `eda_results_for_paper.json` - Comprehensive results in JSON format
- Statistical test results displayed in notebook
- Summary tables for paper

---

## Step 5: Model Training

Train statistical and fuzzy classification models on the extracted features.

### 5.1 Statistical Models (LDA and Logistic Regression)

Run PCA for dimensionality reduction and visualization:

```bash
python src/models.py pca \
    --features features_100k.csv \
    --label-col label \
    --n-components 2 \
    --out pca_scores.csv \
    --plot pca_scatter.png
```

Train and evaluate classifiers with cross-validation:

```bash
python src/models.py classify \
    --features features_100k.csv \
    --label-col label \
    --topic-col topic \
    --n-splits 5 \
    --roc-out roc_results.pkl \
    --pr-out pr_results.pkl
```

### 5.2 Fuzzy Classifier

The fuzzy classifier is implemented in `src/fuzzy.py` and can be used programmatically:

```python
from fuzzy import FuzzyClassifier
import pandas as pd

# Load features
df = pd.read_csv('features_100k.csv')

# Split into train/test
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Create and train fuzzy classifier
fuzzy_clf = FuzzyClassifier(pos_label='human', neg_label='llm')
fuzzy_clf.fit(train_df, label_col='label')

# Predict
predictions = fuzzy_clf.predict(test_df)
probabilities = fuzzy_clf.predict_proba(test_df)

# Evaluate
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(test_df['label'], predictions))
print(f"ROC AUC: {roc_auc_score(test_df['label'] == 'human', probabilities[:, 1]):.4f}")
```

### Model Parameters

**Statistical Models:**
- Cross-validation: 5-fold GroupKFold (by topic) or StratifiedKFold
- Standardization: StandardScaler (zero mean, unit variance)
- LDA: Default scikit-learn parameters
- Logistic Regression: max_iter=1000

**Fuzzy Classifier:**
- Membership functions: Triangular
- Quantiles used: 33%, 50%, 66%
- Orientation: Learned from median comparison
- Aggregation: Mean of membership degrees

### Expected Outputs

- `pca_scores.csv` - Principal component scores
- `pca_scatter.png` - 2D visualization of PC1 vs PC2
- `roc_results.pkl` - ROC curve data for each fold and classifier
- `pr_results.pkl` - Precision-recall curve data
- Model performance metrics printed to console

---

## Step 6: Compile Papers

Compile the LaTeX papers documenting the statistical and fuzzy approaches.

### 6.1 Compile Statistical Paper

Navigate to the statistical paper directory:

```bash
cd paper_stat
```

Compile using pdflatex (run twice for references):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk for automatic compilation:

```bash
latexmk -pdf main.tex
```

### 6.2 Compile Fuzzy Paper

Navigate to the fuzzy paper directory:

```bash
cd ../paper_fuzzy
```

Compile the same way:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or:

```bash
latexmk -pdf main.tex
```

### Compilation Notes

- The first `pdflatex` run generates auxiliary files
- `bibtex` processes citations (ABNT format)
- Second and third `pdflatex` runs resolve references and citations
- Warnings about missing fonts or packages should be addressed by installing missing LaTeX packages

### Expected Outputs

- `paper_stat/main.pdf` - Statistical analysis paper
- `paper_fuzzy/main.pdf` - Fuzzy logic classification paper
- Various auxiliary files (.aux, .log, .bbl, .blg, .out)

---

## Expected Results

### Data Statistics

**Combined Dataset:**
- Total samples: ~69 million rows (processed version)
- Human texts: ~98.4%
- LLM texts: ~1.6%

**Balanced Dataset:**
- Approximately 1.3 million samples
- 50/50 split between human and LLM texts

### Feature Statistics (Sample Analysis)

| Feature | Human Mean | LLM Mean | Cohen's d | Effect Size | p-value |
|---------|------------|----------|-----------|-------------|---------|
| Length | 3734.3 | 546.0 | 1.018 | Large | < 0.001 |
| Word Count | 595.9 | 92.7 | 1.000 | Large | < 0.001 |
| Sentence Count | 31.8 | 5.9 | 0.824 | Large | < 0.001 |
| Avg Word Length | 5.3 | 9.1 | -0.835 | Large | < 0.001 |
| Punctuation Count | 89.4 | 11.6 | 0.926 | Large | < 0.001 |

**Key Findings:**
- All 5 features show statistically significant differences (p < 0.001)
- Effect sizes are uniformly large (|Cohen's d| > 0.8)
- Human texts are generally longer with more sentences and punctuation
- LLM texts show higher average word length (possibly due to different vocabulary)

### Vocabulary Analysis

- Human vocabulary: ~36,455 unique words
- LLM vocabulary: ~2,081 unique words
- Human type-token ratio: 0.0927
- LLM type-token ratio: 0.3347
- Top 100 words overlap: 15% (indicating distinct vocabularies)

### Model Performance

**Expected AUC Scores:**

Based on the feature separability and effect sizes, expected performance:

- **Logistic Regression:** AUC > 0.85
- **Linear Discriminant Analysis (LDA):** AUC > 0.80
- **Fuzzy Classifier:** AUC > 0.75

**Cross-Validation:**
- 5-fold cross-validation with topic-based splitting
- Consistent performance across folds
- Minimal overfitting due to clear feature separation

### PCA Results

- First two principal components capture majority of variance
- Clear separation between human and LLM clusters in PC space
- PC1 primarily captures text length and complexity
- PC2 captures linguistic style differences

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Errors During Data Processing

**Problem:** `MemoryError` or system freeze when processing large files.

**Solutions:**
- Reduce batch size in `0. process_data.ipynb`:
  ```python
  batch_size = 25000  # instead of 50000
  ```
- Close other applications to free RAM
- Process smaller samples first to verify workflow
- Use swap space if available

#### 2. Missing Data Files

**Problem:** `FileNotFoundError` when running scripts.

**Solutions:**
- Verify all data files are in the `data/` directory
- Check file names match exactly (case-sensitive)
- Ensure Parquet files are in correct subdirectories
- Download missing datasets from original sources

#### 3. LaTeX Compilation Errors

**Problem:** Missing packages or compilation failures.

**Solutions:**
- Install complete LaTeX distribution:
  ```bash
  sudo apt-get install texlive-full texlive-lang-portuguese
  ```
- Update LaTeX package manager:
  ```bash
  tlmgr update --self --all
  ```
- Check for specific missing packages in .log file
- Install Brazilian Portuguese support specifically

#### 4. Feature Extraction Takes Too Long

**Problem:** Feature extraction is very slow on large datasets.

**Solutions:**
- Start with a smaller sample:
  ```bash
  head -100000 balanced.csv > balanced_sample.csv
  python src/features.py --input balanced_sample.csv --output features_sample.csv --lang pt
  ```
- Process in batches programmatically
- Use multiprocessing for parallel feature extraction

#### 5. Import Errors

**Problem:** `ModuleNotFoundError` or import failures.

**Solutions:**
- Ensure virtual environment is activated
- Reinstall requirements:
  ```bash
  pip install -r requirements.txt --upgrade
  ```
- Install spaCy language model:
  ```bash
  python -m spacy download pt_core_news_sm
  ```
- Check Python version (3.8+)

#### 6. Encoding Issues

**Problem:** `UnicodeDecodeError` when reading files.

**Solutions:**
- Ensure files are UTF-8 encoded
- Specify encoding explicitly:
  ```python
  pd.read_csv('file.csv', encoding='utf-8')
  ```
- Use `errors='ignore'` or `errors='replace'` as fallback

#### 7. Jupyter Kernel Crashes

**Problem:** Kernel dies during notebook execution.

**Solutions:**
- Reduce data loading size
- Clear output cells before execution
- Restart kernel and clear all outputs
- Increase available memory
- Process data in smaller chunks

#### 8. Cross-Validation Errors

**Problem:** Not enough samples in groups for GroupKFold.

**Solutions:**
- Ensure topic column exists and has sufficient variety
- Use StratifiedKFold instead if topics are not available
- Reduce n_splits to 3 if dataset is small

#### 9. Fuzzy Classifier Returns All Same Predictions

**Problem:** Classifier predicts only one class.

**Solutions:**
- Check training data has both classes
- Verify features are not all zeros or NaN
- Ensure feature scaling/normalization if needed
- Check orientation calculation in membership functions

#### 10. Disk Space Issues

**Problem:** Not enough space for intermediate files.

**Solutions:**
- Clean intermediate results from `0. process_data.ipynb`
- Remove old CSV files after feature extraction
- Compress or archive datasets not in active use
- Use external storage for large Parquet files

---

## File Structure

```
prob_est/
├── data/                                   # Raw data directory
│   ├── sharegpt-portuguese.json           # LLM conversations
│   ├── imdb-reviews-pt-br.csv             # Translated reviews
│   ├── boolq.csv                          # Human passages
│   ├── validation_bool.csv                # More human passages
│   ├── brwac/                             # Human web corpus
│   │   └── *.parquet
│   └── canarim/                           # LLM outputs
│       └── *.parquet
│
├── src/                                   # Source code
│   ├── features.py                        # Feature extraction module
│   ├── models.py                          # Statistical classifiers (LDA, LogReg)
│   └── fuzzy.py                           # Fuzzy logic classifier
│
├── docs/                                  # Documentation
│   └── REPRODUCIBILITY.md                 # This file
│
├── paper_stat/                            # Statistical analysis paper
│   ├── main.tex                           # Main LaTeX file
│   ├── sections/                          # Paper sections
│   │   ├── introduction.tex
│   │   ├── methods.tex
│   │   ├── results.tex
│   │   └── discussion.tex
│   ├── references.bib                     # Bibliography
│   └── main.pdf                           # Compiled PDF
│
├── paper_fuzzy/                           # Fuzzy logic paper
│   ├── main.tex                           # Main LaTeX file
│   ├── sections/                          # Paper sections
│   │   ├── introduction.tex
│   │   ├── methods.tex
│   │   ├── results.tex
│   │   └── discussion.tex
│   ├── references.bib                     # Bibliography
│   └── main.pdf                           # Compiled PDF
│
├── 0. process_data.ipynb                  # Data preprocessing notebook
├── EDA.ipynb                              # Exploratory data analysis notebook
│
├── combined.csv                           # Full combined dataset (~8.7 GB)
├── balanced.csv                           # Balanced dataset (~3.2 GB)
├── processed_filtered_chunked_batch.csv   # Processed dataset (~8.4 GB)
├── balanced_sample_100k.csv               # Sample for quick experiments
├── features_100k.csv                      # Extracted features from sample
├── pca_scores.csv                         # PCA results
├── roc_results.pkl                        # ROC curve data
├── pr_results.pkl                         # Precision-recall data
├── eda_results_for_paper.json            # EDA statistics export
│
├── requirements.txt                       # Python dependencies
└── README.md                              # Project overview
```

### Important File Descriptions

**Input Data:**
- `data/*.csv`, `data/*.json`, `data/*/*.parquet`: Original data sources

**Preprocessed Data:**
- `combined.csv`: All sources merged, unbalanced
- `balanced.csv`: Balanced version for training
- `processed_filtered_chunked_batch.csv`: Filtered and chunked

**Feature Data:**
- `features_100k.csv`: Features extracted from sample
- `features_balanced.csv`: Features from full balanced set

**Model Outputs:**
- `pca_scores.csv`: Principal component scores
- `*.pkl`: Serialized model results
- `*.json`: Analysis results for papers

**Papers:**
- `paper_stat/main.pdf`: Statistical approach paper
- `paper_fuzzy/main.pdf`: Fuzzy logic approach paper

**Code:**
- `src/features.py`: Stylometric feature extraction
- `src/models.py`: PCA, LDA, logistic regression
- `src/fuzzy.py`: Fuzzy membership and inference

**Notebooks:**
- `0. process_data.ipynb`: Data collection and preprocessing
- `EDA.ipynb`: Exploratory analysis and visualization

---

## Additional Notes

### Computational Resources

- **Recommended:** 16-32 GB RAM, 4+ CPU cores
- **Minimum:** 8 GB RAM, 2 CPU cores (use smaller samples)
- **Storage:** 50+ GB free space

### Processing Time Estimates

- Data preprocessing: 1-3 hours (depends on RAM and CPU)
- Feature extraction (100k samples): 10-30 minutes
- Feature extraction (full balanced): 2-6 hours
- EDA notebook: 15-45 minutes
- Model training: 5-15 minutes (100k samples)
- LaTeX compilation: 1-2 minutes per paper

### Parallelization

For faster processing, consider:
- Using Dask for large dataset operations
- Multiprocessing in feature extraction
- GPU acceleration for deep learning extensions (if applicable)

### Version Control

Results may vary slightly due to:
- Random seed in cross-validation
- Package version differences
- Operating system differences
- Floating point precision

To ensure reproducibility:
- Use the exact package versions in `requirements.txt`
- Set random seeds explicitly (random_state=42)
- Document any manual interventions

### Getting Help

If you encounter issues not covered in troubleshooting:

1. Check the error message carefully
2. Review the relevant notebook or script
3. Verify file paths and data availability
4. Consult package documentation
5. Check system resources (RAM, disk space)

---

**Last Updated:** 2025-12-06
**Version:** 1.0
**Maintainer:** Victor Lofgren Sattamini
