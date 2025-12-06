# Data Pipeline Documentation

## Overview

This document provides comprehensive documentation of the data collection and preprocessing pipeline used in the LLM-generated text detection study. All data processing is performed in the `0. process_data.ipynb` notebook.

**Pipeline Summary:**
1. Load data from 5 distinct sources
2. Combine into single dataset
3. Filter by minimum length
4. Chunk long documents
5. Create balanced dataset via stratified sampling

---

## 1. Data Sources

### 1.1 ShareGPT-Portuguese (LLM Text)

- **File:** `data/sharegpt-portuguese.json`
- **Format:** JSON
- **Contains:** LLM-generated conversations in Portuguese
- **Label:** `llm` (originally labeled as `gpt`, renamed for consistency)
- **Processing Steps:**
  - Read JSON file with conversation data
  - Extract text from conversation structure
  - Drop `lang` column
  - Assign `llm` label to all entries
- **Code Reference:** Cell `77ce690d` in `0. process_data.ipynb`

### 1.2 IMDB Reviews PT-BR (LLM Translations)

- **File:** `data/imdb-reviews-pt-br.csv`
- **Format:** CSV
- **Contains:** Movie reviews machine-translated to Portuguese
- **Label:** `llm` (machine-translated text classified as LLM-generated)
- **Processing Steps:**
  - Load CSV file
  - Drop columns: `id`, `text_en`, `sentiment`
  - Rename `text_pt` column to `text`
  - Assign `llm` label to all entries
- **Code Reference:** Cell `b13140c3` in `0. process_data.ipynb`

### 1.3 BoolQ Dataset (Human Text)

- **Files:**
  - `data/boolq.csv`
  - `data/validation_bool.csv`
- **Format:** CSV
- **Contains:** Human-written passages in Portuguese for boolean question answering
- **Label:** `human`
- **Processing Steps:**
  - Load both training and validation CSV files
  - Extract `passage` column as text
  - Assign `human` label to all entries
  - Process both files separately, then combine
- **Code Reference:** Cells `968b117d` and `a9514170` in `0. process_data.ipynb`

### 1.4 BrWaC Corpus (Human Text)

- **Files:** `data/brwac/*.parquet` (21 parquet files)
- **Format:** Parquet
- **Contains:** Brazilian Web as Corpus - web-scraped Portuguese text from Brazilian websites
- **Label:** `human`
- **Size:** Largest data source in the pipeline
- **Processing Steps:**
  1. Iterate through all 21 parquet files in batches
  2. Process in batches of 100,000 rows to manage memory
  3. Extract nested JSON structure from `paragraphs` field
  4. Join paragraphs with newlines using `process_paragraphs_to_text()` function
  5. Assign `human` label to all entries
- **Memory Optimization:** Uses `pyarrow.parquet` for efficient batch processing
- **Code Reference:** Cell `48e5f9a1` in `0. process_data.ipynb`

**Processing Function:**
```python
def process_paragraphs_to_text(data_dict):
    """
    Process data in the format {'paragraphs': [['text1'], ['text2'], ...]}
    into a single text block.
    """
    if 'paragraphs' not in data_dict:
        raise ValueError("Data must contain 'paragraphs' key")

    all_text = []
    for paragraph in data_dict['paragraphs']:
        if isinstance(paragraph, list):
            paragraph_text = ' '.join(paragraph)
        else:
            paragraph_text = str(paragraph)
        all_text.append(paragraph_text)

    return '\n'.join(all_text)
```

### 1.5 Canarim Dataset (LLM Text)

- **Files:** `data/canarim/*.parquet`
- **Format:** Parquet
- **Contains:** LLM-generated outputs
- **Label:** `llm`
- **Processing Steps:**
  - Iterate through all parquet files in directory
  - Extract `output` column as text
  - Assign `llm` label to all entries
- **Code Reference:** Cell `2ee38d72` in `0. process_data.ipynb`

---

## 2. Data Combination

**Objective:** Combine all 5 data sources into a unified dataset

**Method:**
- Use `pd.concat()` to concatenate all DataFrames
- Reset index to create continuous row numbering
- No deduplication performed (assumes sources are distinct)

**Output:** `combined.csv`

**Initial Statistics:**
- Total samples: 2,331,317 rows
- Label distribution calculated and displayed
- Saved as intermediate file for reproducibility

**Code Reference:** Cell `8f7d7a8c` in `0. process_data.ipynb`

---

## 3. Text Filtering and Chunking

The preprocessing pipeline applies two critical transformations to ensure text samples are suitable for stylometric analysis: length-based filtering and chunking of long documents.

### 3.1 Length-Based Filtering

**Purpose:** Remove texts too short to provide meaningful stylometric features

**Parameters:**
- `min_length`: 100 characters
- **Note:** Function default is 200 characters, but executed with min_length=100 (line 21 of cell `a1070493`)
- Rationale: Stylometric features require sufficient text for reliable measurement

**Implementation:**
```python
df_filtered = df[df['text'].str.len() >= min_length].copy()
```

**Statistics (with min_length=100):**
- Original dataset: 2,331,317 rows
- Removed short texts: 171,510 entries (7.4%)
- Remaining after filtering: 2,159,807 rows

**Code Reference:** Function `filter_and_chunk_text_batch()` in cell `156b6a5f`; Execution call in cell `a1070493`

### 3.2 Text Chunking for Long Documents

**Purpose:** Split very long documents into analyzable chunks of uniform size

**Parameters:**
- `max_length`: 10,000 characters
- `chunk_overlap`: 0 characters (no overlap between chunks)
- Rationale: Very long texts may have inconsistent style across different sections; chunking provides multiple independent samples

**Algorithm:**

1. **Identify texts requiring chunking:**
   - Texts ≤ 10,000 chars: kept as-is
   - Texts > 10,000 chars: split into chunks

2. **Chunking procedure:**
   - Calculate end position: `end = start + max_length`
   - Search for natural break points (in order of preference):
     - `. ` (sentence end with space)
     - `.\n` (sentence end with newline)
     - `\n\n` (paragraph break)
     - ` ` (space/word boundary)
   - Break at the last occurrence of break character in the range `[start + max_length/2, end]`
   - If no good break point found, hard break at `max_length`

3. **Metadata preservation:**
   - Each chunk labeled with `original_length` (length of source document)
   - Each chunk assigned unique `chunk_id` in format `{original_index}_{chunk_number}`
   - Original label (`human` or `llm`) preserved for all chunks

**Statistics:**
- Texts within normal range: 1,992,995 (92.3%)
- Texts requiring chunking: 166,812 (7.7%)
- Total chunks created: ~69 million entries in final processed dataset

**Implementation:**
```python
def create_text_chunks_efficient(text, max_length, overlap):
    """
    Memory-efficient version of create_text_chunks.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)
    break_points = ['. ', '.\n', '\n\n', ' ']
    half_max = max_length // 2

    while start < text_length:
        end = min(start + max_length, text_length)

        if end < text_length:
            best_break = end
            for break_char in break_points:
                search_start = max(start + half_max, start)
                last_break = text.rfind(break_char, search_start, end)
                if last_break != -1:
                    best_break = last_break + len(break_char)
                    break

            chunk = text[start:best_break].strip()
            next_start = max(start + max_length - overlap, best_break - overlap)
        else:
            chunk = text[start:].strip()
            next_start = text_length

        if chunk:
            chunks.append(chunk)

        start = next_start if next_start > start else start + max_length - overlap

    return chunks
```

**Code Reference:** Functions `filter_and_chunk_text_batch()` and `create_text_chunks_efficient()` in cells `156b6a5f` and `e6dc727f`

**Output:** `processed_filtered_chunked_batch.csv`

### 3.3 Batch Processing for Memory Efficiency

**Challenge:** Processing 2.3M+ entries exceeds available memory for many systems

**Solution:** Batch processing with intermediate saves

**Parameters:**
- `batch_size`: 50,000 rows (adjustable based on available memory, can be reduced to 25,000)
- `intermediate_save_every`: 3 batches (saves progress every 150,000 rows)

**Process:**
1. Read input file in chunks using `pd.read_csv(chunksize=batch_size)`
2. Process each batch independently (filter + chunk)
3. Save intermediate results every N batches
4. Combine all intermediate results at the end
5. Clean up intermediate files

**Memory Optimization Features:**
- Uses generators to avoid loading entire dataset
- Explicit garbage collection after each batch
- Intermediate saves prevent data loss on crash
- Memory usage estimation function provided

**Code Reference:** Functions `filter_and_chunk_text_batch()`, `process_single_batch()`, and `estimate_memory_usage()` in cells `156b6a5f` and `e6dc727f`

---

## 4. Stratified Sampling and Balancing

**Purpose:** Create balanced dataset for unbiased classification

### 4.1 Class Imbalance Problem

After combining all sources, the dataset has unequal representation of human vs LLM texts. Class imbalance can:
- Artificially inflate accuracy metrics
- Bias classifiers toward majority class
- Make performance metrics unreliable

### 4.2 Balancing Strategy: Hybrid Approach

**Method:** Stratified sampling with downsampling and upsampling

**Parameters:**
- `target_ratio`: 0.3 (30% of combined dataset)
- Target balance: 50% human, 50% LLM (exact balance)
- Random seed: 42 (for reproducibility)

**Procedure:**

1. **Separate by class:**
   - Extract all `human` labeled samples
   - Extract all `llm` labeled samples

2. **Calculate target size:**
   ```python
   total_target = int((len(human_samples) + len(llm_samples)) * target_ratio)
   target_per_class = total_target // 2
   ```

3. **Downsample majority class (human):**
   ```python
   human_balanced = human_samples.sample(n=target_size, random_state=42)
   ```

4. **Upsample minority class (LLM) if needed:**
   ```python
   llm_upsampled = llm_samples.sample(n=target_size, replace=True, random_state=42)
   ```
   - Uses `replace=True` to allow sampling with replacement if needed
   - Ensures exact 50/50 balance

5. **Combine and shuffle:**
   ```python
   df_balanced = pd.concat([human_balanced, llm_upsampled], ignore_index=True)
   df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
   ```

**Output Files:**
- From original combined data: `balanced.csv`
- From processed/chunked data: `balanced_processed.csv`

**Final Statistics:**
- Total balanced samples: ~100,000 rows (varies based on source data)
- Human samples: 50% (exact)
- LLM samples: 50% (exact)
- Shuffled randomly to prevent ordering bias

**Code Reference:**
- Function `hybrid_balance()` in cell `7ca3542e`
- Batch version `create_balanced_dataset_batch()` in cell `b83e7e3a`

---

## 5. Pipeline Execution

### 5.1 Complete Pipeline Order

1. **Load Data Sources** (Cells: `77ce690d`, `b13140c3`, `968b117d`, `a9514170`, `48e5f9a1`, `2ee38d72`)
2. **Combine Datasets** (Cell: `8f7d7a8c`)
3. **Save Combined Dataset** (Cell: `1e222958`)
4. **Filter and Chunk** (Cell: `a1070493` - batch version)
5. **Create Balanced Dataset** (Cell: `7ca3542e` or batch equivalent)

### 5.2 Output Files

| Filename | Description | Approx. Size |
|----------|-------------|--------------|
| `combined.csv` | All 5 sources concatenated | 2.3M rows |
| `processed_filtered_chunked_batch.csv` | Filtered (≥100 chars) and chunked (≤10k chars) | ~69M rows |
| `balanced.csv` | Balanced 50/50 from combined data | ~100k rows |
| `balanced_processed.csv` | Balanced 50/50 from processed/chunked data | ~100k rows |

### 5.3 Processing Time Estimates

- **Load BrWaC corpus:** ~7-10 minutes (largest dataset, 21 parquet files)
- **Load other sources:** ~2-3 minutes total
- **Combine and save:** ~1 minute
- **Batch filtering and chunking:** ~30-60 minutes (depends on batch size and available memory)
- **Create balanced dataset:** ~5-10 minutes
- **Total pipeline:** ~45-90 minutes

### 5.4 Memory Requirements

**Minimum:**
- RAM: 8 GB recommended (4 GB possible with smaller batch sizes)
- Disk Space: ~50 GB for all data files

**Optimal:**
- RAM: 16 GB or more
- Disk Space: 100 GB (includes intermediate files)

**Memory Management:**
- Use `estimate_memory_usage()` function before processing
- Adjust `batch_size` parameter based on available RAM
- Monitor intermediate directory size during processing

---

## 6. Quality Assurance

### 6.1 Data Validation Checks

The pipeline includes several validation steps:

1. **Length validation:**
   - Minimum length enforced (100-200 chars)
   - Maximum length enforced via chunking (10,000 chars)

2. **Label validation:**
   - Only two valid labels: `human` and `llm`
   - Label counts printed and verified at each stage

3. **Text quality:**
   - Empty strings removed via `.strip()`
   - Null values handled during filtering

4. **Chunking integrity:**
   - Original length preserved in metadata
   - Chunk IDs unique and traceable
   - Chunk count verified against expected values

### 6.2 Analysis Functions

The notebook provides analysis functions to verify processing results:

```python
# Analyze chunking results
def analyze_chunking_results(df):
    """Provide detailed analysis of the chunking results."""
    # Reports: total rows, chunking ratio, length distribution, breakdown by label

# Analyze processed file without loading all into memory
def analyze_processed_results(processed_file):
    """Analyze results of batch processing in chunks."""
    # Reports: total rows, label counts, chunk counts, length statistics
```

**Code Reference:** Cells `3cdc8230` and `b83e7e3a`

---

## 7. Reproducibility

### 7.1 Random Seeds

All random operations use `random_state=42` for reproducibility:
- Balanced dataset sampling
- Dataset shuffling
- Train/test splits (in downstream analysis)

### 7.2 Dependencies

**Required Python packages:**
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `pyarrow` - Efficient parquet file reading
- `tqdm` - Progress bars for batch processing

**Installation:**
```bash
pip install pandas numpy pyarrow tqdm
```

### 7.3 File Path Assumptions

All data files expected in `data/` subdirectory:
```
data/
├── sharegpt-portuguese.json
├── imdb-reviews-pt-br.csv
├── boolq.csv
├── validation_bool.csv
├── brwac/
│   ├── file_001.parquet
│   ├── file_002.parquet
│   └── ... (21 files total)
└── canarim/
    ├── file_001.parquet
    └── ... (multiple files)
```

---

## 8. Known Limitations and Considerations

### 8.1 Data Source Assumptions

1. **Machine-translated text as LLM:**
   - IMDB reviews are machine-translated, classified as `llm`
   - Assumption: Machine translation exhibits similar stylometric patterns to LLM generation
   - May introduce confounding factors

2. **Web-scraped text as Human:**
   - BrWaC corpus may contain some LLM-generated text if scraped from recent websites
   - Assumption: Pre-2020 web text is predominantly human-authored

3. **Chunking creates dependencies:**
   - Multiple chunks from same document are not independent samples
   - May inflate apparent sample size
   - Consider aggregating or deduplicating in downstream analysis

### 8.2 Preprocessing Choices

1. **Minimum length threshold:**
   - Set to 100 chars initially, recommended 200 chars
   - Too low: noisy features from short texts
   - Too high: lose valid samples

2. **Maximum length and chunking:**
   - 10,000 char threshold is arbitrary
   - No overlap between chunks (could lose cross-chunk patterns)
   - Chunk boundary detection imperfect (may split mid-sentence)

3. **Balancing approach:**
   - 50/50 balance may not reflect real-world distribution
   - Upsampling with replacement introduces duplicate samples
   - Alternative: keep imbalanced data, use weighted metrics

### 8.3 Computational Constraints

1. **Memory limitations:**
   - Batch processing required for large datasets
   - Intermediate saves add I/O overhead
   - May not be reproducible across different batch sizes

2. **Processing time:**
   - BrWaC processing is bottleneck (~10 minutes for 21 files)
   - Consider caching or pre-processing

---

## 9. References and Citations

**Data Sources:**
- BrWaC: Brazilian Web as Corpus
- ShareGPT: Community-contributed LLM conversations
- IMDB Reviews: Movie review dataset
- BoolQ: Boolean question-answering dataset
- Canarim: LLM-generated text dataset

**Methods:**
- Stratified sampling: Ensures representative class distribution
- Batch processing: Memory-efficient handling of large datasets
- Text chunking: Standard technique for handling variable-length documents

---

## 10. Contact and Maintenance

**Notebook Location:** `/home/vlofgren/Projects/mestrado/prob_est/0. process_data.ipynb`

**Last Updated:** December 2025

**For questions or issues:** Refer to main project documentation or contact repository maintainer.

---

## Summary Statistics Table

| Metric | Value |
|--------|-------|
| **Data Sources** | 5 (ShareGPT, IMDB, BoolQ, BrWaC, Canarim) |
| **Original Combined Rows** | 2,331,317 |
| **Rows Removed (too short)** | 171,510 (7.4%) |
| **Rows Requiring Chunking** | 166,812 (7.7%) |
| **Final Processed Rows** | ~69 million (with chunks) |
| **Balanced Dataset Rows** | ~100,000 |
| **Minimum Text Length** | 100-200 characters |
| **Maximum Text Length** | 10,000 characters |
| **Chunk Overlap** | 0 characters |
| **Human/LLM Balance** | 50% / 50% |
| **Random Seed** | 42 |
| **Batch Size** | 25,000-50,000 rows |
| **Processing Time** | 45-90 minutes |

---

**End of Data Pipeline Documentation**
