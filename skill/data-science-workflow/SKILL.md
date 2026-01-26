---
name: data-science-workflow
description: The primary workflow for all data science projects. Use this skill whenever a user asks to train a model, build a model, perform analysis, or do analytics. It autonomously orchestrates the full pipeline - data inspection, cleaning, feature engineering, and AutoML training to deliver the best possible results.
---

# LLM Usage Guide: Production Data Science Workflow (Consultative)

This guide outlines how to handle user requests for model training, especially when instructions are vague (e.g., "Train a model on data.csv").

---

## Universal Workflow Principles

### 1. Documentation First
Always check for and read documentation files before inspecting data. Column descriptions prevent incorrect assumptions.

### 2. Scale Awareness
Check data size before processing or training. Large datasets require sampling for efficient iteration.

### 3. Transparency
Communicate your understanding, assumptions, and plan to the user before executing. Allow them to correct course early.

### 4. Iterative Refinement
Use smaller samples for development iterations. Reserve full data for final model training only.

### 5. Preserve Provenance
When creating processed files, use clear naming that indicates source, processing applied, and sample size.
Example: `train_processed_10k_sample.parquet`

### 6. No NaN/Inf in Features
Never create features that produce NaN or Inf values. Common pitfalls and fixes:
- **Division**: Always use `NULLIF()` → `a / NULLIF(b, 0)`
- **Log/Sqrt**: Guard against zero/negative → `LOG(GREATEST(x, 1))`, `SQRT(GREATEST(x, 0))`
- **Missing propagation**: Use `COALESCE()` → `COALESCE(a, 0) + COALESCE(b, 0)`

---

## Phase 0: Initial Triage (The "Vague Request" Handler)
**Trigger**: User provides data but no specific instructions.

1.  **Inspect First**: ALWAYS call `inspect_data(data_path)` immediately to understand the table structure. If there are multiple files, you should inspect all of them to understand the data structure unless you are confident that the file is not important after reading the documentation.
2.  **Identify Target**:
    - *Confident*: If there is an obvious target (e.g., "churn", "target", "price", "species"), **assume it** and state your assumption.
    - *Ambiguous*: If multiple columns could be targets, **ASK the user**. ("I see 'price' and 'quantity'. Which one are we predicting?")
3.  **Determine Goal (consultative)**:
    - *Confident*: If the target imply the goal (e.g., "fraud" -> minimize false positives), suggest the appropriate metric (Precision/Recall).
    - *Ambiguous*: Ask for the business outcome. ("Are we trying to minimize missing fraud cases, or minimize false alarms?")

---

## Phase 0.5: Dataset Discovery
**Trigger**: Dataset is a directory (not a single file), OR any file > 50MB.

### Step 1: Read Documentation First (MANDATORY)
Before ANY data inspection, search the directory for documentation:
- README files: `README`, `README.md`, `README.txt`
- Description files: Any file containing "description", "metadata", "schema", "dictionary"
- Data dictionaries: `.json`, `.yaml`, `.txt` files that aren't data

**Why**: Documentation explains table relationships, column meanings, and intended use cases. Skipping this leads to incorrect assumptions.

### Step 2: Inventory All Data Files
List all data files and check their sizes/row counts:
```sql
SELECT COUNT(*) as rows FROM 'filename.csv'
```

Categorize files into:
- **Primary table**: Contains the target variable (usually `train`, `main`, or similar naming)
- **Auxiliary tables**: Related data that can be aggregated (transactions, history, logs)
- **Test/submission files**: Held-out data for final predictions

### Step 3: Assess Scale & Plan Accordingly
| Data Scale | Definition | Required Action |
|------------|------------|-----------------|
| Small | < 50K rows | Proceed normally |
| Medium | 50K - 200K rows | Recommend sampling for development |
| Large | > 200K rows | **Require** sampling; inform user |

**Sampling Strategy**: Create a stratified sample preserving target distribution:
```sql
SELECT * FROM data 
ORDER BY RANDOM() 
LIMIT [10-20% of original, max 50K rows]
```

### Step 4: Multi-Table Strategy
If multiple related tables exist:
1. **Identify join keys** (from documentation or column inspection)
2. **Plan aggregations**: How to summarize auxiliary tables to join with primary
3. **Communicate plan** to user before executing

### Step 5: Confirm Scope with User
Before proceeding, state:
> "Dataset contains [X files, Y total rows]. I plan to:
> - Use [primary_table] as the main dataset
> - [Sample to N rows / Use full data]
> - [Aggregate features from auxiliary tables / Use primary only]
> 
> Proceed?"

---

## Phase 1: Project Definition
**Goal**: Lock down success criteria and establish a naive baseline before training.

### Check:
- **Problem Type**: Classification vs Regression
- **Primary Metric**: Choose based on business goal:
    - Safety-critical (fraud, medical) → `Recall`
    - Cost-sensitive (marketing, sales) → `Precision`
    - Balanced → `F1`
    - Regression → `R2` or `MAE`

### Establish Naive Baseline
Use `query_data` to calculate a baseline that any useful model must beat:

**For Classification** (majority class baseline):
```sql
-- Class distribution
SELECT target, COUNT(*) as count, 
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct
FROM data GROUP BY target
```
→ Baseline accuracy = largest class percentage (e.g., if 70% are "good", baseline = 70%)

**For Regression** (mean prediction baseline):
```sql
SELECT 
    AVG(target) as mean_baseline,
    AVG(ABS(target - (SELECT AVG(target) FROM data))) as baseline_MAE
FROM data
```
→ A model must have lower MAE than predicting the mean for every sample.

### Document Baseline to User
State clearly:
> "Naive baseline (always predicting majority class 'good'): 70% accuracy. Our model must exceed this to add value."

---

## Phase 2: EDA (Deep Dive)
**Goal**: Inspect data quality to inform training parameters AND feature engineering opportunities.

### Checklist:
1.  **Skewness**: Use `query_data` to check `AVG(col)` vs `MEDIAN(col)`. -> If high skew, set `transformation=True`.
2.  **Ordinality**: Check for inherent order in categories (e.g., "Low/Med/High", "Junior/Senior", rating scales). -> Map to `ordinal_features`.
3.  **Missingness**:
    - *Simple*: If random/small, use `numeric_imputation` (mean/median) or `categorical_imputation` (mode) params.
    - *Complex*: If structural/logic-based, use `process_data`.
4.  **Class Imbalance**: Check target distribution. If moderate (70-30), may not need `fix_imbalance`. If extreme (95-5), likely beneficial.
5.  **Outliers**: Check for extreme values in numeric columns (use `query_data` with MIN/MAX/STDDEV). -> Consider `remove_outliers=True`.
6.  **Feature Relationships**: Look for potential interactions (e.g., credit_amount & duration -> monthly_payment).

---

## Phase 2.5: Domain Research & Feature Engineering 🔬
**Goal**: Leverage domain knowledge to create high-value features.

### When to Apply:
- **Always consider** for any non-trivial dataset
- **Strongly recommended** when baseline model performance is below expectations
- Skip only for pure exploratory analysis or when time is extremely limited

### Step 1: Identify the Problem Domain

Look at column names and target variable to identify the domain:

| Column Indicators | Likely Domain |
|-------------------|---------------|
| amount, duration, payment, credit, loan | Financial/Credit Risk |
| churn, subscription, tenure, contract | Customer Churn |
| price, sales, inventory, demand | Retail/E-commerce |
| diagnosis, symptoms, age, medication | Healthcare/Medical |
| latitude, longitude, distance, location | Geographic/Spatial |
| timestamp, date, hour, day_of_week | Time Series/Temporal |

### Step 2: Research Domain Best Practices

(REQUIRED) Search web for feature engineering patterns for the identified domain:

**Search Queries** (use 2-3 of these):
- `"[domain] machine learning feature engineering best practices"`
- `"[domain] [problem_type] important features"`

**What to Look For**:
- Common ratios/interactions used by practitioners
- Domain-specific KPIs or business metrics
- Regulatory/compliance considerations

### Step 3: Apply Feature Engineering

Based on domain research and data inspection, create features using `process_data`. Common techniques:

1. **Ratios & Intensities**: Divide related numeric features (e.g., total/count, amount/duration)
2. **Binning**: Group continuous variables into meaningful categories
3. **Aggregations**: If multiple rows per entity, create sum/mean/max/min/count
4. **Interactions**: Multiply/combine features that work together
5. **Business Logic Flags**: Create binary indicators based on domain rules

Remember: Always apply safe patterns from Principle #6 (No NaN/Inf) when creating ratio or derived features.

### Step 4: Document Your Reasoning

For each engineered feature, explain to the user:
- **What**: Name and formula
- **Why**: Business rationale
- **Source**: If from research, cite it

---

## Phase 3: Data Processing with Feature Engineering
**Goal**: Create a reliable, enriched dataset (Parquet format).

### Action:
Use `process_data` with a comprehensive SQL query that:
1.  **CAST types explicitly** (all numeric columns to INTEGER/FLOAT)
2.  **Create engineered features** from Phase 2.5 research
3.  **Handle missing values** (if complex logic needed)
4.  **Save as `.parquet`** (strongly recommended over CSV for type preservation)

### Transparency Rule:
You **MUST** show the full SQL query to the user with comments explaining each engineered feature's business rationale.

### Quality Check:
After processing, call `inspect_data` on the new file to verify:
- All types are correct (no accidental strings for numeric columns)
- New features have reasonable value ranges
- No unexpected missing values or infinite values introduced

---

## Phase 4: Model Training
**Goal**: Train an optimized model with all insights from EDA and feature engineering.

### Key Parameters:
- `optimize`: The metric agreed in Phase 1 (Recall/Precision/F1/R2/MAE)
- `ordinal_features`: **Critical** - Map all ordinal categories with proper ordering
- `fold=5`: For faster iteration (use `fold=10` for final validation)
- `session_id=42`: For reproducibility

### Model Selection Parameters:
- `include_models`: Train only specific models (faster, good for baselines). Examples:
  - Quick baseline: `include_models=['dt']` (Decision Tree, ~30 seconds)
  - Fast ensemble: `include_models=['rf', 'lightgbm', 'xgboost']`
  - Linear baseline: `include_models=['lr']`
- `exclude_models`: Exclude slow or problematic models. Examples:
  - Skip GPU-requiring: `exclude_models=['catboost']`
  - Skip slow models: `exclude_models=['catboost', 'xgboost']`

**Common Model IDs**:
| Classification | Regression |
|---------------|------------|
| `lr` (Logistic Regression) | `lr` (Linear Regression) |
| `dt` (Decision Tree) | `dt` (Decision Tree) |
| `rf` (Random Forest) | `rf` (Random Forest) |
| `xgboost`, `lightgbm`, `catboost` | `xgboost`, `lightgbm`, `catboost` |
| `knn`, `nb`, `svm` | `ridge`, `lasso`, `en` |

### Conditional Parameters (based on EDA):
- `transformation=True`: If skewed distributions detected in Phase 2
- `normalize=True`: Recommended for linear/distance-based models (not needed for tree-based)
- `polynomial_features=True`: Generally beneficial, low risk
- `fix_imbalance=True`: Only if extreme imbalance (>80:20) detected in Phase 2. Use with `numeric_imputation` and `categorical_imputation` parameters.
- `remove_outliers=True`: If extreme outliers detected in Phase 2

### Training Output:
The training result includes:
- **`metadata`**: CV metrics for the best model
- **`test_metrics`**: Holdout set performance
- **`feature_importances`**: Dict of `{feature_name: importance}` sorted by importance (descending)
  - Available for tree-based models (RF, XGBoost, LightGBM, etc.) and linear models
  - Use this to understand which features drive predictions

### Speed Tips:
- For quick baseline: `include_models=['dt']` (~30 seconds)
- For fast iteration: `include_models=['rf', 'lightgbm']` (~2 minutes)
- ⏱️ Expect 3-10 min for full model comparison with ~10K rows

### Document:
Report top 3 models with their metrics to user in table format.

---

## Phase 5: Evaluation & Comparison
**Goal**: Contextualize results against the naive baseline and select best model.

### Comparison Table (Include Baseline):
Always show the naive baseline from Phase 1 for context:

| Config | Model | Accuracy | Recall | Precision | F1 | vs Baseline |
|--------|-------|----------|--------|-----------|-----|-------------|
| Naive guess | — | 70.0% | — | — | — | — |
| Trained model | CatBoost | 77.3% | 75.6% | 74.3% | 74.9% | +7.3 pts ✅ |

### Interpret Results:
- Quantify improvement over naive baseline (\"+7.3 percentage points\", \"10.4% relative improvement\")
- Translate to business impact (\"Catches 73 more bad credits out of 1000\")

### Success Check:
Does the best model meet the Phase 1 Success Criteria?
- ✅ If yes: Proceed to Transparency Report, provide model_id and path
- ❌ If no: Proceed to Phase 6 (Iteration)

---

## Phase 6: Iteration (If Needed)
**Goal**: Systematically improve model performance.

If performance is still below target:

### 1. Analyze Feature Importances
Review `feature_importances` from the training result:
- **High importance features**: Focus engineering efforts here (create interactions, better encodings)
- **Low/zero importance features**: Consider removing to reduce noise
- **Missing domain features**: If expected important features aren't showing up, check for data issues

Example iteration workflow:
```
1. Train baseline: include_models=['rf'] → get feature_importances
2. Identify top 5 features, engineer interactions between them
3. Retrain with engineered features
4. Compare improvement
```

### 2. Feature Engineering Improvements
- Create interactions between top important features
- Try different encodings for high-importance categoricals
- Bin numeric features that show non-linear relationships

### 3. Model Selection
- If linear models perform poorly: Focus on tree-based (`include_models=['rf', 'xgboost', 'lightgbm']`)
- If tree models overfit: Try regularized linear models (`include_models=['ridge', 'lasso']`)

### 4. Other Strategies
- **Try Different Encodings**: Test WoE transformation for categorical features
- **Ensemble Methods**: Combine multiple models
- **Collect More Data**: Sometimes this is the only solution
- **Revisit Metric Choice**: Confirm we're optimizing the right business objective

**Communicate Progress**: Keep user informed of iterations and trade-offs.

---

## Phase 7: Transparency Report (MANDATORY)
**Goal**: Provide complete reproducibility documentation.

After training is complete, you **MUST** provide a summary that enables the user to reproduce your work.

### Feature Engineering Decisions:
List each engineered feature with rationale:

| Feature | Source | Rationale |
|---------|--------|-----------|
| monthly_burden | loan_amount / duration | Captures repayment intensity |
| has_prior_default | Aggregated from history table | Strong risk indicator |

### Data Processing Query:
Provide the complete SQL query used in `process_data`:

```sql
-- Full query used to create training dataset
SELECT 
    id,
    target,
    -- Original features (cast to correct types)
    CAST(age AS INTEGER) as age,
    CAST(income AS FLOAT) as income,
    -- Engineered features
    ROUND(CAST(loan_amount AS FLOAT) / NULLIF(CAST(duration AS FLOAT), 0), 2) as monthly_burden,  -- Repayment capacity
    CASE WHEN employment_years < 1 THEN 'new' 
         WHEN employment_years < 5 THEN 'mid' 
         ELSE 'established' END as employment_category  -- Stability indicator
FROM 'source_data.csv'
WHERE target IS NOT NULL
```

### Model Configuration:
Document the final training parameters used:
- Metric optimized
- Ordinal feature mappings
- Special flags enabled (transformation, fix_imbalance, etc.)

This enables the user to:
1. Understand the reasoning behind each decision
2. Modify and re-run the data processing independently
3. Reproduce the exact model training configuration