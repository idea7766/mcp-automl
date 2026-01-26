import numpy as np
import pandas as pd
import uuid
import os
import json
import asyncio
import duckdb
import logging
import argparse
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import PromptMessage, TextContent
from pycaret.classification import setup as setup_clf, compare_models as compare_models_clf, pull as pull_clf, save_model as save_model_clf, load_model as load_model_clf, predict_model as predict_model_clf, get_config as get_config_clf
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg, pull as pull_reg, save_model as save_model_reg, load_model as load_model_reg, predict_model as predict_model_reg, get_config as get_config_reg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Module-level configuration (set via argparse in main())
EXPERIMENT_DIR = "~/.mcp-automl/experiments"
DEFAULT_SESSION_ID = 42
QUERY_RESULT_LIMIT = 100
SUPPORTED_FILE_FORMATS = ('.csv', '.parquet', '.json')

mcp = FastMCP("mcp-automl")

class PandasJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles pandas NA types and numpy types."""
    
    def default(self, obj):
        # Handle pandas NA types (pd.NA, pd.NaT)
        if pd.isna(obj):
            return None
        # Handle numpy integer types
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        # Handle numpy floating types
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj):
                return None
            return float(obj)
        # Handle numpy boolean
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Let the base class raise the TypeError
        return super().default(obj)

def _load_dataframe_fast(data_path: str, sample_size: int = None, 
                         sample_method: str = 'reservoir') -> pd.DataFrame:
    """
    Unified data loader using DuckDB with optional sampling.
    
    This function provides:
    - Fast I/O using DuckDB for all file formats (CSV, Parquet, JSON)
    - Smart sampling for large files using reservoir sampling
    - Consistent dtype inference by always returning pandas DataFrame
    - Flexible loading: full data for training, sampled data for inspection
    
    Args:
        data_path: Absolute path to the data file (CSV, Parquet, or JSON).
        sample_size: If provided, returns a random sample of this size.
                     If None, loads the entire dataset.
        sample_method: Sampling method to use (default: 'reservoir').
                       Currently only 'reservoir' is supported.
        
    Returns:
        pandas DataFrame with data loaded from file.
        
    Raises:
        ValueError: If file format is not supported or sample_method is invalid.
    """
    # Validate file format
    if not any(data_path.endswith(ext) for ext in SUPPORTED_FILE_FORMATS):
        supported = ', '.join(SUPPORTED_FILE_FORMATS)
        raise ValueError(f"Unsupported file format: {data_path}. Supported formats: {supported}")
    
    # Connect to DuckDB (in-memory)
    con = duckdb.connect(database=':memory:')
    
    # Full load for training (no sampling)
    if sample_size is None:
        logger.debug(f"Loading full dataset from {data_path}")
        return con.execute(f"SELECT * FROM '{data_path}'").df()
    
    # Check total row count to determine if sampling is needed
    total_rows = con.execute(f"SELECT COUNT(*) FROM '{data_path}'").fetchone()[0]
    logger.debug(f"File has {total_rows} total rows, sample_size={sample_size}")
    
    if total_rows <= sample_size:
        # File is small enough, just load everything
        logger.debug(f"File is small ({total_rows} <= {sample_size}), loading all rows")
        return con.execute(f"SELECT * FROM '{data_path}'").df()
    
    # Apply sampling for large files
    if sample_method == 'reservoir':
        logger.info(f"Applying reservoir sampling: {sample_size} rows from {total_rows} total")
        # Reservoir sampling gives truly random sample
        return con.execute(f"""
            SELECT * FROM '{data_path}' 
            USING SAMPLE reservoir({sample_size} ROWS)
        """).df()
    else:
        raise ValueError(f"Unknown sample_method: {sample_method}. Only 'reservoir' is supported.")

def _get_feature_info(get_config_func, target_column: str) -> dict:
    """Extracts feature information from PyCaret config."""
    try:
        X_train = get_config_func('X_train')
        dataset = get_config_func('dataset')
        
        used_features = list(X_train.columns)
        all_cols = list(dataset.columns)
        
        # Deduce ignored features: in dataset but not in X_train and not target
        ignored_features = [c for c in all_cols if c != target_column and c not in used_features]
        
        numeric_features = list(X_train.select_dtypes(include=np.number).columns)
        # Objects and categories are categorical
        categorical_features = list(X_train.select_dtypes(include=['object', 'category']).columns)
        
        return {
            "used_features": used_features,
            "ignored_features": ignored_features,
            "actual_numeric_features": numeric_features,
            "actual_categorical_features": categorical_features
        }
    except Exception as e:
        logger.error(f"Error extracting feature info: {e}", exc_info=True)
        return {}

def _get_feature_importances(model, get_config_func) -> dict:
    """Extracts feature importances from the model if available.
    
    Supports tree-based models (feature_importances_) and linear models (coef_).
    Returns a dict of {feature_name: importance} sorted by importance descending.
    """
    try:
        X_train = get_config_func('X_train')
        feature_names = list(X_train.columns)
        
        # Try tree-based models first (RF, XGBoost, LightGBM, etc.)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_dict = dict(zip(feature_names, [float(x) for x in importances]))
            # Sort by importance descending
            return dict(sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True))
        
        # Try linear models (LogisticRegression, Ridge, Lasso, etc.)
        if hasattr(model, 'coef_'):
            coef = model.coef_
            # For multi-class, coef_ has shape (n_classes, n_features), take mean absolute
            if len(coef.shape) > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)
            importance_dict = dict(zip(feature_names, [float(x) for x in importances]))
            return dict(sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return {}
    except Exception as e:
        logger.warning(f"Could not extract feature importances: {e}")
        return {}

def _save_results(run_id: str, model, results: pd.DataFrame, save_model_func, metadata: dict, test_results: pd.DataFrame = None, feature_importances: dict = None) -> str:
    """Helper to save model and metrics.
    
    Args:
        run_id: Unique run identifier.
        model: The trained model object.
        results: DataFrame containing CV results (from pull()).
        save_model_func: Function to save the model.
        metadata: Dictionary of run configuration.
        test_results: DataFrame containing test/holdout results (from predict_model()).
    """
    # Create directory
    run_dir = os.path.join(EXPERIMENT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(run_dir, "model")
    save_model_func(model, model_path)
    
    # Save CV metrics (best model)
    if not results.empty:
        metrics = results.iloc[0].to_dict()
    else:
        metrics = {}
    
    # Save Test/Holdout metrics
    test_metrics = {}
    if test_results is not None and not test_results.empty:
        # predict_model returns a dataframe where metrics are often in the first row or summary
        # For PyCaret predict_model(), it returns the metrics if data is passed with labels.
        # However, pull() after predict_model() usually contains the metrics.
        # Let's assume test_results is result of pull() after predict_model.
        try:
             test_metrics = test_results.to_dict(orient='records')[0]
        except (KeyError, IndexError) as e:
             logger.warning(f"Could not extract test metrics: {e}")
             test_metrics = {}

    # Merge metadata into metrics for saving
    full_metadata = {**metadata, "cv_metrics": metrics, "test_metrics": test_metrics}
    
    metadata_path = os.path.join(run_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(full_metadata, f, indent=2)

    # Generate HTML Report
    
    # Format lists for HTML
    def fmt_list(l):
        return ", ".join(l) if l else "None"

    # Separate metrics from configuration
    config_metadata = {k: v for k, v in metadata.items() if k not in metrics}
    
    # Generate config rows
    config_rows = ""
    for k, v in config_metadata.items():
        # Clean up keys for display
        display_key = k.replace("_", " ").title()
        
        # Format values
        if isinstance(v, list):
            display_val = fmt_list(v)
        else:
            display_val = str(v)
            
        config_rows += f'<div class="metadata-item"><strong>{display_key}:</strong> {display_val}</div>\n'

    html_content = f"""
    <html>
    <head>
        <title>Training Result - {run_id}</title>
        <style>
            body {{ font-family: monospace, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metadata-item {{ margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        <h1>Training Run: {run_id}</h1>
        
        <h2>Configuration</h2>
        <div class="metadata">
            {config_rows}
        </div>

        <h2>Model Metrics (CV)</h2>
        {results.to_html(classes='table', index=False) if not results.empty else "<p>No results available</p>"}
        
        <h2>Test/Holdout Metrics</h2>
        {test_results.to_html(classes='table', index=False) if test_results is not None and not test_results.empty else "<p>No test results available</p>"}
        
    </body>
    </html>
    """
    
    html_path = os.path.join(run_dir, "result.html")
    with open(html_path, "w") as f:
        f.write(html_content)
        
    return json.dumps({
        "run_id": run_id,
        "model_path": model_path + ".pkl", # pycaret adds .pkl
        "data_path": metadata.get("data_path"),
        "test_data_path": metadata.get("test_data_path"),
        "metadata": metrics,
        "test_metrics": test_metrics,
        "feature_importances": feature_importances if feature_importances else {},
        "report_path": html_path
    }, indent=2)

def _train_classifier_sync(run_id: str, data_path: str, target_column: str, ignore_features: list[str], numeric_features: list[str], categorical_features: list[str],
                           ordinal_features: dict[str, list[str]], date_features: list[str], text_features: list[str], keep_features: list[str],
                           imputation_type: str, numeric_imputation: str, categorical_imputation: str,
                           fix_imbalance: bool, remove_outliers: bool, normalize: bool, normalize_method: str,
                           transformation: bool, transformation_method: str,
                           polynomial_features: bool, interaction_features: list[str], bin_numeric_features: list[str],
                           feature_selection: bool, feature_selection_method: str, n_features_to_select: float,
                           fold_strategy: str, fold: int, n_jobs: int, test_data_path: str = None, optimize: str = None,
                           include_models: list[str] = None, exclude_models: list[str] = None) -> str:
    """Synchronous helper for classifier training."""
    # Use unified loader for consistent dtypes
    data = _load_dataframe_fast(data_path)
    
    session_id = DEFAULT_SESSION_ID

    # Handle Test Data
    test_data = None
    if test_data_path:
        test_data = _load_dataframe_fast(test_data_path)
            
        # Ensure unique indices across train and test
        data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)
        test_data.index = test_data.index + len(data)
    # Filter out None values to let PyCaret defaults take over where appropriate
    setup_params = {
        "data": data,
        "test_data": test_data,
        "target": target_column,
        "session_id": session_id,
        "verbose": False,
        "html": False,
        "ignore_features": ignore_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "ordinal_features": ordinal_features,
        "date_features": date_features,
        "text_features": text_features,
        "keep_features": keep_features,
        "imputation_type": imputation_type,
        "numeric_imputation": numeric_imputation,
        "categorical_imputation": categorical_imputation,
        "fix_imbalance": fix_imbalance,
        "remove_outliers": remove_outliers,
        "normalize": normalize,
        "normalize_method": normalize_method,
        "transformation": transformation,
        "transformation_method": transformation_method,
        "polynomial_features": polynomial_features,
        "interaction_features": interaction_features,
        "bin_numeric_features": bin_numeric_features,
        "feature_selection": feature_selection,
        "feature_selection_method": feature_selection_method,
        "n_features_to_select": n_features_to_select,
        "fold_strategy": fold_strategy,
        "fold": fold,
        "n_jobs": n_jobs
    }
    # Remove None values
    setup_params = {k: v for k, v in setup_params.items() if v is not None}
    
    s = setup_clf(**setup_params)
    
    feature_info = _get_feature_info(get_config_clf, target_column)

    # Only pass sort if optimize is specified
    compare_kwargs = {"n_select": 1, "verbose": False}
    if optimize is not None:
        compare_kwargs["sort"] = optimize
    if include_models is not None:
        compare_kwargs["include"] = include_models
    if exclude_models is not None:
        compare_kwargs["exclude"] = exclude_models
    
    best_model = compare_models_clf(**compare_kwargs)
    if isinstance(best_model, list):
        if not best_model:
            raise ValueError("compare_models returned an empty list. Try relaxing constraints or collecting more data.")
        best_model = best_model[0]
    results = pull_clf()
    
    # Extract feature importances
    feature_importances = _get_feature_importances(best_model, get_config_clf)
    
    # Evaluate on holdout (test_data or split)
    predict_model_clf(best_model)
    test_results = pull_clf()
    
    metadata = {
        "data_path": data_path,
        "test_data_path": test_data_path,
        "target_column": target_column,
        "session_id": session_id,
        "task": "classification",
        "include_models": include_models,
        "exclude_models": exclude_models,
        **setup_params, # Include all setup params in metadata
        **feature_info
    }
    # Remove dataframes/series from metadata if they slipped in (data is in formatting)
    if "data" in metadata: del metadata["data"]
    if "test_data" in metadata: del metadata["test_data"]
    
    return _save_results(run_id, best_model, results, save_model_clf, metadata, test_results, feature_importances)

@mcp.tool()
async def train_classifier(data_path: str, target_column: str, ctx: Context, 
                           ignore_features: list[str] = None, numeric_features: list[str] = None, categorical_features: list[str] = None,
                           ordinal_features: dict[str, list[str]] = None, date_features: list[str] = None, text_features: list[str] = None, keep_features: list[str] = None,
                           imputation_type: str = "simple", numeric_imputation: str = "mean", categorical_imputation: str = "mode",
                           fix_imbalance: bool = False, remove_outliers: bool = False, normalize: bool = False, normalize_method: str = "zscore",
                           transformation: bool = False, transformation_method: str = "yeo-johnson",
                           polynomial_features: bool = False, interaction_features: list[str] = None, bin_numeric_features: list[str] = None,
                           feature_selection: bool = False, feature_selection_method: str = "classic", n_features_to_select: float = 0.2,
                           fold_strategy: str = "kfold", fold: int = 10, n_jobs: int = -1, test_data_path: str = None, optimize: str = None,
                           include_models: list[str] = None, exclude_models: list[str] = None) -> str:
    """
    Train a classification model using PyCaret with advanced configuration.

    - NOTE: Please use absolute paths for data_path and test_data_path to avoid path resolution errors.
    
    Args:
        data_path: Path to dataset (csv/parquet/json).
        target_column: Name of target column.
        test_data_path: Optional path to specific test dataset. If provided, used for evaluation/holdout.
        optimize: Metric to optimize for (e.g., 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC'). Default is 'Accuracy'.
        include_models: List of model IDs to include in comparison (e.g., ['lr', 'dt', 'rf']). If None, all models are compared.
        exclude_models: List of model IDs to exclude from comparison (e.g., ['catboost']). If None, no models are excluded.
        ignore_features: Features to ignore.
        numeric_features: Features to treat as numeric.
        categorical_features: Features to treat as categorical.
        ordinal_features: Dictionary of ordinal features and their order (e.g., {'grade': ['low', 'medium', 'high']}).
        date_features: Features to treat as dates.
        text_features: Features to treat as text (for TF-IDF etc).
        keep_features: Features to ensure are kept.
        imputation_type: 'simple' or 'iterative' (default: 'simple').
        numeric_imputation: 'mean', 'median', 'mode' or int/float (default: 'mean').
        categorical_imputation: 'mode' or str (default: 'mode').
        fix_imbalance: If True, fix imbalance in training data (default: False).
        remove_outliers: If True, remove outliers from training data (default: False).
        normalize: If True, scale features (default: False). Recommended for linear models.
        normalize_method: 'zscore', 'minmax', 'maxabs', 'robust' (default: 'zscore').
        transformation: If True, apply gaussian transformation to make data more normal (default: False).
        transformation_method: 'yeo-johnson' or 'quantile' (default: 'yeo-johnson').
        polynomial_features: If True, create polynomial features (default: False).
        interaction_features: List of features to create interactions for.
        bin_numeric_features: List of numeric features to bin into categories.
        feature_selection: If True, select best features (default: False).
        feature_selection_method: 'classic', 'univariate', 'sequential' (default: 'classic').
        n_features_to_select: Fraction (0.0-1.0) or number of features to select (default: 0.2).
        fold_strategy: 'kfold', 'stratifiedkfold', 'groupkfold', 'timeseries' (default: 'kfold').
        fold: Number of folds (default: 10).
        n_jobs: Number of jobs to run in parallel (-1 for all cores).
    
    Returns:
        JSON string with run_id, model_path, metrics, feature_importances, and report_path.
    """
    try:
        run_id = str(uuid.uuid4())
        await ctx.report_progress(0, 100)
        await ctx.info(f"Starting advanced classification training run {run_id}")
        
        result = await asyncio.to_thread(
            _train_classifier_sync, 
            run_id, data_path, target_column, ignore_features, numeric_features, categorical_features,
            ordinal_features, date_features, text_features, keep_features,
            imputation_type, numeric_imputation, categorical_imputation,
            fix_imbalance, remove_outliers, normalize, normalize_method,
            transformation, transformation_method,
            polynomial_features, interaction_features, bin_numeric_features,
            feature_selection, feature_selection_method, n_features_to_select,
            fold_strategy, fold, n_jobs, test_data_path, optimize,
            include_models, exclude_models
        )
        
        await ctx.report_progress(100, 100)
        await ctx.info(f"Finished classification training run {run_id}")
        return result
    except Exception as e:
        return f"Error training classifier: {str(e)}"

def _train_regressor_sync(run_id: str, data_path: str, target_column: str, ignore_features: list[str], numeric_features: list[str], categorical_features: list[str],
                          ordinal_features: dict[str, list[str]], date_features: list[str], text_features: list[str], keep_features: list[str],
                          imputation_type: str, numeric_imputation: str, categorical_imputation: str,
                          remove_outliers: bool, normalize: bool, normalize_method: str,
                          transformation: bool, transformation_method: str,
                          polynomial_features: bool, interaction_features: list[str], bin_numeric_features: list[str],
                          feature_selection: bool, feature_selection_method: str, n_features_to_select: float,
                          fold_strategy: str, fold: int, n_jobs: int, test_data_path: str = None, optimize: str = "R2",
                          include_models: list[str] = None, exclude_models: list[str] = None) -> str:
    """Synchronous helper for regressor training."""
    # Use unified loader for consistent dtypes
    data = _load_dataframe_fast(data_path)
    
    session_id = DEFAULT_SESSION_ID

    # Handle Test Data
    test_data = None
    if test_data_path:
        test_data = _load_dataframe_fast(test_data_path)
            
        # Ensure unique indices across train and test
        data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)
        test_data.index = test_data.index + len(data)
    # Filter out None values to let PyCaret defaults take over where appropriate
    setup_params = {
        "data": data,
        "test_data": test_data,
        "target": target_column,
        "session_id": session_id,
        "verbose": False,
        "html": False,
        "ignore_features": ignore_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "ordinal_features": ordinal_features,
        "date_features": date_features,
        "text_features": text_features,
        "keep_features": keep_features,
        "imputation_type": imputation_type,
        "numeric_imputation": numeric_imputation,
        "categorical_imputation": categorical_imputation,
        "remove_outliers": remove_outliers,
        "normalize": normalize,
        "normalize_method": normalize_method,
        "transformation": transformation,
        "transformation_method": transformation_method,
        "polynomial_features": polynomial_features,
        "interaction_features": interaction_features,
        "bin_numeric_features": bin_numeric_features,
        "feature_selection": feature_selection,
        "feature_selection_method": feature_selection_method,
        "n_features_to_select": n_features_to_select,
        "fold_strategy": fold_strategy,
        "fold": fold,
        "n_jobs": n_jobs
    }
    # Remove None values
    setup_params = {k: v for k, v in setup_params.items() if v is not None}
    
    s = setup_reg(**setup_params)
    
    feature_info = _get_feature_info(get_config_reg, target_column)
    
    # Only pass sort if optimize is specified
    compare_kwargs = {"n_select": 1, "verbose": False}
    if optimize is not None:
        compare_kwargs["sort"] = optimize
    if include_models is not None:
        compare_kwargs["include"] = include_models
    if exclude_models is not None:
        compare_kwargs["exclude"] = exclude_models
    
    best_model = compare_models_reg(**compare_kwargs)
    if isinstance(best_model, list):
        if not best_model:
            raise ValueError("compare_models returned an empty list. Try relaxing constraints or collecting more data.")
        best_model = best_model[0]
    results = pull_reg()
    
    # Extract feature importances
    feature_importances = _get_feature_importances(best_model, get_config_reg)
    
    # Evaluate on holdout
    predict_model_reg(best_model)
    test_results = pull_reg()
    
    metadata = {
        "data_path": data_path,
        "test_data_path": test_data_path,
        "target_column": target_column,
        "session_id": session_id,
        "task": "regression",
        "include_models": include_models,
        "exclude_models": exclude_models,
        **setup_params, # Include all setup params in metadata
        **feature_info
    }
    # Remove dataframes/series from metadata if they slipped in
    if "data" in metadata: del metadata["data"]
    if "test_data" in metadata: del metadata["test_data"]
    
    return _save_results(run_id, best_model, results, save_model_reg, metadata, test_results, feature_importances)

@mcp.tool()
async def train_regressor(data_path: str, target_column: str, ctx: Context, 
                          ignore_features: list[str] = None, numeric_features: list[str] = None, categorical_features: list[str] = None,
                          ordinal_features: dict[str, list[str]] = None, date_features: list[str] = None, text_features: list[str] = None, keep_features: list[str] = None,
                          imputation_type: str = "simple", numeric_imputation: str = "mean", categorical_imputation: str = "mode",
                          remove_outliers: bool = False, normalize: bool = False, normalize_method: str = "zscore",
                          transformation: bool = False, transformation_method: str = "yeo-johnson",
                          polynomial_features: bool = False, interaction_features: list[str] = None, bin_numeric_features: list[str] = None,
                          feature_selection: bool = False, feature_selection_method: str = "classic", n_features_to_select: float = 0.2,
                          fold_strategy: str = "kfold", fold: int = 10, n_jobs: int = -1, test_data_path: str = None, optimize: str = "R2",
                          include_models: list[str] = None, exclude_models: list[str] = None) -> str:
    """
    Train a regression model using PyCaret with advanced configuration.

    - NOTE: Please use absolute paths for data_path and test_data_path to avoid path resolution errors.
    
    Args:
        data_path: Path to dataset (csv/parquet/json).
        target_column: Name of target column.
        test_data_path: Optional path to specific test dataset. If provided, used for evaluation/holdout.
        optimize: Metric to optimize for (e.g., 'R2', 'RMSE', 'MAE', 'MSE', 'RMSLE', 'MAPE'). Default is 'R2'.
        include_models: List of model IDs to include in comparison (e.g., ['lr', 'dt', 'rf']). If None, all models are compared.
        exclude_models: List of model IDs to exclude from comparison (e.g., ['catboost']). If None, no models are excluded.
        ignore_features: Features to ignore.
        numeric_features: Features to treat as numeric.
        categorical_features: Features to treat as categorical.
        ordinal_features: Dictionary of ordinal features and their order.
        date_features: Features to treat as dates.
        text_features: Features to treat as text (for TF-IDF etc).
        keep_features: Features to ensure are kept.
        imputation_type: 'simple' or 'iterative' (default: 'simple').
        numeric_imputation: 'mean', 'median', 'mode' or int/float (default: 'mean').
        categorical_imputation: 'mode' or str (default: 'mode').
        remove_outliers: If True, remove outliers from training data (default: False).
        normalize: If True, scale features (default: False). Recommended for linear models.
        normalize_method: 'zscore', 'minmax', 'maxabs', 'robust' (default: 'zscore').
        transformation: If True, apply gaussian transformation to make data more normal (default: False).
        transformation_method: 'yeo-johnson' or 'quantile' (default: 'yeo-johnson').
        polynomial_features: If True, create polynomial features (default: False).
        interaction_features: List of features to create interactions for.
        bin_numeric_features: List of numeric features to bin into categories.
        feature_selection: If True, select best features (default: False).
        feature_selection_method: 'classic', 'univariate', 'sequential' (default: 'classic').
        n_features_to_select: Fraction (0.0-1.0) or number of features to select (default: 0.2).
        fold_strategy: 'kfold', 'stratifiedkfold', 'groupkfold', 'timeseries' (default: 'kfold').
        fold: Number of folds (default: 10).
        n_jobs: Number of jobs to run in parallel (-1 for all cores).
    
    Returns:
        JSON string with run_id, model_path, metrics, feature_importances, and report_path.
    """
    try:
        run_id = str(uuid.uuid4())
        await ctx.report_progress(0, 100)
        await ctx.info(f"Starting advanced regression training run {run_id}")
        
        result = await asyncio.to_thread(
            _train_regressor_sync, 
            run_id, data_path, target_column, ignore_features, numeric_features, categorical_features,
            ordinal_features, date_features, text_features, keep_features,
            imputation_type, numeric_imputation, categorical_imputation,
            remove_outliers, normalize, normalize_method,
            transformation, transformation_method,
            polynomial_features, interaction_features, bin_numeric_features,
            feature_selection, feature_selection_method, n_features_to_select,
            fold_strategy, fold, n_jobs, test_data_path, optimize,
            include_models, exclude_models
        )
        
        await ctx.report_progress(100, 100)
        await ctx.info(f"Finished regression training run {run_id}")
        return result
    except Exception as e:
        return f"Error training regressor: {str(e)}"



def _predict_sync(run_id: str, data_path: str) -> str:
    """Synchronous helper for predictions."""
    run_dir = os.path.join(EXPERIMENT_DIR, run_id)
    metadata_path = os.path.join(run_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        return f"Error: Run ID {run_id} not found."

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    task = metadata.get("task")
    model_path = os.path.join(run_dir, "model")
    
    # Load data using unified loader
    if not os.path.exists(data_path):
        return f"Error: Data file not found at {data_path}"
        
    try:
        input_data = _load_dataframe_fast(data_path)
    except Exception as e:
        return f"Error loading data file: {str(e)}"

    if task == "classification":
        model = load_model_clf(model_path)
        predictions = predict_model_clf(model, data=input_data)
    elif task == "regression":
        model = load_model_reg(model_path)
        predictions = predict_model_reg(model, data=input_data)
    else:
        return f"Error: Unknown task type '{task}' in metadata."
        
    # Save predictions
    predictions_dir = os.path.join(run_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    prediction_id = str(uuid.uuid4())
    prediction_file = f"prediction_{prediction_id}.json"
    prediction_path = os.path.join(predictions_dir, prediction_file)
    
    predictions.to_json(prediction_path, orient="records", indent=2)
    
    return prediction_path

@mcp.tool()
async def predict(run_id: str, data_path: str, ctx: Context = None) -> str:
    """
    Make predictions using a trained model.

    - NOTE: Please use absolute paths for data_path to avoid path resolution errors.

    Args:
        run_id: The ID of the training run (returned by train_classifier or train_regressor).
        data_path: The path to the CSV or JSON file containing the input data.

    Returns:
        The absolute path to the JSON file containing the predictions.
    """
    try:
        if ctx:
            await ctx.report_progress(0, 100)
            await ctx.info(f"Loading model and making predictions...")
        
        result = await asyncio.to_thread(_predict_sync, run_id, data_path)
        
        if ctx:
            await ctx.report_progress(100, 100)
            await ctx.info("Prediction complete")
        
        return result
    except Exception as e:
        logger.error(f"Error making predictions: {e}", exc_info=True)
        return f"Error making predictions: {str(e)}"

def _inspect_data_sync(data_path: str, n_rows: int = 5) -> str:
    """Synchronous helper for data inspection using unified loader."""
    # Use unified loader with sampling for large files
    # Sample size of 10,000 rows for statistics computation
    SAMPLE_SIZE = 10000
    
    # Get total row count first
    con = duckdb.connect(database=':memory:')
    total_rows = con.execute(f"SELECT COUNT(*) FROM '{data_path}'").fetchone()[0]
    
    # Load data using unified loader (with sampling if needed)
    data = _load_dataframe_fast(data_path, sample_size=SAMPLE_SIZE)
    
    # Structure
    structure = {
        "rows": total_rows,  # Report actual total
        "columns": len(data.columns),
        "column_names": list(data.columns),
        "dtypes": data.dtypes.astype(str).to_dict()
    }
    
    # Statistics (computed on sample if file is large)
    stats = {
        "missing_values": data.isnull().sum().to_dict(),
        "missing_ratio": (data.isnull().sum() / len(data)).to_dict(),
        "unique_values": data.nunique().to_dict()
    }
    
    # Add note if sampling was used
    if total_rows > SAMPLE_SIZE:
        stats["⚠️ note"] = f"Statistics computed on {SAMPLE_SIZE} row sample from {total_rows} total rows"
    
    # Previews (Column-oriented)
    # For preview, always get first/last rows from original file
    preview_df = con.execute(f"SELECT * FROM '{data_path}' LIMIT {n_rows}").df()
    tail_df = con.execute(f"""
        SELECT * FROM '{data_path}' 
        OFFSET {max(0, total_rows - n_rows)}
    """).df()
    
    seed = 42
    previews = {
        "head": preview_df.to_dict(orient="list"),
        "tail": tail_df.to_dict(orient="list"),
        "sample": data.sample(min(n_rows, len(data)), random_state=seed).to_dict(orient="list")
    }
    
    return json.dumps({
        "structure": structure,
        "statistics": stats,
        "previews": previews
    }, indent=2, cls=PandasJSONEncoder)

@mcp.tool()
async def inspect_data(data_path: str, n_rows: int = 5, ctx: Context = None) -> str:
    """
    Get comprehensive statistics and a preview of the dataset to understand its quality and structure.
    Use this to check for missing values, unique counts, and basic data types.

    - NOTE: Please use absolute paths for data_path to avoid path resolution errors.
    - NOTE: For files larger than 10,000 rows, statistics are computed on a random sample for performance.
    
    Args:
        data_path: Path to the CSV, Parquet, or JSON file.
        n_rows: Number of rows to show in head/tail/sample previews (default: 5).
        
    Returns:
        JSON string containing structure, stats, and previews.
    """
    try:
        if ctx:
            await ctx.report_progress(0, 100)
            await ctx.info(f"Inspecting data from {data_path}")
        
        result = await asyncio.to_thread(_inspect_data_sync, data_path, n_rows)
        
        if ctx:
            await ctx.report_progress(100, 100)
            await ctx.info("Data inspection complete")
        
        return result
    except Exception as e:
        logger.error(f"Error inspecting data: {e}", exc_info=True)
        return f"Error inspecting data: {str(e)}"

def _query_data_sync(query: str) -> str:
    """Synchronous helper for DuckDB queries."""
    con = duckdb.connect(database=':memory:')
    
    # Security/Limit check: enforce LIMIT if not present?
    # For now, just truncating the result df is safer and easier than parsing SQL.
    
    df = con.execute(query).df()
    
    if len(df) > QUERY_RESULT_LIMIT:
         df = df.head(QUERY_RESULT_LIMIT)
         
    return df.to_json(orient="records", date_format="iso")

@mcp.tool()
async def query_data(query: str, ctx: Context = None) -> str:
    """
    Execute a DuckDB SQL query on data files (CSV, Parquet, JSON) to gain deeper insights.
    
    CRITICAL: This is your PRIMARY tool for advanced data exploration.
    - Use this to aggregate data (GROUP BY), join multiple files, calculate derived metrics, or inspect specific subsets.
    - Prefer this over 'inspect_data' when you need to answer specific questions about the data distribution or relationships.
    - You can query files directly in the FROM clause, e.g., "SELECT category, AVG(price) FROM 'data.csv' GROUP BY category".

    - NOTE: Please use absolute paths for files in your FROM clause to avoid path resolution errors.
    
    Args:
        query: Standard DuckDB SQL query.
        
    Returns:
        JSON string containing the query results (limit 100 rows).
    """
    try:
        if ctx:
            await ctx.report_progress(0, 100)
            await ctx.info("Executing query...")
        
        result = await asyncio.to_thread(_query_data_sync, query)
        
        if ctx:
            await ctx.report_progress(100, 100)
            await ctx.info("Query complete")
        
        return result
    except Exception as e:
        logger.error(f"Error executing query: {e}", exc_info=True)
        return f"Error executing query: {str(e)}"

def _process_data_sync(query: str, output_path: str) -> str:
    """Synchronous helper for process_data."""
    try:
        con = duckdb.connect(database=':memory:')
        
        df = con.execute(query).df()
        
        if output_path.endswith(".csv"):
            df.to_csv(output_path, index=False)
        elif output_path.endswith(".parquet"):
            df.to_parquet(output_path, index=False)
        elif output_path.endswith(".json"):
            df.to_json(output_path, orient="records", indent=2)
        else:
            return "Error: Output path must end with .csv, .parquet, or .json"
            
        return f"Successfully processed data and saved to {output_path}. Rows: {len(df)}"
    except Exception as e:
        return f"Error processing data: {str(e)}"

@mcp.tool()
async def process_data(query: str, output_path: str, ctx: Context) -> str:
    """
    Execute a DuckDB SQL query to transform data and save it to a new file.
    
    CRITICAL: This is your PRIMARY tool for Feature Engineering and Data Cleaning.
    - Use this to create new features, clean dirty data, handle missing values (COALESCE), or join datasets.
    - You MUST use this tool to prepare the data before training if feature engineering is needed.
    - Example: "SELECT *, price/sqft as price_per_sqft, COALESCE(garage, 0) as garage_clean FROM 'train.csv'"
    
    IMPORTANT: Strongly RECOMMEND using '.parquet' extension for output_path (e.g. 'clean_data.parquet'). 
    - Parquet preserves data types (int, float, string, date) much better than CSV.
    - CSV often loses type information (everything becomes string or inferred incorrectly).

    - NOTE: Please use absolute paths for files in your query and for output_path to avoid path resolution errors.
    
    Args:
        query: Standard DuckDB SQL query.
        output_path: Absolute path to save the result (must be .csv, .parquet, or .json).
        
    Returns:
        Confirmation message with the output path.
    """
    try:
        await ctx.report_progress(0, 100)
        await ctx.info(f"Starting data processing task...")
        
        result = await asyncio.to_thread(_process_data_sync, query, output_path)
        
        await ctx.report_progress(100, 100)
        await ctx.info(f"Finished data processing task.")
        return result
    except Exception as e:
        return f"Error in process_data: {str(e)}"

def main():
    """Main entry point with argument parsing."""
    global EXPERIMENT_DIR, DEFAULT_SESSION_ID
    
    parser = argparse.ArgumentParser(
        description='MCP PyCaret Server - AutoML service using PyCaret',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--experiment-dir',
        type=str,
        default=EXPERIMENT_DIR,
        help='Directory to store experiment results and trained models'
    )
    parser.add_argument(
        '--session-id',
        type=int,
        default=DEFAULT_SESSION_ID,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Update module-level configuration
    EXPERIMENT_DIR = os.path.expanduser(args.experiment_dir)
    DEFAULT_SESSION_ID = args.session_id
    
    # Ensure experiment directory exists
    Path(EXPERIMENT_DIR).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting MCP PyCaret Server")
    logger.info(f"Experiment directory: {EXPERIMENT_DIR}")
    logger.info(f"Session ID: {DEFAULT_SESSION_ID}")
    
    mcp.run()

if __name__ == "__main__":
    main()
