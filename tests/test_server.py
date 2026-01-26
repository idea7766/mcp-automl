import pytest
import pandas as pd
import json
import os
import asyncio
import tempfile
import shutil
from pathlib import Path
from mcp_automl.server import (
    train_classifier,
    train_regressor,
    predict,
    inspect_data,
    query_data,
    process_data
)



class MockContext:
    """Mock Context for testing async tools."""
    
    def __init__(self):
        self.progress_reports = []
        self.info_messages = []
    
    async def report_progress(self, current: float, total: float):
        self.progress_reports.append((current, total))
    
    async def info(self, message: str):
        self.info_messages.append(message)


@pytest.fixture
def tmp_models_dir(tmp_path, monkeypatch):
    """Create a temporary models directory and patch server to use it."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    # Patch the base_dir in the _save_results function
    import mcp_automl.server as server_module
    
    original_save = server_module._save_results
    
    def patched_save_results(run_id, model, results, save_model_func, metadata, test_results=None, feature_importances=None):
        # Override to use temporary directory
        run_dir = models_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        model_path = run_dir / "model"
        save_model_func(model, str(model_path))
        
        if not results.empty:
            metrics = results.iloc[0].to_dict()
        else:
            metrics = {}
        
        test_metrics = {}
        if test_results is not None and not test_results.empty:
            try:
                test_metrics = test_results.to_dict(orient='records')[0]
            except:
                test_metrics = {}
        
        full_metadata = {**metadata, "cv_metrics": metrics, "test_metrics": test_metrics}
        
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)
        
        html_path = run_dir / "result.html"
        html_path.write_text("<html><body>Test Report</body></html>")
        
        return json.dumps({
            "run_id": run_id,
            "model_path": str(model_path) + ".pkl",
            "data_path": metadata.get("data_path"),
            "test_data_path": metadata.get("test_data_path"),
            "metadata": metrics,
            "test_metrics": test_metrics,
            "feature_importances": feature_importances if feature_importances else {},
            "report_path": str(html_path)
        }, indent=2)
    
    monkeypatch.setattr(server_module, "_save_results", patched_save_results)
    
    # Patch predict function to use tmp models_dir
    original_predict_impl = server_module.predict
    
    def patched_predict(run_id: str, data_path: str) -> str:
        try:
            run_dir = models_dir / run_id
            metadata_path = run_dir / "metadata.json"
            
            if not metadata_path.exists():
                return f"Error: Run ID {run_id} not found."
            
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            task = metadata.get("task")
            model_path = run_dir / "model"
            
            if not Path(data_path).exists():
                return f"Error: Data file not found at {data_path}"
            
            try:
                if data_path.endswith(".csv"):
                    input_data = pd.read_csv(data_path)
                elif data_path.endswith(".json"):
                    input_data = pd.read_json(data_path)
                else:
                    return "Error: Unsupported file format. Please use .csv or .json."
            except Exception as e:
                return f"Error parsing data file: {str(e)}"
            
            if task == "classification":
                from pycaret.classification import load_model, predict_model
                model = load_model(str(model_path))
                predictions = predict_model(model, data=input_data)
            elif task == "regression":
                from pycaret.regression import load_model, predict_model
                model = load_model(str(model_path))
                predictions = predict_model(model, data=input_data)
            else:
                return f"Error: Unknown task type '{task}' in metadata."
            
            predictions_dir = run_dir / "predictions"
            predictions_dir.mkdir(exist_ok=True)
            
            import uuid
            prediction_id = str(uuid.uuid4())
            prediction_file = f"prediction_{prediction_id}.json"
            prediction_path = predictions_dir / prediction_file
            
            predictions.to_json(prediction_path, orient="records", indent=2)
            
            return str(prediction_path)
        
        except Exception as e:
            return f"Error making predictions: {str(e)}"
    
    # Create a mock @mcp.tool decorated version
    import functools
    patched_predict_decorated = functools.wraps(original_predict_impl)(patched_predict)
    monkeypatch.setattr(server_module, "predict", patched_predict_decorated)
    
    yield models_dir



@pytest.fixture
def sample_classification_data(tmp_path):
    """Create sample classification dataset."""
    data = {
        'feature1': list(range(50)),
        'feature2': [i * 2 for i in range(50)],
        'feature3': [i * 0.5 for i in range(50)],
        'target': [0] * 25 + [1] * 25
    }
    df = pd.DataFrame(data)
    data_path = tmp_path / "classification_data.csv"
    df.to_csv(data_path, index=False)
    return str(data_path)


@pytest.fixture
def sample_regression_data(tmp_path):
    """Create sample regression dataset."""
    data = {
        'feature1': list(range(50)),
        'feature2': [i * 2 for i in range(50)],
        'feature3': [i * 0.5 for i in range(50)],
        'target': [float(i * 1.5) for i in range(50)]
    }
    df = pd.DataFrame(data)
    data_path = tmp_path / "regression_data.csv"
    df.to_csv(data_path, index=False)
    return str(data_path)


@pytest.fixture
def mock_context():
    """Create a mock context for async tools."""
    return MockContext()


# Test sync tools

def test_inspect_data(sample_classification_data):
    """Test inspect_data tool."""
    result = asyncio.run(inspect_data(sample_classification_data, n_rows=5))
    
    # Should return valid JSON
    data = json.loads(result)
    
    assert "structure" in data
    assert "statistics" in data
    assert "previews" in data
    
    assert data["structure"]["rows"] == 50
    assert data["structure"]["columns"] == 4
    assert "target" in data["structure"]["column_names"]
    
    assert "missing_values" in data["statistics"]
    assert "unique_values" in data["statistics"]


def test_query_data(sample_classification_data):
    """Test query_data tool."""
    query = f"SELECT COUNT(*) as count, AVG(feature1) as avg_f1 FROM '{sample_classification_data}'"
    result = asyncio.run(query_data(query))
    
    # Should return valid JSON
    data = json.loads(result)
    
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["count"] == 50
    assert data[0]["avg_f1"] == 24.5


def test_query_data_limit(sample_classification_data):
    """Test query_data respects row limit."""
    # This would return all 50 rows, but should be limited to 100
    query = f"SELECT * FROM '{sample_classification_data}'"
    result = asyncio.run(query_data(query))
    
    data = json.loads(result)
    assert len(data) == 50  # Less than limit


# Test async tools

def test_process_data(sample_classification_data, tmp_path, mock_context):
    """Test process_data tool."""
    output_path = tmp_path / "processed.parquet"
    query = f"""
        SELECT 
            feature1,
            feature2,
            feature1 + feature2 as feature_combined,
            target
        FROM '{sample_classification_data}'
    """
    
    result = asyncio.run(process_data(query, str(output_path), mock_context))
    
    assert "Successfully processed" in result
    assert output_path.exists()
    
    # Verify processed data
    df = pd.read_parquet(output_path)
    assert len(df) == 50
    assert "feature_combined" in df.columns
    assert df["feature_combined"].iloc[0] == 0  # 0 + 0


def test_train_classifier(sample_classification_data, tmp_models_dir, mock_context):
    """Test train_classifier tool."""
    result_json = asyncio.run(train_classifier(
        data_path=sample_classification_data,
        target_column="target",
        ctx=mock_context,
        fold=3,  # Reduce folds for speed
        n_features_to_select=1.0  # Use all features
    ))
    
    # Should return valid JSON
    result = json.loads(result_json)
    
    assert "run_id" in result
    assert "model_path" in result
    assert "metadata" in result
    assert "test_metrics" in result
    
    # Verify model file exists
    assert Path(result["model_path"]).exists()
    
    # Verify metadata
    run_id = result["run_id"]
    metadata_path = tmp_models_dir / run_id / "metadata.json"
    assert metadata_path.exists()
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    assert metadata["task"] == "classification"
    assert metadata["target_column"] == "target"
    
    # Verify progress was reported
    assert len(mock_context.progress_reports) > 0
    assert len(mock_context.info_messages) > 0


def test_train_regressor(sample_regression_data, tmp_models_dir, mock_context):
    """Test train_regressor tool."""
    result_json = asyncio.run(train_regressor(
        data_path=sample_regression_data,
        target_column="target",
        ctx=mock_context,
        fold=3,  # Reduce folds for speed
        n_features_to_select=1.0  # Use all features
    ))
    
    # Should return valid JSON
    result = json.loads(result_json)
    
    assert "run_id" in result
    assert "model_path" in result
    assert "metadata" in result
    assert "test_metrics" in result
    
    # Verify model file exists
    assert Path(result["model_path"]).exists()
    
    # Verify metadata
    run_id = result["run_id"]
    metadata_path = tmp_models_dir / run_id / "metadata.json"
    assert metadata_path.exists()
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    assert metadata["task"] == "regression"
    assert metadata["target_column"] == "target"


def test_train_classifier_with_test_data(sample_classification_data, tmp_path, tmp_models_dir, mock_context):
    """Test train_classifier with separate test data."""
    # Create test data
    df = pd.read_csv(sample_classification_data)
    test_df = df.sample(n=10, random_state=42)
    test_path = tmp_path / "test_data.csv"
    test_df.to_csv(test_path, index=False)
    
    result_json = asyncio.run(train_classifier(
        data_path=sample_classification_data,
        target_column="target",
        ctx=mock_context,
        test_data_path=str(test_path),
        fold=3,
        n_features_to_select=1.0
    ))
    
    result = json.loads(result_json)
    assert "test_metrics" in result
    assert result["test_data_path"] == str(test_path)



def test_predict_simple_validation(sample_classification_data, tmp_models_dir, mock_context):
    """Test predict tool with basic validation - full integration test would require actual model directory."""
    # This test validates the predict function's error handling
    # Full integration testing of predict is complex due to directory patching with FastMCP decorators
    
    # Test with invalid run_id
    result = asyncio.run(predict("invalid-run-id-12345", sample_classification_data))
    assert result.startswith("Error:")
    assert "not found" in result.lower()


def test_predict_invalid_run_id(sample_classification_data):
    """Test predict with invalid run_id."""
    result = asyncio.run(predict("invalid-run-id", sample_classification_data))
    assert result.startswith("Error:")
    assert "not found" in result


def test_inspect_data_invalid_format(tmp_path):
    """Test inspect_data with unsupported file format."""
    invalid_path = tmp_path / "data.txt"
    invalid_path.write_text("some data")
    
    result = asyncio.run(inspect_data(str(invalid_path)))
    assert "Error" in result
    # DuckDB provides its own error message for unsupported formats


def test_query_data_invalid_query():
    """Test query_data with invalid SQL."""
    result = asyncio.run(query_data("INVALID SQL QUERY"))
    assert "Error executing query:" in result


def test_inspect_data_with_na_values(tmp_path):
    """Test inspect_data with pandas NA values (NAType).
    
    Reproduces the error: "Object of type NAType is not JSON serializable"
    This happens when the dataset contains pd.NA values from nullable dtypes.
    """
    # Create data with pd.NA values using nullable Int64 dtype
    data = {
        'int_col': pd.array([1, 2, pd.NA, 4, 5], dtype="Int64"),  # Nullable Int64
        'float_col': pd.array([1.0, pd.NA, 3.0, 4.0, 5.0], dtype="Float64"),  # Nullable Float64
        'str_col': pd.array(['a', 'b', pd.NA, 'd', 'e'], dtype="string"),  # Nullable string
        'target': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    # Save as parquet to preserve nullable dtypes (CSV would convert to regular NaN)
    data_path = tmp_path / "data_with_na.parquet"
    df.to_parquet(data_path, index=False)
    
    # This should NOT raise "Object of type NAType is not JSON serializable"
    result = asyncio.run(inspect_data(str(data_path), n_rows=3))
    
    # Should return valid JSON
    parsed = json.loads(result)
    
    assert "structure" in parsed
    assert "statistics" in parsed
    assert "previews" in parsed
    
    # Verify missing values are correctly counted
    assert parsed["statistics"]["missing_values"]["int_col"] == 1
    assert parsed["statistics"]["missing_values"]["float_col"] == 1
    assert parsed["statistics"]["missing_values"]["str_col"] == 1
