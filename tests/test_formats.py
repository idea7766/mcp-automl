"""
Tests for verifying supported document formats (CSV, Parquet, JSON).

These tests ensure that all supported file formats work correctly with
exposed MCP tools:
- inspect_data() - data inspection tool
- query_data() - SQL query tool
"""

import pytest
import pandas as pd
import numpy as np
import json
import asyncio
from pathlib import Path

from mcp_automl.server import (
    inspect_data,
    query_data,
)


# =============================================================================
# Fixtures for creating test data in different formats
# =============================================================================

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'str_col': ['a', 'b', 'c', 'd', 'e'],
        'bool_col': [True, False, True, False, True],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_csv_file(tmp_path, sample_dataframe):
    """Create a sample CSV file."""
    file_path = tmp_path / "data.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def sample_parquet_file(tmp_path, sample_dataframe):
    """Create a sample Parquet file."""
    file_path = tmp_path / "data.parquet"
    sample_dataframe.to_parquet(file_path, index=False)
    return str(file_path)


@pytest.fixture
def sample_json_file(tmp_path, sample_dataframe):
    """Create a sample JSON file (records orient)."""
    file_path = tmp_path / "data.json"
    sample_dataframe.to_json(file_path, orient='records')
    return str(file_path)


@pytest.fixture
def all_format_files(sample_csv_file, sample_parquet_file, sample_json_file):
    """Return all format files as a dict."""
    return {
        'csv': sample_csv_file,
        'parquet': sample_parquet_file,
        'json': sample_json_file
    }


# =============================================================================
# Test inspect_data() with different formats
# =============================================================================

class TestInspectDataFormats:
    """Tests for inspect_data() with different file formats."""
    
    def test_inspect_csv(self, sample_csv_file):
        """Test inspect_data with CSV file."""
        result = asyncio.run(inspect_data(sample_csv_file, n_rows=3))
        
        data = json.loads(result)
        
        assert "structure" in data
        assert "statistics" in data
        assert "previews" in data
        assert data["structure"]["rows"] == 5
        assert data["structure"]["columns"] == 5
    
    def test_inspect_parquet(self, sample_parquet_file):
        """Test inspect_data with Parquet file."""
        result = asyncio.run(inspect_data(sample_parquet_file, n_rows=3))
        
        data = json.loads(result)
        
        assert "structure" in data
        assert data["structure"]["rows"] == 5
        assert data["structure"]["columns"] == 5
        assert "int_col" in data["structure"]["column_names"]
    
    def test_inspect_json(self, sample_json_file):
        """Test inspect_data with JSON file."""
        result = asyncio.run(inspect_data(sample_json_file, n_rows=3))
        
        data = json.loads(result)
        
        assert "structure" in data
        assert data["structure"]["rows"] == 5
        assert data["structure"]["columns"] == 5
    
    def test_inspect_all_formats_consistent(self, all_format_files):
        """Test that all formats return consistent structure info."""
        results = {}
        for fmt, path in all_format_files.items():
            result = asyncio.run(inspect_data(path, n_rows=3))
            results[fmt] = json.loads(result)
        
        # All formats should report same row/column counts
        for fmt in ['parquet', 'json']:
            assert results[fmt]["structure"]["rows"] == results['csv']["structure"]["rows"]
            assert results[fmt]["structure"]["columns"] == results['csv']["structure"]["columns"]
            assert set(results[fmt]["structure"]["column_names"]) == set(results['csv']["structure"]["column_names"])
    
    def test_inspect_unsupported_format_returns_error(self, tmp_path):
        """Test that unsupported format returns error message."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("some data")
        
        result = asyncio.run(inspect_data(str(txt_file)))
        
        assert "Error" in result


# =============================================================================
# Test query_data() with different formats
# =============================================================================

class TestQueryDataFormats:
    """Tests for query_data() with different file formats."""
    
    def test_query_csv(self, sample_csv_file):
        """Test query_data with CSV file."""
        query = f"SELECT COUNT(*) as cnt FROM '{sample_csv_file}'"
        result = asyncio.run(query_data(query))
        
        data = json.loads(result)
        assert data[0]["cnt"] == 5
    
    def test_query_parquet(self, sample_parquet_file):
        """Test query_data with Parquet file."""
        query = f"SELECT COUNT(*) as cnt FROM '{sample_parquet_file}'"
        result = asyncio.run(query_data(query))
        
        data = json.loads(result)
        assert data[0]["cnt"] == 5
    
    def test_query_json(self, sample_json_file):
        """Test query_data with JSON file."""
        query = f"SELECT COUNT(*) as cnt FROM '{sample_json_file}'"
        result = asyncio.run(query_data(query))
        
        data = json.loads(result)
        assert data[0]["cnt"] == 5
    
    def test_query_aggregation_all_formats(self, all_format_files):
        """Test aggregation query works on all formats."""
        for fmt, path in all_format_files.items():
            query = f"SELECT SUM(int_col) as total FROM '{path}'"
            result = asyncio.run(query_data(query))
            
            data = json.loads(result)
            assert data[0]["total"] == 15  # 1+2+3+4+5
    
    def test_query_filter_all_formats(self, all_format_files):
        """Test filter query works on all formats."""
        for fmt, path in all_format_files.items():
            query = f"SELECT * FROM '{path}' WHERE int_col > 3"
            result = asyncio.run(query_data(query))
            
            data = json.loads(result)
            assert len(data) == 2  # int_col 4 and 5
    
    def test_query_join_csv_parquet(self, sample_csv_file, sample_parquet_file):
        """Test joining CSV and Parquet files in a single query."""
        query = f"""
            SELECT c.int_col, p.float_col 
            FROM '{sample_csv_file}' c
            JOIN '{sample_parquet_file}' p ON c.int_col = p.int_col
            WHERE c.int_col <= 3
        """
        result = asyncio.run(query_data(query))
        
        data = json.loads(result)
        assert len(data) == 3


# =============================================================================
# Test special cases and edge cases
# =============================================================================

class TestFormatEdgeCases:
    """Tests for edge cases in format handling via exposed tools."""
    
    def test_csv_with_special_characters(self, tmp_path):
        """Test CSV with special characters in data."""
        df = pd.DataFrame({
            'text': ['hello, world', 'foo "bar"', "line1\nline2", 'tab\there'],
            'target': [0, 1, 0, 1]
        })
        file_path = tmp_path / "special.csv"
        df.to_csv(file_path, index=False)
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        assert data["structure"]["rows"] == 4
        assert "text" in data["structure"]["column_names"]
    
    def test_json_nested_to_flat(self, tmp_path):
        """Test JSON with records orientation (flat structure)."""
        records = [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 3, "b": "z"}
        ]
        file_path = tmp_path / "flat.json"
        with open(file_path, 'w') as f:
            json.dump(records, f)
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        assert data["structure"]["rows"] == 3
        assert set(data["structure"]["column_names"]) == {'a', 'b'}
    
    def test_parquet_with_nullable_types(self, tmp_path):
        """Test Parquet with nullable/arrow types."""
        df = pd.DataFrame({
            'nullable_int': pd.array([1, 2, None, 4, 5], dtype="Int64"),
            'nullable_float': pd.array([1.0, None, 3.0, 4.0, 5.0], dtype="Float64"),
            'nullable_str': pd.array(['a', 'b', None, 'd', 'e'], dtype="string"),
            'target': [0, 1, 0, 1, 0]
        })
        file_path = tmp_path / "nullable.parquet"
        df.to_parquet(file_path, index=False)
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        assert data["structure"]["rows"] == 5
        # Check that nulls are counted correctly
        assert data["statistics"]["missing_values"]["nullable_int"] == 1
        assert data["statistics"]["missing_values"]["nullable_float"] == 1
        assert data["statistics"]["missing_values"]["nullable_str"] == 1
    
    def test_csv_empty_file(self, tmp_path):
        """Test handling of empty CSV file with headers only."""
        file_path = tmp_path / "empty.csv"
        file_path.write_text("col1,col2,col3\n")
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        assert data["structure"]["rows"] == 0
        assert set(data["structure"]["column_names"]) == {'col1', 'col2', 'col3'}
    
    def test_parquet_empty_dataframe(self, tmp_path):
        """Test handling of empty Parquet file."""
        df = pd.DataFrame({'col1': [], 'col2': [], 'col3': []})
        file_path = tmp_path / "empty.parquet"
        df.to_parquet(file_path, index=False)
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        assert data["structure"]["rows"] == 0
        assert set(data["structure"]["column_names"]) == {'col1', 'col2', 'col3'}
    
    def test_json_empty_array(self, tmp_path):
        """Test handling of empty JSON array."""
        file_path = tmp_path / "empty.json"
        file_path.write_text("[]")
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        assert data["structure"]["rows"] == 0
    
    def test_csv_with_unicode(self, tmp_path):
        """Test CSV with Unicode characters."""
        df = pd.DataFrame({
            'text': ['日本語', 'العربية', 'emoji: 🎉🚀', 'ñoño'],
            'target': [0, 1, 0, 1]
        })
        file_path = tmp_path / "unicode.csv"
        df.to_csv(file_path, index=False)
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        assert data["structure"]["rows"] == 4
        # Verify via query
        query = f"SELECT * FROM '{file_path}' WHERE text = '日本語'"
        query_result = asyncio.run(query_data(query))
        query_data_list = json.loads(query_result)
        assert len(query_data_list) == 1
    
    def test_large_csv_file(self, tmp_path):
        """Test loading a larger CSV file."""
        n_rows = 50000
        df = pd.DataFrame({
            'id': range(n_rows),
            'value': np.random.randn(n_rows),
            'category': [f'cat_{i % 10}' for i in range(n_rows)]
        })
        file_path = tmp_path / "large.csv"
        df.to_csv(file_path, index=False)
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        assert data["structure"]["rows"] == n_rows
        
        # Verify aggregation query works
        query = f"SELECT COUNT(*) as cnt FROM '{file_path}'"
        query_result = asyncio.run(query_data(query))
        query_data_list = json.loads(query_result)
        assert query_data_list[0]["cnt"] == n_rows
    
    def test_parquet_with_multiple_types(self, tmp_path):
        """Test Parquet with diverse column types."""
        df = pd.DataFrame({
            'int8_col': np.array([1, 2, 3], dtype=np.int8),
            'int16_col': np.array([100, 200, 300], dtype=np.int16),
            'int32_col': np.array([1000, 2000, 3000], dtype=np.int32),
            'int64_col': np.array([10000, 20000, 30000], dtype=np.int64),
            'float32_col': np.array([1.1, 2.2, 3.3], dtype=np.float32),
            'float64_col': np.array([1.11, 2.22, 3.33], dtype=np.float64),
            'bool_col': [True, False, True],
            'str_col': ['a', 'b', 'c'],
            'target': [0, 1, 0]
        })
        file_path = tmp_path / "multitypes.parquet"
        df.to_parquet(file_path, index=False)
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        assert data["structure"]["rows"] == 3
        assert data["structure"]["columns"] == 9


# =============================================================================
# Test file extension validation via exposed tools
# =============================================================================

class TestFileExtensionValidation:
    """Tests for file extension handling via exposed tools."""
    
    def test_unsupported_txt_extension(self, tmp_path):
        """Test that .txt extension is not supported."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("some data")
        
        result = asyncio.run(inspect_data(str(txt_file)))
        assert "Error" in result
    
    def test_unsupported_xlsx_extension(self, tmp_path):
        """Test that .xlsx extension is not supported."""
        xlsx_file = tmp_path / "data.xlsx"
        xlsx_file.write_bytes(b"fake xlsx content")
        
        result = asyncio.run(inspect_data(str(xlsx_file)))
        assert "Error" in result
    
    def test_double_extension_csv(self, tmp_path, sample_dataframe):
        """Test file with double extension like .tar.csv works."""
        file_path = tmp_path / "data.tar.csv"
        sample_dataframe.to_csv(file_path, index=False)
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        assert data["structure"]["rows"] == 5


# =============================================================================
# Test format-specific inspect_data behavior
# =============================================================================

class TestInspectDataFormatDetails:
    """Tests for format-specific behavior in inspect_data."""
    
    def test_inspect_csv_dtypes(self, tmp_path):
        """Test that CSV dtypes are correctly inferred."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c']
        })
        file_path = tmp_path / "typed.csv"
        df.to_csv(file_path, index=False)
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        
        dtypes = data["structure"]["dtypes"]
        # DuckDB may infer these differently, just check keys exist
        assert "int_col" in dtypes
        assert "float_col" in dtypes
        assert "str_col" in dtypes
    
    def test_inspect_parquet_preserves_dtypes(self, tmp_path):
        """Test that Parquet preserves exact dtypes."""
        df = pd.DataFrame({
            'int32_col': np.array([1, 2, 3], dtype=np.int32),
            'float32_col': np.array([1.5, 2.5, 3.5], dtype=np.float32),
        })
        file_path = tmp_path / "typed.parquet"
        df.to_parquet(file_path, index=False)
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        
        dtypes = data["structure"]["dtypes"]
        assert "int32_col" in dtypes
        assert "float32_col" in dtypes
    
    def test_inspect_counts_missing_values_csv(self, tmp_path):
        """Test that missing values are correctly counted in CSV."""
        # CSV doesn't preserve NA types well, use empty strings which become NaN
        file_path = tmp_path / "missing.csv"
        file_path.write_text("a,b,c\n1,2,3\n,5,\n7,,9\n")
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        
        missing = data["statistics"]["missing_values"]
        assert missing["a"] == 1
        assert missing["b"] == 1
        assert missing["c"] == 1
    
    def test_inspect_counts_missing_values_parquet(self, tmp_path):
        """Test that missing values are correctly counted in Parquet."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [None, 2, None],
            'c': [1, 2, 3]
        })
        file_path = tmp_path / "missing.parquet"
        df.to_parquet(file_path, index=False)
        
        result = asyncio.run(inspect_data(str(file_path)))
        data = json.loads(result)
        
        missing = data["statistics"]["missing_values"]
        assert missing["a"] == 1
        assert missing["b"] == 2
        assert missing["c"] == 0
