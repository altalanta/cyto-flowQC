"""Unit tests for the Streamlit launcher."""
import unittest.mock
from pathlib import Path
from streamlit.testing.v1 import AppTest
import pytest

# Mark all tests in this file as streamlit apps
pytestmark = pytest.mark.streamlit

@pytest.fixture
def mock_popen():
    """Fixture to mock subprocess.Popen."""
    with unittest.mock.patch("subprocess.Popen") as mock_popen:
        process_mock = unittest.mock.Mock()
        attrs = {"stdout.readline.side_effect": ["log line 1\n", "log line 2\n", ""], "wait.return_value": 0, "returncode": 0}
        process_mock.configure_mock(**attrs)
        mock_popen.return_value = process_mock
        yield mock_popen, process_mock

def test_launcher_runs_pipeline(mock_popen):
    """Test that the launcher constructs and runs the pipeline command correctly."""
    mock_popen_fn, process_mock = mock_popen
    
    at = AppTest.from_file("src/cytoflow_qc/launcher.py")
    at.run()

    # Simulate file uploads
    at.sidebar.file_uploader[0].set_value(b"sample_id,file_path\n1,a.fcs", "samplesheet.csv")
    at.sidebar.file_uploader[1].set_value(b"channels: {}", "config.yaml")
    
    # Simulate button click
    at.sidebar.button[0].click().run()

    # Check that the pipeline was called
    assert mock_popen_fn.called
    
    # Check that the success message is displayed
    assert "Pipeline completed successfully!" in at.success[0].value

def test_launcher_handles_error(mock_popen):
    """Test that the launcher displays an error message on pipeline failure."""
    mock_popen_fn, process_mock = mock_popen
    process_mock.returncode = 1 # Simulate a failed run
    
    at = AppTest.from_file("src/cytoflow_qc/launcher.py")
    at.run()

    at.sidebar.file_uploader[0].set_value(b"sample_id,file_path\n1,a.fcs", "samplesheet.csv")
    at.sidebar.file_uploader[1].set_value(b"channels: {}", "config.yaml")
    at.sidebar.button[0].click().run()

    assert "Pipeline failed with exit code 1." in at.error[0].value
