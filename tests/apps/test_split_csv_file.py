import shutil
import subprocess
import tempfile
from pathlib import Path

apps_directory_path = Path(__file__).parents[2] / "apps"
test_data_path = Path(__file__).parents[1] / "data"


def test_split_csv_file():
    """Test the split_csv_file.py script.

    Just check this script can be executed without any errors.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        shutil.copy(
            test_data_path / "toy_dataset" / "labels/labels.csv",
            Path(tmp_dir) / "labels.csv",
        )
        command = (
            f"python {apps_directory_path / 'split_csv_file.py'} "
            f"--csv_path {Path(tmp_dir) / 'labels.csv'} "
        )
        subprocess.check_call(command, shell=True)
