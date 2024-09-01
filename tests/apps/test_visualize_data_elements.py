import subprocess
import tempfile
from pathlib import Path

apps_directory_path = Path(__file__).parents[2] / "apps"
hydra_test_config_folder = Path(__file__).parent / "configs"


def test_visualize_data_elements():
    """Test the visualize_data_elements.py script.

    Just check this script can be executed without any errors.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        command = (
            f"python {apps_directory_path / 'visualize_dataset_elements.py'} "
            f"-cd {hydra_test_config_folder} "
            "-cn test_visualize_dataset_elements "
            f"hydra.run.dir={tmp_dir} "
        )
        subprocess.check_call(command, shell=True)
