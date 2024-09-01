import glob
import subprocess
import tempfile
from pathlib import Path

apps_directory_path = Path(__file__).parents[2] / "apps"
hydra_test_config_folder = Path(__file__).parent / "configs"


def test_training_inference():
    """Test the training and inference scripts.

    Just check these scripts can be executed without any errors.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        command = (
            f"python {apps_directory_path / 'train.py'} "
            f"-cd {hydra_test_config_folder} "
            "-cn test_train "
            f"hydra.run.dir={str(Path(tmp_dir) / '1')} "
        )
        subprocess.check_call(command, shell=True)

        best_model_checkpoints = glob.glob(
            str(Path(tmp_dir) / "1/checkpoints/best*.ckpt")
        )
        assert len(best_model_checkpoints) == 1
        best_model_checkpoint = best_model_checkpoints[0].replace("=", r"\=")

        command = (
            f"python {apps_directory_path / 'inference.py'} "
            f"-cd {hydra_test_config_folder} "
            "-cn test_inference "
            f"'checkpoint={best_model_checkpoint}' "
            f"hydra.run.dir={str(Path(tmp_dir) / '2')} "
        )
        subprocess.check_call(command, shell=True)

        command = (
            f"python {apps_directory_path / 'convert_model_to_onnx.py'} "
            f"-cd {hydra_test_config_folder} "
            "-cn test_convert_model_to_onnx "
            f"'checkpoint={best_model_checkpoint}' "
            f"hydra.run.dir={str(Path(tmp_dir) / '3')} "
        )
        subprocess.check_call(command, shell=True)

        command = (
            f"python {apps_directory_path / 'inference_onnx.py'} "
            f"-cd {hydra_test_config_folder} "
            "-cn test_inference_onnx "
            f"onnx_file_path={str(Path(tmp_dir) / '3/model.onnx')} "
            f"hydra.run.dir={str(Path(tmp_dir) / '4')} "
        )
        subprocess.check_call(command, shell=True)
