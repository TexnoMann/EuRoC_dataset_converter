import click
import gdown
import zipfile
import pathlib
import os

EuRoC_DATASET_LINKS = {
    "MH_01_easy": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip",
    "MH_02_easy": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip",
    "MH_03_medium": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.zip",
    "MH_04_difficult": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_04_difficult/MH_04_difficult.zip",
    "MH_05_difficult": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip",
    "V1_01_easy": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip",
    "V1_02_medium": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip",
    "V1_03_difficult": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_03_difficult/V1_03_difficult.zip",
    "V2_01_easy": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_01_easy/V2_01_easy.zip",
    "V2_02_medium": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip",
    "V2_03_difficult": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_03_difficult/V2_03_difficult.zip"
}

@click.command(name="download_dataset")
@click.option("--output_dir", required=True, help="Path to save the downloaded dataset")
@click.option("--dataset_name", required=True, type=click.Choice(EuRoC_DATASET_LINKS.keys()),help="Name of the EuRoC dataset to download (e.g., 'MH_01_easy')")
def download_dataset(output_dir, dataset_name):
    """
    Downloads a specified EuRoC MAV dataset from the internet.
    """
    url = EuRoC_DATASET_LINKS[dataset_name]

    output_dir = pathlib.Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir/f"{dataset_name}.zip"
    
    click.echo(f"Downloading {dataset_name} dataset...")
    gdown.download(url, str(output_path), quiet=False)
    
    click.echo(f"Extracting {dataset_name} dataset...")
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(str(output_dir/f"{dataset_name}"))

    os.remove(output_path) # Remove the zip file after extraction
    
    click.echo(f"Dataset {dataset_name} downloaded and extracted to {output_dir}")
    click.echo("Done.")

if __name__ == "__main__":
    download_dataset()