import sys, os
import copy
import click
from pathlib import Path

import yaml

@click.command(name="convert_dataset")
@click.option("--base_dir", required=True, help="Path to EuRoC dataset")
@click.option("--type", default=True, help="Type of the output dataset")
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),  help="Path to config with conversion parameters")
def convert_dataset(base_dir, type, config):
    from euroc_converter.datasets import EuRoCDataset
    try:
        with open(config, 'r') as file:
            config_data = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: config not found by path: '{config}'.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        exit()

    euroc_dataset = EuRoCDataset(basedir=base_dir, cfg=config_data)
    euroc_dataset.parse()