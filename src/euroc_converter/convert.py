import sys, os
import copy
import click
from pathlib import Path

import yaml
from euroc_converter.datasets import EuRoCDataset


@click.command(name="collect_dataset")
@click.option("--base_dir", required=True, help="Path to EuRoC dataset")
@click.option("--type", default=True, help="Type of the output dataset")
@click.option("--include_imu", default=True, help="Should be IMU data included in data")
@click.option("--config", default="config/default.yaml", help="Path to config with conversion parameters", cls=Path)
def convert_dataset(base_dir, type, include_imu, config):
    
    euroc_dataset = EuRoCDataset(basedir=base_dir, )