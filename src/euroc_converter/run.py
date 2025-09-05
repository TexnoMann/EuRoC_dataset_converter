import click

from euroc_converter.convert import convert_dataset


@click.group()
def euroc_converter():
    """ Tools to convert EuRoC dataset into TUM or Replica formats.
    """
euroc_converter.add_command(convert_dataset)


def main():
    euroc_converter()

if __name__ == "__main__":
    main()