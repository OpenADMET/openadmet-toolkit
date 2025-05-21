import click

from openadmet.toolkit.cli.cofolding import cofolding


@click.group()
def cli():
    """OpenADMET CLI"""
    pass


cli.add_command(cofolding)
