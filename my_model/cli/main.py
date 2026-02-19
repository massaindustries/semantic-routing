import typer

app = typer.Typer()


def main():
    """Entry point for the CLI when invoked via console script."""
    app()

@app.command()
def hello(name: str = "world"):
    """Simple hello command for testing the CLI entrypoint."""
    typer.echo(f"Hello, {name}!")

if __name__ == "__main__":
    app()
