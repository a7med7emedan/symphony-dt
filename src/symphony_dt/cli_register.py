from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from symphony_dt.aggregate.all import aggregate_all_results
from symphony_dt.logging import configure_logging
from symphony_dt.parity import parity_check_run_dir

console = Console()


def register_module24_commands(app: typer.Typer) -> None:
    @app.command()
    def aggregate_all(
        runs_root: Path = typer.Option(..., "--runs-root", exists=True, file_okay=False),
        out_dir: Path = typer.Option(..., "--out-dir"),
    ) -> None:
        configure_logging("INFO")
        written = aggregate_all_results(runs_root=runs_root, out_dir=out_dir)
        console.print(f"[bold green]OK[/bold green] Aggregated {len(written)} tables into: {out_dir}")

    @app.command()
    def parity_check(
        run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
        notebook: Optional[Path] = typer.Option(None, "--notebook"),
    ) -> None:
        configure_logging("INFO")
        res = parity_check_run_dir(run_dir=run_dir, notebook=notebook)
        if res.passed:
            console.print("[bold green]OK[/bold green] Parity check passed.")
            return
        if res.missing_files:
            console.print("[bold red]Missing files:[/bold red] " + ", ".join(res.missing_files))
        if res.errors:
            for k, v in res.errors.items():
                console.print(f"[bold red]Schema error[/bold red] {k}: {v}")
        raise typer.Exit(code=2)
