"""Command line: fluxpipe validate | prompts | budget"""

from __future__ import annotations

import typer

from .bible import load_bible
from .config import load_settings
from .ledger import Ledger
from .prompts import build_prompt, reference_images

app = typer.Typer(help="A consistency pipeline for FLUX.", no_args_is_help=True)


@app.command()
def validate() -> None:
    """Check the story bible: every shot's character, costume and location must exist."""
    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    problems = bible.problems()
    if problems:
        for p in problems:
            typer.secho(f"  x {p}", fg="red")
        raise typer.Exit(1)
    typer.secho(
        f"Bible OK: {len(bible.characters)} character(s), {len(bible.locations)} location(s),"
        f" {len(bible.shots)} shot(s).",
        fg="green",
    )


@app.command()
def prompts(shot: str = typer.Option(None, help="Only this shot id.")) -> None:
    """Print the prompt (and attached references) each shot will be generated from."""
    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    for s in bible.shots:
        if shot and s.id != shot:
            continue
        typer.secho(f"[{s.id}] {s.duration_s:g}s", bold=True)
        typer.echo(build_prompt(bible, s))
        refs = reference_images(bible, s)
        typer.secho(f"  refs: {', '.join(refs) if refs else '(none yet, locked at M3)'}\n", dim=True)


@app.command()
def budget() -> None:
    """Show estimated spend so far against the cap."""
    settings = load_settings()
    ledger = Ledger(settings.ledger_path, settings.spend_cap_usd)
    rows = ledger.summary_by_stage()
    if rows:
        for r in rows:
            typer.echo(f"  {r['stage']:<10} {r['status']:<8} {r['calls']:>4} calls  ${r['usd']:.2f}")
    else:
        typer.echo("  No calls yet.")
    typer.echo(
        f"Spent ~${ledger.total_spent():.2f} of ${settings.spend_cap_usd:.2f}"
        f" (${ledger.remaining():.2f} left). Dry run: {settings.dry_run}."
    )


if __name__ == "__main__":
    app()
