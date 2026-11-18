"""Command line: fluxpipe validate | prompts | diff | generate | credits | budget"""

from __future__ import annotations

import json

import typer

from .bfl_client import BFLClient, BFLError
from .bible import load_bible
from .config import IMAGE_PRICES_USD, load_settings
from .generate import DEFAULT_MODEL, DEFAULT_SIZE, keyframe_request
from .ledger import Ledger, SpendCapExceeded
from .prompts import _flatten, as_text, build_structured, reference_images
from .prompts import diff as prompt_diff


def _all_fields(a: dict, b: dict) -> set[str]:
    return _flatten(a).keys() | _flatten(b).keys()


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
def prompts(
    shot: str = typer.Option(None, help="Only this shot id."),
    as_: str = typer.Option("text", "--as", help="text (readable) or json (what the API gets)."),
) -> None:
    """Print the prompt (and attached references) each shot will be generated from."""
    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    for s in bible.shots:
        if shot and s.id != shot:
            continue
        structured = build_structured(bible, s)
        typer.secho(f"[{s.id}] {s.duration_s:g}s", bold=True)
        typer.echo(json.dumps(structured, indent=2) if as_ == "json" else as_text(structured))
        refs = reference_images(bible, s)
        typer.secho(f"  refs: {', '.join(refs) if refs else '(none yet, locked at M3)'}\n", dim=True)


@app.command()
def diff(shot_a: str, shot_b: str) -> None:
    """Show exactly which prompt fields differ between two shots. Everything else is identical."""
    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    a = build_structured(bible, bible.shot(shot_a))
    b = build_structured(bible, bible.shot(shot_b))
    changes = prompt_diff(a, b)
    if not changes:
        typer.echo("Identical prompts.")
        return
    for field, va, vb in changes:
        typer.secho(field, bold=True)
        typer.secho(f"  {shot_a}: {va}", fg="red")
        typer.secho(f"  {shot_b}: {vb}", fg="green")
    same = len(_all_fields(a, b)) - len(changes)
    typer.echo(f"\n{len(changes)} field(s) differ; {same} identical.")


@app.command()
def generate(
    shot: str,
    model: str = typer.Option(DEFAULT_MODEL, help=f"One of: {', '.join(IMAGE_PRICES_USD)}."),
    width: int = typer.Option(DEFAULT_SIZE[0]),
    height: int = typer.Option(DEFAULT_SIZE[1]),
    seed: int = typer.Option(None, help="Default: a fixed seed derived from the shot id."),
    stage: str = typer.Option("test", help="Ledger stage: test | keyframe | baseline ..."),
) -> None:
    """Generate one still for a shot. With DRY_RUN=true it is logged but not sent."""
    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    ledger = Ledger(settings.ledger_path, settings.spend_cap_usd)
    try:
        req = keyframe_request(
            bible, bible.shot(shot), model=model, width=width, height=height, seed=seed
        )
    except (ValueError, KeyError) as exc:
        typer.secho(f"  x {exc}", fg="red")
        raise typer.Exit(2)
    seed_used = req.payload["seed"]
    out = settings.assets_dir / stage / f"{shot}_{model}_s{seed_used}.png"
    typer.echo(f"{shot}: {model} {width}x{height} seed {seed_used}, estimated ${req.est_cost_usd:.3f}")

    client = BFLClient(settings.api_key, ledger, dry_run=settings.dry_run)
    before = None if settings.dry_run else client.credits()
    try:
        result = client.run_job(
            endpoint=req.endpoint, payload=req.payload, est_cost_usd=req.est_cost_usd,
            stage=stage, out_path=out, shot_id=shot,
        )
    except (SpendCapExceeded, BFLError) as exc:
        typer.secho(f"  x {exc}", fg="red")
        raise typer.Exit(1)

    if result.dry_run:
        typer.secho(f"  dry run: logged as ledger row {result.ledger_id}, nothing sent.", fg="yellow")
        return
    typer.secho(f"  saved {out.relative_to(settings.assets_dir.parent)}", fg="green")
    reported = f"${result.actual_cost_usd:.3f}" if result.actual_cost_usd is not None else "not reported"
    typer.echo(f"  cost: estimated ${req.est_cost_usd:.3f}, BFL reported {reported}")
    after = client.credits()
    typer.echo(f"  credits: {before:.2f} -> {after:.2f} (used {before - after:.2f} = ${(before - after) / 100:.3f})")


@app.command()
def credits() -> None:
    """Your BFL credit balance (1 credit = $0.01). Free to call."""
    settings = load_settings()
    client = BFLClient(settings.api_key, Ledger(settings.ledger_path, settings.spend_cap_usd))
    try:
        balance = client.credits()
        typer.echo(f"{balance:.2f} credits (${balance / 100:.2f})")
    except BFLError as exc:
        typer.secho(f"  x {exc}", fg="red")
        raise typer.Exit(1)


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
