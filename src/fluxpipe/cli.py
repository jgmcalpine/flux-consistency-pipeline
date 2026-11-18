"""Command line: fluxpipe validate | prompts | diff | generate | baseline | score | refs | lock | credits | budget"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .bfl_client import BFLClient, BFLError
from .bible import load_bible
from .config import IMAGE_PRICES_USD, load_settings
from .generate import BASELINE_MODEL, DEFAULT_MODEL, DEFAULT_SIZE, baseline_request, keyframe_request
from .ledger import Ledger, SpendCapExceeded
from .refs import candidates as ref_candidates
from .refs import lock as lock_ref
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
        f" {len(bible.props)} prop(s), {len(bible.shots)} shot(s).",
        fg="green",
    )


@app.command()
def prompts(
    shot: str = typer.Option(None, help="Only this shot id."),
    as_: str = typer.Option("text", "--as", help="text (readable) or json (what the API gets)."),
    baseline: bool = typer.Option(False, "--baseline", help="Show the naive M4 prompts instead."),
) -> None:
    """Print the prompt (and attached references) each shot will be generated from."""
    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    for s in bible.shots:
        if shot and s.id != shot:
            continue
        if baseline:
            if bible.baseline is None:
                typer.secho("  x no bible/baseline.yaml", fg="red")
                raise typer.Exit(2)
            typer.secho(f"[{s.id}] baseline", bold=True)
            typer.echo(bible.baseline.prompt(s.id))
            typer.secho("  refs: none (that's the point)\n", dim=True)
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
    typer.secho("  (the balance from `fluxpipe credits` takes a minute or two to catch up)", dim=True)


@app.command()
def baseline(
    shot: str = typer.Option(None, help="Only this shot id (default: all of them)."),
    model: str = typer.Option(BASELINE_MODEL, help="Keep it the same as the keyframes'."),
) -> None:
    """M4: the naive "before" stills. Plain prompts, no references, same model/size/seed as keyframes."""
    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    try:
        shots = [bible.shot(shot)] if shot else bible.shots
        reqs = [(s, baseline_request(bible, s, model=model)) for s in shots]
    except (ValueError, KeyError) as exc:
        typer.secho(f"  x {exc}", fg="red")
        raise typer.Exit(2)
    total = sum(r.est_cost_usd for _, r in reqs)
    typer.echo(f"{len(reqs)} baseline still(s) on {model}: ~${total:.3f}")

    client = BFLClient(settings.api_key, Ledger(settings.ledger_path, settings.spend_cap_usd),
                       dry_run=settings.dry_run)
    root = settings.assets_dir.parent
    for s, req in reqs:
        out = settings.assets_dir / "baseline" / f"{s.id}_{model}_s{req.payload['seed']}.png"
        try:
            result = client.run_job(
                endpoint=req.endpoint, payload=req.payload, est_cost_usd=req.est_cost_usd,
                stage="baseline", out_path=out, shot_id=s.id,
            )
        except (SpendCapExceeded, BFLError) as exc:
            typer.secho(f"  x {s.id}: {exc}", fg="red")
            raise typer.Exit(1)
        tag = f"(dry run, ledger row {result.ledger_id}) " if result.dry_run else ""
        typer.echo(f"  {tag}{out.relative_to(root)}  ~${req.est_cost_usd:.3f}")


@app.command()
def score(
    folder: Path = typer.Argument(..., help="A folder of stills, e.g. results/baseline"),
    character: str = typer.Option("ismael", help="Whose locked face to compare against."),
) -> None:
    """Face identity QA: cosine similarity of each still's face to the locked face. Free (runs locally)."""
    from .qa.face import SAME_PERSON, SMALL_FACE_PX, insightface_detector, score_images

    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    char = bible.characters.get(character)
    if char is None or not char.reference_images:
        typer.secho(f"  x no locked face for '{character}'", fg="red")
        raise typer.Exit(2)
    images = sorted(p for p in folder.glob("*.png"))
    if not images:
        typer.secho(f"  x no .png files in {folder}", fg="red")
        raise typer.Exit(2)

    typer.echo("Loading InsightFace (buffalo_l)...")
    scores, pairs = score_images(insightface_detector(), bible.ref_path(char.reference_images[0]), images)
    colour = {"same": "green", "different": "red", "too small": "yellow", "no face": "yellow"}
    for sc in scores:
        sim = f"{sc.similarity:.3f}" if sc.similarity is not None else "  -  "
        px = f"{sc.height_px}px" if sc.height_px else "-"
        typer.secho(f"  {sc.image:<32} {sim}  face {px:>6}  {sc.verdict}", fg=colour[sc.verdict])
    judged = [sc.similarity for sc in scores if sc.verdict in ("same", "different")]
    if judged:
        typer.echo(f"Mean over {len(judged)} judgeable face(s): {sum(judged) / len(judged):.3f}"
                   f"  (same person >= {SAME_PERSON}; faces under {SMALL_FACE_PX}px not judged)")
    if pairs:
        typer.echo(f"Shot vs shot ({len(pairs)} pairs): mean {sum(pairs) / len(pairs):.3f},"
                   f" min {min(pairs):.3f}, max {max(pairs):.3f}  (is it the same man in every shot?)")
    out = folder / "faces.json"
    out.write_text(json.dumps({
        "reference": char.reference_images[0],
        "same_person_threshold": SAME_PERSON,
        "shots": [sc.as_dict() for sc in scores],
        "pairwise": pairs,
    }, indent=2) + "\n")
    typer.secho(f"  wrote {out}", dim=True)


@app.command()
def refs(
    kind: str = typer.Argument(..., help="face | sheet | plate | prop"),
    target: str = typer.Argument(..., help="Character id (face, sheet), location id (plate) or prop id (prop)."),
    costume: str = typer.Option(None, help="For sheet: which costume."),
    with_: list[str] = typer.Option(None, "--with", help="For prop: a locked prop to show beside it."),
    n: int = typer.Option(4, help="How many candidates (different seeds) to make."),
    model: str = typer.Option("flux-2-pro"),
) -> None:
    """Make candidate reference images to choose from. Then lock your favourite with `fluxpipe lock`."""
    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    try:
        cands = ref_candidates(
            bible, kind, target, out_dir=settings.assets_dir / "refs", n=n, model=model,
            costume=costume, with_props=with_,
        )
    except (ValueError, KeyError) as exc:
        typer.secho(f"  x {exc}", fg="red")
        raise typer.Exit(2)
    total = sum(c.request.est_cost_usd for c in cands)
    typer.echo(f"{n} x {kind} for {target}{f' ({costume})' if costume else ''} on {model}: ~${total:.3f}")

    client = BFLClient(settings.api_key, Ledger(settings.ledger_path, settings.spend_cap_usd),
                       dry_run=settings.dry_run)
    for c in cands:
        try:
            result = client.run_job(
                endpoint=c.request.endpoint, payload=c.request.payload,
                est_cost_usd=c.request.est_cost_usd, stage="refs",
                out_path=c.out, shot_id=f"{c.target}:{c.out.name.split('_s')[0]}", refs=c.request.refs,
            )
        except (SpendCapExceeded, BFLError) as exc:
            typer.secho(f"  x {exc}", fg="red")
            raise typer.Exit(1)
        root = settings.assets_dir.parent
        typer.echo(f"  {'(dry run) ' if result.dry_run else ''}{c.out.relative_to(root)}")
    if not settings.dry_run:
        typer.echo("Open them, pick one, then: fluxpipe lock <path>")


@app.command()
def lock(candidate: Path) -> None:
    """Lock a chosen candidate as the canonical reference: copied into bible/refs/, recorded in the YAML."""
    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    try:
        what, warnings = lock_ref(bible, settings.bible_dir, candidate.resolve())
    except (ValueError, KeyError) as exc:
        typer.secho(f"  x {exc}", fg="red")
        raise typer.Exit(2)
    typer.secho(f"Locked {what}", fg="green")
    for w in warnings:
        typer.secho(f"  ! {w}", fg="yellow")


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
