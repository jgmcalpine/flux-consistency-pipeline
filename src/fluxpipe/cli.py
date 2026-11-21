"""Command line: fluxpipe validate | prompts | diff | generate | baseline | keyframes | edit | pick | video | score-clip | qa | demo | cut | report | score | refs | lock | credits | budget"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer

from .bfl_client import BFLClient, BFLError
from .bible import load_bible
from .config import IMAGE_PRICES_USD, load_settings
from .generate import (
    BASELINE_MODEL, DEFAULT_MODEL, DEFAULT_SIZE, baseline_request, edit_request, fingerprint,
    keyframe_request, shot_seed,
)
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
    shot: list[str] = typer.Option(None, help="Only these shot ids (repeat --shot). Default: all."),
    model: str = typer.Option(BASELINE_MODEL, help="Keep it the same as the keyframes'."),
) -> None:
    """M4: the naive "before" stills. Plain prompts, no references, same model/size/seed as keyframes."""
    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    try:
        shots = [bible.shot(i) for i in shot] if shot else bible.shots
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
def keyframes(
    shot: list[str] = typer.Option(None, help="Only these shot ids (repeat --shot). Default: all."),
    n: int = typer.Option(3, help="Candidates per round (different seeds)."),
    max_rounds: int = typer.Option(2, help="Rounds per shot if no candidate passes face QA."),
    model: str = typer.Option(BASELINE_MODEL, help="Keep it the same as the baseline's."),
    redo: bool = typer.Option(False, "--redo", help="Regenerate even if the pick is up to date."),
) -> None:
    """M5: N candidates per shot with the locked refs; face QA picks the best; capped retries."""
    from .keyframes import BY_EYE, DRY_RUN, PASS, run_shot, summary

    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    picks_dir = settings.assets_dir.parent / "results" / "keyframes"
    picks_file = picks_dir / "picks.json"
    picks = json.loads(picks_file.read_text()) if picks_file.is_file() else {}
    try:
        shots = [bible.shot(i) for i in shot] if shot else bible.shots
        reqs = {s.id: keyframe_request(bible, s, model=model) for s in shots}
        prints = {i: fingerprint(r.endpoint, r.payload, r.refs) for i, r in reqs.items()}
        if not redo:
            # A pick is up to date only if it came from exactly today's request (prompt, settings, refs).
            done = [s.id for s in shots if picks.get(s.id, {}).get("fingerprint") == prints[s.id]]
            if done:
                typer.secho(f"Up to date, skipping: {', '.join(done)} (--redo to regenerate).", dim=True)
            stale = [s.id for s in shots if s.id in picks and s.id not in done]
            if stale:
                typer.secho(f"Bible changed since the pick, regenerating: {', '.join(stale)}", fg="yellow")
            shots = [s for s in shots if s.id not in done]
        if not shots:
            typer.echo("Nothing to do.")
            return
        per_call = {s.id: reqs[s.id].est_cost_usd for s in shots}
    except (ValueError, KeyError) as exc:
        typer.secho(f"  x {exc}", fg="red")
        raise typer.Exit(2)
    first = n * sum(per_call.values())
    typer.echo(f"{len(shots)} shot(s) x {n} candidates on {model}: ~${first:.2f}"
               f" (worst case with {max_rounds} rounds: ~${first * max_rounds:.2f})")

    ledger = Ledger(settings.ledger_path, settings.spend_cap_usd)
    client = BFLClient(settings.api_key, ledger, dry_run=settings.dry_run)
    detect = None
    if not settings.dry_run:
        from .qa.face import insightface_detector, reference_face, score_one

        detect = insightface_detector()
        ref_faces: dict = {}  # character id -> its locked face's embedding

        def judge_for(char_id: str):
            if char_id not in ref_faces:
                face_ref = bible.characters[char_id].reference_images[0]
                ref_faces[char_id] = reference_face(detect, bible.ref_path(face_ref))
            return lambda path: score_one(detect, ref_faces[char_id], path)[0]

    work = settings.assets_dir / "keyframes"
    root = settings.assets_dir.parent
    colour = {PASS: "green", BY_EYE: "yellow", DRY_RUN: "yellow"}

    for s in shots:
        def generate(seed: int, s=s) -> Path | None:
            req = keyframe_request(bible, s, model=model, seed=seed)
            out = work / s.id / f"{s.id}_{model}_s{seed}.png"
            res = client.run_job(endpoint=req.endpoint, payload=req.payload,
                                 est_cost_usd=req.est_cost_usd, stage="keyframe",
                                 out_path=out, shot_id=s.id, refs=req.refs)
            if res.dry_run:
                typer.secho(f"  (dry run, ledger row {res.ledger_id}) {s.id} seed {seed}"
                            f"  ~${req.est_cost_usd:.3f}", dim=True)
                return None
            return out

        try:
            result = run_shot(s.id, n=n, max_rounds=max_rounds, generate=generate,
                              judge=judge_for(s.character) if detect else None,
                              expect_face=s.face)
        except (SpendCapExceeded, BFLError) as exc:
            typer.secho(f"  x {s.id}: {exc}", fg="red")
            raise typer.Exit(1)
        if result.status == DRY_RUN:
            continue
        for c in result.candidates:
            sim = f"{c.face.similarity:.3f}" if c.face and c.face.similarity is not None else "  -  "
            mark = "*" if c is result.pick else " "
            typer.echo(f"   {mark} {c.path.relative_to(root)}  {sim}  {c.face.verdict if c.face else ''}")
        typer.secho(f"  {s.id}: {result.status}", fg=colour.get(result.status, "red"), bold=True)
        picks_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result.pick.path, picks_dir / f"{s.id}.png")
        picks[s.id] = summary(result, prints[s.id])
        picks_file.write_text(json.dumps(dict(sorted(picks.items())), indent=2) + "\n")

    if not settings.dry_run:
        typer.echo(f"Picks copied to {picks_dir.relative_to(root)}/ ({picks_file.name} has every score)."
                   f" Look at them, especially 'by eye' and 'below' shots.")


@app.command()
def edit(
    shot: str = typer.Argument(..., help="Shot id with an `edit` in shots.yaml, e.g. s11"),
    n: int = typer.Option(2, help="How many edit candidates (different seeds)."),
) -> None:
    """Apply the shot's `edit` to its current pick (FLUX.2 editing). Choose one with `fluxpipe pick`."""
    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    picks_dir = settings.assets_dir.parent / "results" / "keyframes"
    picks_file = picks_dir / "picks.json"
    picks = json.loads(picks_file.read_text()) if picks_file.is_file() else {}
    try:
        s = bible.shot(shot)
        entry = picks[shot]
    except KeyError as exc:
        typer.secho(f"  x {exc} (no shot, or no keyframe pick yet: run `fluxpipe keyframes` first)", fg="red")
        raise typer.Exit(2)
    current = keyframe_request(bible, s, model=BASELINE_MODEL)
    if entry.get("fingerprint") != fingerprint(current.endpoint, current.payload, current.refs):
        typer.secho(f"  x {shot}'s keyframe is out of date with the bible: run `fluxpipe keyframes` first",
                    fg="red")
        raise typer.Exit(2)
    # Always edit the generated keyframe, never an earlier edit: edits don't stack up drift.
    generated = [c["image"] for c in entry["candidates"]]
    base = entry["pick"] if entry["pick"] in generated else entry["source"]
    source = settings.assets_dir / "keyframes" / shot / base
    seeds = [(shot_seed(shot) + 1000 + i) % 2**32 for i in range(n)]
    try:
        extra = [bible.ref_path(r) for r in s.edit_refs]
        reqs = [edit_request(s, source, seed=seed, extra=extra) for seed in seeds]
    except ValueError as exc:
        typer.secho(f"  x {exc}", fg="red")
        raise typer.Exit(2)
    typer.echo(f"{n} edit(s) of {source.name}: ~${sum(r.est_cost_usd for r in reqs):.3f}")
    typer.secho(f"  prompt: {reqs[0].payload['prompt']}", dim=True)

    client = BFLClient(settings.api_key, Ledger(settings.ledger_path, settings.spend_cap_usd),
                       dry_run=settings.dry_run)
    made = []
    for seed, req in zip(seeds, reqs):
        out = settings.assets_dir / "keyframes" / shot / f"{shot}_edit_s{seed}.png"
        try:
            res = client.run_job(endpoint=req.endpoint, payload=req.payload, est_cost_usd=req.est_cost_usd,
                                 stage="edit", out_path=out, shot_id=shot, refs=req.refs)
        except (SpendCapExceeded, BFLError) as exc:
            typer.secho(f"  x {exc}", fg="red")
            raise typer.Exit(1)
        if res.dry_run:
            typer.secho(f"  (dry run, ledger row {res.ledger_id}) seed {seed}  ~${req.est_cost_usd:.3f}", dim=True)
            continue
        made.append({"seed": seed, "image": out.name, "edit_of": source.name})
        typer.echo(f"  {out.relative_to(settings.assets_dir.parent)}")
    if made:
        entry["source"] = source.name
        entry["edits"] = [e for e in entry.get("edits", []) if e["image"] not in {m["image"] for m in made}] + made
        picks_file.write_text(json.dumps(dict(sorted(picks.items())), indent=2) + "\n")
        typer.echo(f"Look at them, then: fluxpipe pick {shot} <file>")


@app.command()
def pick(
    shot: str = typer.Argument(..., help="Shot id, e.g. s09"),
    image: str = typer.Argument(..., help="A candidate's file name or path, e.g. s09_flux-2-pro_s699773409.png"),
    note: str = typer.Option("", help="Why (kept in picks.json)."),
) -> None:
    """Override a keyframe pick by eye: copy that candidate to results/keyframes/ and record it."""
    image = Path(image).name  # a path copied from the terminal works as well as the bare name
    settings = load_settings()
    picks_dir = settings.assets_dir.parent / "results" / "keyframes"
    picks_file = picks_dir / "picks.json"
    picks = json.loads(picks_file.read_text()) if picks_file.is_file() else {}
    entry = picks.get(shot)
    choices = [] if entry is None else [c["image"] for c in entry["candidates"] + entry.get("edits", [])]
    if image not in choices:
        typer.secho(f"  x {image} is not a candidate for {shot}. Candidates: {', '.join(choices)}", fg="red")
        raise typer.Exit(2)
    shutil.copyfile(settings.assets_dir / "keyframes" / shot / image, picks_dir / f"{shot}.png")
    entry.update(pick=image, picked_by="eye", note=note)
    picks_file.write_text(json.dumps(dict(sorted(picks.items())), indent=2) + "\n")
    typer.secho(f"{shot}: picked {image} by eye", fg="green")


VIDEO_POLL_TIMEOUT_S = 1800  # video jobs take minutes, not seconds


@app.command()
def video(
    shot: str = typer.Argument(..., help="Shot id, e.g. s02"),
    takes: int = typer.Option(1, help="How many draft takes to make (each a fresh roll; no seeds)."),
    enhance: str = typer.Option(None, help="A draft (file name or path) to render at full quality."),
    resolution: str = typer.Option("hd", help="For --enhance: hd (native, $0.15/s) or fhd/qhd/uhd (upsampled)."),
) -> None:
    """M6: animate a shot's picked keyframe. Drafts first ($0.06/s); then --enhance the keeper."""
    from .video import draft_request, enhance_request, find_draft_cache, fit_start_frame

    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    root = settings.assets_dir.parent
    keyframe = root / "results" / "keyframes" / f"{shot}.png"
    work = settings.assets_dir / "video" / shot
    clips_dir = root / "results" / "video"
    manifest_file = clips_dir / "clips.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else {}
    try:
        s = bible.shot(shot)
    except KeyError as exc:
        typer.secho(f"  x {exc}", fg="red")
        raise typer.Exit(2)
    if not keyframe.is_file():
        typer.secho(f"  x no keyframe pick for {shot}: run `fluxpipe keyframes` first", fg="red")
        raise typer.Exit(2)

    client = BFLClient(settings.api_key, Ledger(settings.ledger_path, settings.spend_cap_usd),
                       dry_run=settings.dry_run)
    entry = manifest.setdefault(shot, {"keyframe": keyframe.name, "takes": [], "final": None})
    picks_file = root / "results" / "keyframes" / "picks.json"
    # Which candidate the clip starts from: picks can change later, which makes a clip out of date.
    pick = json.loads(picks_file.read_text()).get(shot, {}).get("pick") if picks_file.is_file() else None

    if enhance:
        name = Path(enhance).name
        cache = work / name.replace(".mp4", ".cache.bin")
        if not cache.is_file():
            typer.secho(f"  x no draft_cache saved for {name} (looked for {cache.name})", fg="red")
            raise typer.Exit(2)
        req = enhance_request(s, cache, resolution=resolution)
        out = work / f"{shot}_final_{resolution}.mp4"
        typer.echo(f"{shot}: enhance {name} at {resolution}: ~${req.est_cost_usd:.2f}")
        try:
            res = client.run_job(endpoint=req.endpoint, payload=req.payload, est_cost_usd=req.est_cost_usd,
                                 stage="video", out_path=out, shot_id=shot, refs=req.refs,
                                 poll_timeout_s=VIDEO_POLL_TIMEOUT_S)
        except (SpendCapExceeded, BFLError) as exc:
            typer.secho(f"  x {exc}", fg="red")
            raise typer.Exit(1)
        if res.dry_run:
            typer.secho(f"  (dry run, ledger row {res.ledger_id})", dim=True)
            return
        clips_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, clips_dir / f"{shot}.mp4")
        take = next((t for t in entry["takes"] if t["file"] == name), {})
        entry["final"] = {"from": name, "from_pick": take.get("from_pick"), "resolution": resolution,
                          "file": out.name, "cost_usd": res.actual_cost_usd}
        manifest_file.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")
        typer.secho(f"  saved {out.relative_to(root)} -> results/video/{shot}.mp4", fg="green")
        typer.echo(f"  cost: estimated ${req.est_cost_usd:.2f}, BFL reported "
                   f"{'$%.2f' % res.actual_cost_usd if res.actual_cost_usd is not None else 'not reported'}")
        return

    start = fit_start_frame(keyframe, work / "start.png")
    first = len(list(work.glob(f"{shot}_draft*.mp4"))) + 1
    reqs = [draft_request(s, start) for _ in range(takes)]
    typer.echo(f"{shot}: {takes} draft take(s), {reqs[0].payload['duration']} s hd: "
               f"~${sum(r.est_cost_usd for r in reqs):.2f}")
    typer.secho(f"  motion: {reqs[0].payload['prompt']}", dim=True)
    for i, req in enumerate(reqs):
        out = work / f"{shot}_draft{first + i}.mp4"
        try:
            res = client.run_job(endpoint=req.endpoint, payload=req.payload, est_cost_usd=req.est_cost_usd,
                                 stage="video", out_path=out, shot_id=shot, refs=req.refs,
                                 poll_timeout_s=VIDEO_POLL_TIMEOUT_S)
        except (SpendCapExceeded, BFLError) as exc:
            typer.secho(f"  x {exc}", fg="red")
            raise typer.Exit(1)
        if res.dry_run:
            typer.secho(f"  (dry run, ledger row {res.ledger_id}) {out.name}  ~${req.est_cost_usd:.2f}", dim=True)
            continue
        # Keep the whole poll response: what a draft returns is only partly documented.
        out.with_suffix(".json").write_text(json.dumps(res.poll, indent=2) + "\n")
        cache_url = find_draft_cache(res.poll or {})
        cache = None
        if cache_url:  # download now: result URLs expire after about 2 hours
            cache = client.fetch(cache_url, out.with_suffix(".cache.bin"))
        else:
            typer.secho(f"  ! no draft_cache in the response (see {out.with_suffix('.json').name});"
                        " this take can't be enhanced", fg="yellow")
        entry["takes"].append({"file": out.name, "from_pick": pick, "cost_usd": res.actual_cost_usd,
                               "draft_cache": cache.name if cache else None})
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")
        cost = f"${res.actual_cost_usd:.2f}" if res.actual_cost_usd is not None else "not reported"
        typer.secho(f"  saved {out.relative_to(root)}  (estimated ${req.est_cost_usd:.2f}, BFL reported {cost})",
                    fg="green")
    if not settings.dry_run:
        typer.echo(f"Watch them, then: fluxpipe score-clip <file>  and  fluxpipe video {shot} --enhance <file>")


DEMO_SHOTS = ["s02", "s04", "s08"]  # the face shots with an enhanced clip (s07's face is too small)


@app.command()
def demo(
    shot: list[str] = typer.Option(DEMO_SHOTS, help="Shots to show, in order (repeat the option)."),
    character: str = typer.Option("ismael", help="Whose locked face the scores compare against."),
    out: Path = typer.Option(None, help="Output mp4 (default results/demo.mp4)."),
) -> None:
    """M7-lite: side-by-side of plain prompts vs the pipeline, with face scores. Free (ffmpeg + Pillow)."""
    from .demo import DemoShot, build, load_scores

    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    root = settings.assets_dir.parent
    results = root / "results"
    char = bible.characters.get(character)
    if char is None or not char.reference_images:
        typer.secho(f"  x no locked face for '{character}'", fg="red")
        raise typer.Exit(2)
    base_scores = load_scores(results / "baseline" / "faces.json")
    pipe_scores = load_scores(results / "keyframes" / "faces.json")
    picks = json.loads((results / "keyframes" / "picks.json").read_text())
    clips_file = results / "video" / "clips.json"
    clips = json.loads(clips_file.read_text()) if clips_file.is_file() else {}

    shots, problems = [], []
    for sid in shot:
        stills = sorted((results / "baseline").glob(f"{sid}_*.png"))
        clip = results / "video" / f"{sid}.mp4"
        if not stills:
            problems.append(f"{sid}: no baseline still in results/baseline/")
        if not clip.is_file():
            problems.append(f"{sid}: no enhanced clip (fluxpipe video {sid} --enhance <draft>)")
        final = clips.get(sid, {}).get("final") or {}
        pick = picks.get(sid, {}).get("pick")
        if clip.is_file() and final.get("from_pick") != pick:
            problems.append(f"{sid}: the clip was animated from {final.get('from_pick')}, but the pick is {pick}")
        if sid not in base_scores or sid not in pipe_scores:
            problems.append(f"{sid}: no face score (fluxpipe score results/baseline|keyframes)")
        if not problems:
            shots.append(DemoShot(sid, stills[0], clip, base_scores[sid], pipe_scores[sid]))
    if problems:
        for p in problems:
            typer.secho(f"  x {p}", fg="red")
        raise typer.Exit(2)

    out = out or results / "demo.mp4"
    spend = Ledger(settings.ledger_path, settings.spend_cap_usd).total_spent()
    typer.echo(f"Building {len(shots)} shots: {', '.join(s.id for s in shots)}...")
    info = build(shots, bible.ref_path(char.reference_images[0]), out, settings.assets_dir / "demo",
                 spend_usd=spend)
    for s in shots:
        b, p = s.base.label(), s.pipe.label()
        typer.echo(f"  {s.id}: plain {b[0]} ({b[1]})  vs  pipeline {p[0]} ({p[1]})")
    if info["headline"]:
        h = info["headline"]
        typer.echo(f"Headline over {', '.join(h['shots'])}: {h['baseline']:.3f} -> {h['pipeline']:.3f}")
    typer.secho(f"  wrote {out.relative_to(root) if out.is_relative_to(root) else out}"
                f"  (parts in {(settings.assets_dir / 'demo').relative_to(root)}/)", fg="green")


@app.command()
def cut(
    use: list[str] = typer.Option([], help="Use this take for a shot, e.g. --use s07=s07_draft3.mp4 (repeat)."),
    title: str = typer.Option("The Green Crystal", help="Title card text."),
    out: Path = typer.Option(None, help="Output mp4 (default results/film.mp4)."),
) -> None:
    """M7: the rough cut, every shot in story order from clips of the current picks. Free (ffmpeg)."""
    from .cut import build as build_cut
    from .cut import choose_clips

    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    root = settings.assets_dir.parent
    results = root / "results"
    picks = json.loads((results / "keyframes" / "picks.json").read_text())
    clips_file = results / "video" / "clips.json"
    clips = json.loads(clips_file.read_text()) if clips_file.is_file() else {}
    overrides = {}
    for u in use:
        if "=" not in u:
            typer.secho(f"  x --use takes SHOT=FILE, got {u!r}", fg="red")
            raise typer.Exit(2)
        sid, name = u.split("=", 1)
        overrides[sid] = name
    shots, problems = choose_clips([(s.id, s.location) for s in bible.shots], picks, clips,
                                   results / "video", settings.assets_dir / "video", overrides)
    if problems:
        for p in problems:
            typer.secho(f"  x {p}", fg="red")
        raise typer.Exit(2)
    for s in shots:
        s.cut_to = bible.shot(s.id).cut_to

    finals = sum(s.kind == "final" for s in shots)
    spend = Ledger(settings.ledger_path, settings.spend_cap_usd).total_spent()
    quality = (f"{finals} finished, {len(shots) - finals} at draft quality" if finals < len(shots)
               else "all finished")
    end_lines = [
        "Every frame made with FLUX",
        "stills: FLUX.2 [pro] from a story bible and locked references",
        "motion: FLUX 3 video, image-to-video from each still",
        f"{len(shots)} shots, {quality}  ·  total API spend ${spend:.2f}",
    ]
    out = out or results / "film.mp4"
    for s in shots:
        trim = f"  (first {s.cut_to:g} s)" if s.cut_to else ""
        typer.echo(f"  {s.id}  {s.kind:<5}  {s.clip.relative_to(root)}{trim}")
    info = build_cut(shots, out, settings.assets_dir / "cut", title=title,
                     subtitle="a rough cut" if finals < len(shots) else "", end_lines=end_lines)
    typer.secho(f"  wrote {out.relative_to(root) if out.is_relative_to(root) else out}"
                f"  ({info['story_s']:.1f} s of story; parts in assets/cut/)", fg="green")


@app.command()
def report() -> None:
    """README evidence: the cost table from the ledger, figures from the results, web-size videos. Free."""
    from .demo import load_scores
    from .report import (cost_markdown, cost_rows, cost_summary, fill_markers, headline_markdown, film_strip, frame_at,
                         refs_figure, stills_figure, web_video_cmd)
    import subprocess

    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    root = settings.assets_dir.parent
    results = root / "results"
    out = results / "readme"
    out.mkdir(parents=True, exist_ok=True)
    work = settings.assets_dir / "report"
    work.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(settings.ledger_path, settings.spend_cap_usd)
    calls = [dict(r) for r in ledger.db.execute(
        "SELECT stage, model, est_cost_usd, actual_cost_usd, json_extract(params, '$.mode') AS mode,"
        " json_extract(params, '$.draft') AS draft FROM generations WHERE status IN ('done', 'failed')")]
    rows = cost_rows(calls)
    costs = cost_markdown(rows)
    (results / "costs.md").write_text(costs)
    typer.echo(f"  results/costs.md: {len(calls)} paid calls")
    readme = root / "README.md"
    qa_md = results / "video" / "qa.md"
    if readme.is_file() and qa_md.is_file():
        from .demo import headline
        base_f = json.loads((results / "baseline" / "faces.json").read_text())
        pipe_f = json.loads((results / "keyframes" / "faces.json").read_text())
        base_s, pipe_s = load_scores(results / "baseline" / "faces.json"), load_scores(results / "keyframes" / "faces.json")
        head = headline(DEMO_SHOTS, base_s, pipe_s)
        qa_totals = json.loads((results / "video" / "qa.json").read_text())["totals"]
        top = headline_markdown(base_f["pairwise"], pipe_f["pairwise"], sum(s.judged for s in base_s.values()),
                                sum(s.judged for s in pipe_s.values()), head, qa_totals, rows)
        readme.write_text(fill_markers(readme.read_text(),
                                       {"headline": top, "costs": cost_summary(rows, calls, len(bible.shots)),
                                        "qa": qa_md.read_text()}))
        typer.echo("  README.md: cost and QA tables refreshed")

    shots = [s.id for s in bible.shots]
    baseline = {s: sorted((results / "baseline").glob(f"{s}_*.png"))[0] for s in shots}
    pipeline = {s: results / "keyframes" / f"{s}.png" for s in shots}
    stills_figure(shots, baseline, pipeline, load_scores(results / "baseline" / "faces.json"),
                  load_scores(results / "keyframes" / "faces.json")).save(out / "stills.jpg", quality=88)

    char = bible.characters["ismael"]
    refs = [("face", char.reference_images[0]), *(("costume sheet", r) for r in char.costume_references["beachwear"])]
    refs += [(f"place: {l.id}", l.reference_images[0]) for l in bible.locations.values() if l.reference_images]
    refs += [(f"prop: {p.id}", p.reference_images[0]) for p in bible.props.values() if p.reference_images]
    refs_figure([(label, bible.ref_path(r)) for label, r in refs]).save(out / "refs.jpg", quality=88)

    demo_mp4, film_mp4 = results / "demo.mp4", results / "film.mp4"
    if demo_mp4.is_file():  # the s04 side-by-side: 4 s intro + s02's 5 s, then halfway into s04
        from PIL import Image
        Image.open(frame_at(demo_mp4, 11.5, work / "hero.png")).resize((1280, 720), Image.LANCZOS).save(out / "hero.jpg", quality=90)
    qa_file = results / "video" / "qa.json"
    if film_mp4.is_file() and qa_file.is_file():
        t, frames = 3.0, []  # after the 3 s title card
        for r in json.loads(qa_file.read_text())["shots"]:
            frames.append((r["shot"], frame_at(film_mp4, t + r["seconds"] / 2, work / f"film_{r['shot']}.png")))
            t += r["seconds"]
        film_strip(frames).save(out / "film.jpg", quality=88)
    for name in ("hero", "film", "stills", "refs"):
        if (out / f"{name}.jpg").is_file():
            typer.echo(f"  results/readme/{name}.jpg ({(out / f'{name}.jpg').stat().st_size / 1e3:.0f} kB)")

    web = settings.assets_dir / "web"
    web.mkdir(parents=True, exist_ok=True)
    for src, width, crf in ((demo_mp4, 1280, 23), (film_mp4, 960, 27)):  # the demo's text must stay sharp
        if src.is_file():
            subprocess.run(web_video_cmd(src, web / src.name, width, crf), check=True)
            mb = (web / src.name).stat().st_size / 1e6
            typer.echo(f"  assets/web/{src.name}: {mb:.1f} MB (drag into the README on github.com to play inline)")


@app.command("score-clip")
def score_clip(
    clip: Path = typer.Argument(..., help="An mp4, e.g. assets/video/s02/s02_draft1.mp4"),
    character: str = typer.Option("ismael", help="Whose locked face to compare against."),
    fps: float = typer.Option(2.0, help="Frames sampled per second."),
) -> None:
    """Video QA: face identity in frames sampled across a clip (first, mean, min, drift). Free."""
    from .qa.face import MAX_PITCH_DEG, MAX_YAW_DEG, insightface_detector, score_images
    from .qa.video import extract_frames, summarize

    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    char = bible.characters.get(character)
    if char is None or not char.reference_images or not clip.is_file():
        typer.secho(f"  x need an existing clip and a locked face for '{character}'", fg="red")
        raise typer.Exit(2)
    frames = extract_frames(clip, clip.parent / f"frames_{clip.stem}", fps=fps)
    typer.echo(f"{clip.name}: {len(frames)} frames at {fps:g}/s. Loading InsightFace...")
    scores, _ = score_images(insightface_detector(), bible.ref_path(char.reference_images[0]), frames)
    for i, sc in enumerate(scores):
        sim = f"{sc.similarity:.3f}" if sc.similarity is not None else "  -  "
        pose = f"pitch {sc.pitch:6.1f}  yaw {sc.yaw:6.1f}" if sc.pitch is not None else " " * 25
        typer.echo(f"  t={i / fps:4.1f}s  {sim}  {pose}  {sc.verdict}")
    summary = summarize(scores)
    if summary["judged"]:
        typer.secho(f"first {summary['first']:.3f}, mean {summary['mean']:.3f}, min {summary['min']:.3f},"
                    f" drift {summary['drift']:.3f} over {summary['judged']} of {summary['frames']} frames",
                    bold=True)
    else:
        typer.echo("No judgeable face in any frame (wide, hands only or turned away): judge this clip by eye.")
    if summary["turned_away"]:
        typer.secho(f"  {summary['turned_away']} frame(s) not judged: head turned more than "
                    f"{MAX_PITCH_DEG} deg up/down or {MAX_YAW_DEG} deg sideways. Check those by eye.", dim=True)
    from .qa.background import measure

    bg = measure(clip)
    colour = {"steady": "green", "slight jitter": "yellow", "jittery": "red", "not tracked": "yellow"}
    typer.secho(f"background: jitter {bg.jitter_px:.2f} px ({bg.verdict}), camera {bg.camera}",
                fg=colour[bg.verdict], bold=True)
    out = clip.with_suffix(".faces.json")
    out.write_text(json.dumps({"summary": summary, "background": bg.as_dict(),
                               "frames": [x.as_dict() for x in scores]}, indent=2) + "\n")
    typer.secho(f"  wrote {out}", dim=True)


@app.command()
def qa(
    character: str = typer.Option("ismael", help="Whose locked face to compare against."),
) -> None:
    """Film QA: face identity and background jitter for every clip in the cut. Free (runs locally)."""
    from .cut import choose_clips
    from .demo import clip_seconds
    from .qa.background import measure
    from .qa.face import insightface_detector, score_images
    from .qa.report import markdown, row, totals
    from .qa.video import extract_frames, summarize

    settings = load_settings()
    bible = load_bible(settings.bible_dir)
    root = settings.assets_dir.parent
    results = root / "results"
    char = bible.characters.get(character)
    picks = json.loads((results / "keyframes" / "picks.json").read_text())
    clips_file = results / "video" / "clips.json"
    clips = json.loads(clips_file.read_text()) if clips_file.is_file() else {}
    shots, problems = choose_clips([(s.id, s.location) for s in bible.shots], picks, clips,
                                   results / "video", settings.assets_dir / "video")
    if problems or char is None or not char.reference_images:
        for p in problems or [f"no locked face for '{character}'"]:
            typer.secho(f"  x {p}", fg="red")
        raise typer.Exit(2)

    typer.echo(f"QA of the {len(shots)} clips the cut uses. Loading InsightFace...")
    detect = insightface_detector()
    ref = bible.ref_path(char.reference_images[0])
    rows = []
    for s in shots:
        shot = bible.shot(s.id)
        work = settings.assets_dir / "qa" / s.id
        face_summary = None
        if shot.face:
            frames = extract_frames(s.clip, work, fps=2.0, seconds=shot.cut_to)
            scores, _ = score_images(detect, ref, frames)
            face_summary = summarize(scores)
        bg = measure(s.clip, seconds=shot.cut_to)
        seconds = min(clip_seconds(s.clip), shot.cut_to or 1e9)
        r = row(s.id, s.kind, seconds, face_summary, shot.face, bg.as_dict())
        rows.append(r)
        typer.echo(f"  {s.id}  {s.kind:<5}  face: {r['face_text']:<58}  background: {bg.jitter_px:.2f} px"
                   f" {bg.verdict}, camera {bg.camera}")
    t = totals(rows)
    (results / "video" / "qa.json").write_text(json.dumps({"totals": t, "shots": rows}, indent=2) + "\n")
    (results / "video" / "qa.md").write_text(markdown(rows))
    typer.secho(f"Face: {t['face_frames_same']} of {t['face_frames']} judged frames the same person"
                f" (worst {t['face_worst']:.2f}). Background: {t['steady']} of {t['shots']} steady,"
                f" worst {t['jitter_max']:.2f} px.", bold=True)
    typer.secho("  wrote results/video/qa.json and results/video/qa.md", dim=True)


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
    colour = {"same": "green", "different": "red", "too small": "yellow", "no face": "yellow",
              "turned away": "yellow"}
    for sc in scores:
        sim = f"{sc.similarity:.3f}" if sc.similarity is not None else "  -  "
        px = f"{sc.height_px}px" if sc.height_px else "-"
        typer.secho(f"  {sc.image:<32} {sim}  face {px:>6}  {sc.verdict}", fg=colour[sc.verdict])
    judged = [sc.similarity for sc in scores if sc.verdict in ("same", "different")]
    if judged:
        typer.echo(f"Mean over {len(judged)} judgeable face(s): {sum(judged) / len(judged):.3f}"
                   f"  (same person >= {SAME_PERSON}; faces under {SMALL_FACE_PX}px or turned away not judged)")
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
