"""Data mining with CLUE: find the people in images or video frames whose <body part> or body shape is most uncertain
(or most certain).

    python mine.py --input <image folder | video file> --part arm [--side left] [--certain] --query "<the request>" [--num 5]
    python mine.py --input <image folder | video file> --shape [--certain] --query "<the request>" [--num 5]
    python mine.py --input <image folder | video file> --export scores.csv

--export writes every scanned person's scores to one CSV, for curating data by any threshold, instead of a query.

Every query gets a new folder queries/<query>__<input>/ with query.json and two rankings of the top --num people:
    --part / --group:
        gap_to_rest_of_body/   the part stands out: its uncertainty level minus that of the most uncertain other body group
                               (--certain: the level of the most certain other body group minus the part's)
        part_uncertainty/      the part itself is uncertain: its uncertainty level (--certain: part_certainty/, lowest first)
    --shape:
        gap_to_overall_uncertainty/  the body shape stands out: the percentile of the body's sigma from body shape alone
                                     minus that of its overall sigma (--certain: the other way round)
        shape_uncertainty/           the body shape itself is uncertain: that shape percentile (--certain:
                                     shape_certainty/, lowest first)
each holding one Input | SAM 3D Body | CLUE crop per person and results.csv. SKILL.md explains how an agent turns a request
into a command.

Every vertex's sigma becomes its percentile among the values that vertex takes over a reference set of in-the-wild video
(../ITW/inference/percentiles/clue.npz); a body group's level is the mean percentile of its vertices, and a part's level the
mean of its groups' levels. Body shape is CLUE's variance of MHR's 20 body-shape parameters alone, propagated to the
vertices; head and hands are left out, and its percentile comes from a per-person reference (shape_percentiles.npz). An
input is scanned once: scans/<input>.npz keeps, for every person and each of the 19 body groups, the level and the mean
variance (total and body shape alone), so further queries on it only rank and render.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent

GROUPS = ("root", "c_spine", "c_neck", "c_head", "c_jaw", "l_clavicle", "l_uparm", "l_lowarm", "l_hand", "r_clavicle",
          "r_uparm", "r_lowarm", "r_hand", "l_upleg", "l_lowleg", "l_foot", "r_upleg", "r_lowleg", "r_foot")
SIDED = {                                       # part -> its groups on one side (prefixed l_ / r_)
    "arm": ("uparm", "lowarm", "hand"), "upper_arm": ("uparm",), "forearm": ("lowarm",), "hand": ("hand",),
    "clavicle": ("clavicle",), "leg": ("upleg", "lowleg", "foot"), "thigh": ("upleg",), "lower_leg": ("lowleg",),
    "foot": ("foot",),
}
CENTRAL = {"head": ("c_neck", "c_head", "c_jaw"), "neck": ("c_neck",), "torso": ("root", "c_spine", "l_clavicle", "r_clavicle")}
PARTS = tuple(SIDED) + tuple(CENTRAL)
GROUP_WORDS = {"root": "pelvis", "c_spine": "spine", "c_neck": "neck", "c_head": "head", "c_jaw": "jaw"}   # --group names
GROUP_WORDS.update({f"{s}_{g}": f"{side} {word}" for s, side in (("l", "left"), ("r", "right"))
                    for g, word in (("clavicle", "clavicle"), ("uparm", "upper arm"), ("lowarm", "forearm"), ("hand", "hand"),
                                    ("upleg", "thigh"), ("lowleg", "lower leg"), ("foot", "foot"))})
IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEOS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
NOT_BODY = ("c_neck", "c_head", "c_jaw", "l_hand", "r_hand")       # left out of the body-shape measures: own shape parameters
SHAPE_REFERENCE = HERE / "shape_percentiles.npz"
INFERENCE = HERE.parent / "ITW/inference"       # CLUE's inference code; HERE follows a link back into the repository
CROP = 512                                      # px, side of each square crop


def part_groups(part: str, side: str) -> list[str]:
    """The body groups of a part: one side ("left", "right") or both pooled ("both") for arms, hands, legs, feet...;
    head, neck and torso have no side."""
    if part in CENTRAL:
        if side != "both":
            raise SystemExit(f"{part} has no side")
        return list(CENTRAL[part])
    sides = ("l", "r") if side == "both" else (side[0],)
    return [f"{s}_{g}" for s in sides for g in SIDED[part]]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60] or "query"


# ----------------------------------------------------------------------------------------------------------- scanning
def input_sources(path: Path) -> tuple[str, list[Path]]:
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGES)
        skipped = [p.name for p in path.iterdir() if p.suffix.lower() in VIDEOS]
        if skipped:
            print(f"videos in the folder are not scanned (give each video as --input): {skipped}", flush=True)
        if not images:
            raise SystemExit(f"no images in {path}")
        return "images", images
    if path.suffix.lower() in VIDEOS:
        return "video", [path]
    raise SystemExit(f"--input must be a folder of images or a video file: {path}")


def scan_path(path: Path, kind: str, sources: list[Path], scans: Path) -> Path:
    key = hashlib.sha1("|".join(f"{p.resolve()}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in sources).encode()).hexdigest()[:10]
    return scans / f"{slug(path.stem if kind == 'video' else path.name)}__{key}.npz"


def group_stats(table, vertex_groups, persons) -> dict[str, np.ndarray]:
    """Per body group, (P, 19) each: ``levels``, the mean per-vertex percentile in [0, 1]; ``var`` and ``shape_var``, the
    mean per-vertex variance (sigma^2, cm^2) in total and from body shape alone."""
    import torch

    sigma = np.stack([p.clue_sigma_cm for p in persons])                             # (P, V)
    u = table(sigma).float()
    var = torch.as_tensor(sigma, device=u.device) ** 2
    shape_var = torch.as_tensor(np.stack([p.clue_shape_sigma_cm for p in persons]), device=u.device) ** 2
    return {key: np.stack([x[:, g].mean(1).cpu().numpy() for g in vertex_groups], 1).astype(np.float32)
            for key, x in (("levels", u), ("var", var), ("shape_var", shape_var))}


def pipeline(device: str | None):
    from clue_inference.pipeline import Pipeline

    return Pipeline(("sam3dbody", "clue"), device)                  # scans run CLUE only; SAM 3D Body for the crops


def scan(pipe, kind: str, sources: list[Path], args) -> dict:
    import torch

    from clue_inference.framing import content_rect
    from clue_inference.model import cliff_intrinsics
    from clue_inference.pipeline import VIDEO_BOXES, _fit, _frames, drop_duplicates
    from clue_inference.tracking import people_per_frame

    z = np.load(HERE / "part_groups.npz")
    vertex_groups = [torch.as_tensor(z[f"verts_{g}"].astype(np.int64), device=pipe.device) for g in GROUPS]
    table = pipe.tables["clue"]
    stats = ("levels", "var", "shape_var")
    rows = {k: [] for k in stats + ("source", "frame", "time", "track", "box", "cam", "rect")}

    def add(persons, source, frame, t, cam, rect):
        if not persons:
            return
        for key, value in group_stats(table, vertex_groups, persons).items():
            rows[key].append(value)
        for p in persons:
            rows["source"].append(source); rows["frame"].append(frame); rows["time"].append(t); rows["track"].append(p.track_id)
            rows["box"].append(p.box); rows["cam"].append(cam); rows["rect"].append(rect)

    fps = 0.0
    if kind == "images":
        for start in range(0, len(sources), 8):
            frames = [_fit(cv2.imread(str(p), cv2.IMREAD_COLOR), args.max_side_image) for p in sources[start:start + 8]]
            rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            rects = [content_rect(f) for f in frames]
            cams = [cliff_intrinsics(f.shape[1], f.shape[0], r) for f, r in zip(frames, rects)]
            boxes = pipe.detector(rgb)
            preds = pipe.models.predict(rgb, [[(i, b[:4]) for i, b in enumerate(bx)] for bx in boxes], cams, models=("clue",),
                                        shape=True)
            for k, persons in enumerate(preds):
                add(drop_duplicates(persons, cams[k]), start + k, -1, float("nan"), cams[k], rects[k])
            print(f"scanned {min(start + 8, len(sources))}/{len(sources)} images", flush=True)
    else:
        cap = cv2.VideoCapture(str(sources[0]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        detections, rects, batch = [], [], []
        for frame in _frames(sources[0], None, args.max_side_video):
            rects.append(content_rect(frame)); batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(batch) == 8:
                detections += pipe.detector(batch, **VIDEO_BOXES); batch.clear()
        if batch:
            detections += pipe.detector(batch, **VIDEO_BOXES)
        people = people_per_frame(detections, min_length=min(30, max(len(rects) // 2, 1)), smooth_window=5)
        frames, first = [], 0
        for frame in _frames(sources[0], len(rects), args.max_side_video):
            frames.append(frame)
            if len(frames) == 64 or first + len(frames) == len(rects):
                cams = [cliff_intrinsics(f.shape[1], f.shape[0], rects[first + k]) for k, f in enumerate(frames)]
                preds = pipe.models.predict([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], people[first:first + len(frames)], cams,
                                            models=("clue",), shape=True)
                for k, persons in enumerate(preds):
                    add(persons, 0, first + k, (first + k) / fps, cams[k], rects[first + k])
                first += len(frames); frames.clear()
                print(f"scanned {first}/{len(rects)} frames", flush=True)
    if not rows["source"]:
        raise SystemExit("no people found")
    data = {k: np.concatenate(v) if k in stats else np.asarray(v) for k, v in rows.items()}
    data["meta"] = np.asarray(json.dumps({"input": str(args.input.absolute()), "kind": kind, "sources": [str(p.absolute()) for p in sources],
                                          "fps": fps, "max_side": args.max_side_image if kind == "images" else args.max_side_video,
                                          "groups": GROUPS, "model": "CLUE", "created": datetime.now().isoformat(timespec="seconds")}))
    return data


# ------------------------------------------------------------------------------------------------------------ ranking
SAME_PERSON_OVERLAP = 0.6                       # video: this share of the smaller box inside the other = one person


def box_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the smaller box that lies inside the other: stays high when the tracker's box for the same person
    changes size (a new track for the same person), unlike IoU."""
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return float(max(w, 0.0) * max(h, 0.0) / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def rank(score: np.ndarray, data: dict, num: int, window: float) -> list[int]:
    """Top ``num`` people by ``score``. In a video, a person appears at most once within +-``window`` seconds: the same
    track, or a box of which >= 60% lies inside the earlier pick's (or the other way round): the tracker can lose a person
    and pick them up again under a new id, often with a differently sized box."""
    def same_moment(i: int, j: int) -> bool:
        return (abs(data["time"][j] - data["time"][i]) <= window
                and (data["track"][j] == data["track"][i] or box_overlap(data["box"][i], data["box"][j]) >= SAME_PERSON_OVERLAP))

    picked = []
    for i in np.argsort(-score, kind="stable"):
        if data["frame"][i] >= 0 and any(same_moment(i, j) for j in picked):
            continue
        picked.append(int(i))
        if len(picked) == num:
            break
    return picked


# ---------------------------------------------------------------------------------------------------------- rendering
def square_crop(image: np.ndarray, box: np.ndarray, margin: float = 1.3) -> np.ndarray:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = int(round(margin * max(box[2] - box[0], box[3] - box[1]) / 2))
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    x0, y0 = int(round(cx)), int(round(cy))               # the crop's corner in the padded picture = its centre in the original
    return cv2.resize(padded[y0:y0 + 2 * half, x0:x0 + 2 * half], (CROP, CROP), interpolation=cv2.INTER_AREA)


def load_frames(data: dict, meta: dict, wanted: set) -> dict:
    """(source, frame) -> BGR picture, as scanned (same downscaling)."""
    from clue_inference.pipeline import _fit, _frames

    out = {}
    if meta["kind"] == "images":
        for source, frame in wanted:
            out[(source, frame)] = _fit(cv2.imread(meta["sources"][source], cv2.IMREAD_COLOR), meta["max_side"])
    else:
        last = max(frame for _, frame in wanted)
        for index, picture in enumerate(_frames(meta["sources"][0], last + 1, meta["max_side"])):
            if (0, index) in wanted:
                out[(0, index)] = picture
    return out


def where(data: dict, meta: dict, i: int) -> list[str]:
    """Person i's source file, frame, time (s), track id (images: detection index) and box, as CSV fields."""
    frame = int(data["frame"][i])
    return ([Path(meta["sources"][int(data["source"][i])]).name, str(frame) if frame >= 0 else "",
             f"{data['time'][i]:.2f}" if frame >= 0 else "", str(int(data["track"][i]))] + [f"{v:.1f}" for v in data["box"][i]])


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def render(pipe, data: dict, meta: dict, picked: dict, folder: Path, columns: dict, rankings: dict) -> None:
    """One Input | SAM 3D Body | CLUE crop per picked person (only that person's meshes: SAM 3D Body's in grey, CLUE's
    coloured by percentile) and results.csv: ``score`` is the column ``rankings[folder]`` of ``columns``."""
    from clue_inference.render import Layout

    pictures = load_frames(data, meta, {(int(data["source"][i]), int(data["frame"][i])) for ids in picked.values() for i in ids})
    layout = Layout(CROP, CROP, ("Input", "SAM 3D Body", "CLUE"))
    for name, ids in picked.items():
        out = folder / name
        out.mkdir(parents=True, exist_ok=True)
        rows = [["rank", "file", "score", *columns, "source", "frame", "time_s", "person", "box_x0", "box_y0", "box_x1", "box_y1"]]
        for r, i in enumerate(ids, 1):
            source, frame = int(data["source"][i]), int(data["frame"][i])
            picture = pictures[(source, frame)]
            person = pipe.models.predict([cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)], [[(int(data["track"][i]), data["box"][i])]],
                                         [data["cam"][i]])[0][0]
            u = pipe.tables["clue"](person.clue_sigma_cm)
            grey = pipe.raster(picture, data["cam"][i], [(person.sam3d, None)], data["rect"][i])
            overlay = pipe.raster(picture, data["cam"][i], [(person.clue, u)], data["rect"][i])
            stem = Path(meta["sources"][source]).stem + (f"_f{frame:06d}" if frame >= 0 else "")
            file = f"{r:03d}_{stem}_p{int(data['track'][i])}.jpg"
            cv2.imwrite(str(out / file), layout(*(square_crop(panel, data["box"][i]) for panel in (picture, grey, overlay))),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows.append([str(r), file, f"{columns[rankings[name]][i]:.2f}", *(f"{v[i]:.2f}" for v in columns.values()),
                         *where(data, meta, i)])
        write_csv(out / "results.csv", rows)


# --------------------------------------------------------------------------------------------------------------- main
def part_scores(data: dict, groups: list[str], certain: bool) -> tuple[dict, dict, dict]:
    """--part / --group: columns (percentile points), ranking folder -> column, and the definitions for query.json."""
    chosen = [GROUPS.index(g) for g in groups]
    others = [k for k in range(len(GROUPS)) if k not in chosen]
    level = data["levels"][:, chosen].mean(1)
    if certain:
        gap = data["levels"][:, others].min(1) - level
        rankings = {"gap_to_rest_of_body": "gap", "part_certainty": "level"}
        definitions = {
            "gap_to_rest_of_body": "level of the most certain other body group minus the level of the part (percentile "
                                   "points, highest first)",
            "part_certainty": "level of the part (percentile points, lowest first)"}
    else:
        gap = level - data["levels"][:, others].max(1)
        rankings = {"gap_to_rest_of_body": "gap", "part_uncertainty": "level"}
        definitions = {
            "gap_to_rest_of_body": "level of the part minus the level of the most uncertain other body group (percentile "
                                   "points, highest first)",
            "part_uncertainty": "level of the part (percentile points, highest first)"}
    definitions["level"] = "mean over the part's body groups of each group's mean per-vertex percentile (0-100)"
    definitions["percentiles"] = ("CLUE's per-vertex sigma ranked among that vertex's values over 6,118 person-frames of "
                                  "in-the-wild video")
    return {"level": 100 * level, "gap": 100 * gap}, rankings, definitions


def shape_levels(data: dict) -> dict:
    """Per person, the body's sigma from body shape alone and its overall sigma (cm; the root of the mean over the 14 body
    groups, head, neck and hands left out, of each group's mean squared sigma; the groups weigh equally, as in a part's
    level) and their percentiles in a fixed reference (shape_percentiles.npz), in [0, 1]."""
    body = [k for k, g in enumerate(GROUPS) if g not in NOT_BODY]
    shape_cm = np.sqrt(data["shape_var"][:, body].mean(1))
    overall_cm = np.sqrt(data["var"][:, body].mean(1))
    z = np.load(SHAPE_REFERENCE)
    return {"shape_cm": shape_cm, "overall_cm": overall_cm, "description": str(np.asarray(z["description"])),
            "shape_level": np.interp(shape_cm, z["shape_sigma_at_levels"], z["levels"]),
            "overall_level": np.interp(overall_cm, z["body_sigma_at_levels"], z["levels"])}


def shape_scores(data: dict, certain: bool) -> tuple[dict, dict, dict]:
    """--shape: columns (percentile points, cm), ranking folder -> column, and the definitions for query.json. Body shape's
    sigma alone mostly follows the overall one (small, distant people are uncertain in everything), so the gap compares
    the two percentiles."""
    body = shape_levels(data)
    order = "lowest first" if certain else "highest first"
    level = "shape_certainty" if certain else "shape_uncertainty"
    rankings = {"gap_to_overall_uncertainty": "gap", level: "shape_level"}
    definitions = {
        "gap_to_overall_uncertainty": ("overall level minus shape level" if certain else "shape level minus overall level")
                                      + " (percentile points, highest first)",
        level: f"shape level (percentile points, {order})",
        "shape_level": "percentile of the body's sigma from body shape alone in the reference (0-100)",
        "overall_level": "percentile of the body's overall sigma (shape and pose) in the reference (0-100)",
        "body_sigma": f"root of the mean over the 14 body groups other than {', '.join(NOT_BODY)} of each group's mean "
                      f"squared sigma (cm); head and hands have shape parameters of their own",
        "body_shape": "CLUE's variance of MHR's 20 body-shape parameters alone (identity 0-19), propagated to the vertices",
        "reference": body["description"]}
    shape_level, overall_level = body["shape_level"], body["overall_level"]
    columns = {"shape_level": 100 * shape_level, "overall_level": 100 * overall_level,
               "gap": 100 * (overall_level - shape_level if certain else shape_level - overall_level),
               "shape_sigma_cm": body["shape_cm"], "overall_sigma_cm": body["overall_cm"]}
    return columns, rankings, definitions


def export(data: dict, meta: dict, path: Path) -> None:
    """--export: one row per scanned person: where they are, the level (percentile points) and RMS sigma (cm) of each of
    the 19 body groups, and the body-shape and overall levels and sigmas of --shape."""
    body = shape_levels(data)
    sigma = np.sqrt(data["var"])
    rows = [["source", "frame", "time_s", "person", "box_x0", "box_y0", "box_x1", "box_y1"]
            + [f"level_{g}" for g in GROUPS] + [f"sigma_cm_{g}" for g in GROUPS]
            + ["shape_level", "overall_level", "shape_sigma_cm", "overall_sigma_cm"]]
    for i in range(len(data["levels"])):
        rows.append(where(data, meta, i) + [f"{100 * v:.2f}" for v in data["levels"][i]] + [f"{v:.3f}" for v in sigma[i]]
                    + [f"{100 * body['shape_level'][i]:.2f}", f"{100 * body['overall_level'][i]:.2f}",
                       f"{body['shape_cm'][i]:.3f}", f"{body['overall_cm'][i]:.3f}"])
    write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="a folder of images or a video file")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--part", choices=PARTS, help="body part to mine (SKILL.md: words -> part)")
    what.add_argument("--group", choices=GROUPS, help="one of MHR's 19 body groups on its own (instead of --part)")
    what.add_argument("--shape", action="store_true", help="the body shape (instead of a part)")
    what.add_argument("--export", type=Path, metavar="CSV", help="write every scanned person's scores to this CSV (instead of a query)")
    parser.add_argument("--side", choices=("both", "left", "right"), default="both",
                        help="the person's own left or right; both (default) pools the two sides into one part")
    parser.add_argument("--certain", action="store_true", help="the most certain people instead of the most uncertain")
    parser.add_argument("--query", default=None, help="the request in the user's words (kept in query.json and the folder name)")
    parser.add_argument("--num", type=int, default=5, help="people per ranking (default 5)")
    parser.add_argument("--window", type=float, default=2.0, help="video: a person at most once per ranking within +-this many seconds")
    parser.add_argument("--out", type=Path, default=HERE / "queries", help="where the query folders go")
    parser.add_argument("--scans", type=Path, default=HERE / "scans", help="scan cache")
    parser.add_argument("--rescan", action="store_true", help="scan the input again even when a scan exists")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-side-image", type=int, default=1600)
    parser.add_argument("--max-side-video", type=int, default=1280)
    args = parser.parse_args()

    if args.side != "both" and not args.part:
        raise SystemExit("--side goes with --part; a --group already has its side (l_ / r_), and --shape and --export have none")
    if not (INFERENCE / "clue_inference").is_dir():
        raise SystemExit(f"mine.py runs CLUE from the repository's ITW/inference folder, next to this folder, and did not find "
                         f"it at {INFERENCE}: this folder was copied out of the repository. It only works inside the "
                         f"repository: link it into your agent's skills folder, never copy it.")
    sys.path.insert(0, str(INFERENCE))
    kind, sources = input_sources(args.input)
    cache = scan_path(args.input, kind, sources, args.scans)
    start = time.perf_counter()

    pipe = None
    data = dict(np.load(cache, allow_pickle=False)) if cache.exists() and not args.rescan else None
    if data is not None and "shape_var" not in data:
        print(f"the scan {cache} predates body-shape mining: scanning again", flush=True)
        data = None
    if data is None:
        pipe = pipeline(args.device)
        data = scan(pipe, kind, sources, args)
        args.scans.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **data)
        print(f"scan saved: {cache}", flush=True)
    else:
        print(f"using the scan {cache}", flush=True)
    meta = json.loads(str(data["meta"]))
    if args.export:
        export(data, meta, args.export)
        print(f"{len(data['levels'])} people scanned; every person's scores -> {args.export}")
        return

    groups = [] if args.shape else [args.group] if args.group else part_groups(args.part, args.side)
    what = ("body shape" if args.shape else GROUP_WORDS[args.group] if args.group else
            f"{args.side + ' ' if args.side != 'both' else ''}{args.part.replace('_', ' ')}")
    query = args.query or (f"{what}, most certain" if args.certain else what)
    if args.shape:
        columns, rankings, definitions = shape_scores(data, args.certain)
    else:
        columns, rankings, definitions = part_scores(data, groups, args.certain)
    picked = {}
    for name, column in rankings.items():
        lowest_first = args.certain and column != "gap"
        picked[name] = rank(-columns[column] if lowest_first else columns[column], data, args.num, args.window)

    folder = args.out / f"{slug(query)}__{slug(Path(meta['input']).stem if kind == 'video' else Path(meta['input']).name)}"
    base, n = folder, 2
    while folder.exists():
        folder, n = Path(f"{base}_{n}"), n + 1
    render(pipe or pipeline(args.device), data, meta, picked, folder, columns, rankings)
    (folder / "query.json").write_text(json.dumps({
        "query": query, "mode": "shape" if args.shape else "group" if args.group else "part",
        "direction": "certain" if args.certain else "uncertain", "part": args.part,
        "side": args.side if args.part else None, "groups": groups, "input": str(args.input), "kind": kind,
        "scan": str(cache.relative_to(HERE) if cache.is_relative_to(HERE) else cache), "people_scanned": int(len(data["levels"])), "num": args.num,
        "video_window_s": args.window if kind == "video" else None, "definitions": definitions,
        "created": datetime.now().isoformat(timespec="seconds")}, indent=2) + "\n")
    print(f"query '{query}': {len(data['levels'])} people scanned; top {args.num} per ranking -> {folder} "
          f"({time.perf_counter() - start:.0f} s)")
    for name, ids in picked.items():
        top = ", ".join(f"{Path(meta['sources'][int(data['source'][i])]).name}"
                        f"{'@' + format(data['time'][i], '.1f') + 's' if data['frame'][i] >= 0 else ''} ({columns[rankings[name]][i]:.1f})"
                        for i in ids[:3])
        print(f"  {name}: {top}")


if __name__ == "__main__":
    main()
