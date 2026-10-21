"""SAM 3D Body and CLUE on images and videos.

    python run.py --input examples/six_friends.jpg examples/video.mp4 --output results
    python run.py --input my_photos/ --output results --model sam3dbody_clue clue

Every image is written to <output>/<name>.jpg and every video to <output>/<name>.mp4, with the panels
Input | SAM 3D Body | SAM 3D Body (CLUE) | CLUE (or Input | the chosen models). The CLUE models' meshes are coloured by
the uncertainty of each vertex.
The models are read from the assets folder; missing ones are downloaded there from Hugging Face (README, Setup).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEOS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def collect(paths: list[Path]) -> list[Path]:
    """Files as given, and the images and videos directly inside given folders."""
    files = []
    for path in paths:
        if path.is_dir():
            files += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGES | VIDEOS)
        elif path.suffix.lower() in IMAGES | VIDEOS:
            files.append(path)
        else:
            raise SystemExit(f"not an image, video or folder: {path}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="images, videos and/or folders of them")
    parser.add_argument("--output", type=Path, required=True, help="folder for the results")
    parser.add_argument("--model", nargs="+", choices=("sam3dbody", "sam3dbody_clue", "clue"), default=None,
                        help="one or more of sam3dbody, sam3dbody_clue, clue (default: all three); only those run")
    parser.add_argument("--device", default=None, help="default: cuda when available, else cpu (slow)")
    parser.add_argument("--max-side-image", type=int, default=1600, help="images are downscaled to this long side")
    parser.add_argument("--max-side-video", type=int, default=1280, help="videos are downscaled to this long side")
    parser.add_argument("--max-seconds", type=float, default=None, help="only the first seconds of each video")
    args = parser.parse_args()

    files = collect(args.input)
    if not files:
        raise SystemExit("no images or videos found")
    from clue_inference.model import MODELS
    from clue_inference.pipeline import Pipeline

    pipeline = Pipeline(tuple(args.model) if args.model else MODELS, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    for path in files:
        if path.suffix.lower() in IMAGES:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"{path}: could not read, skipped", flush=True)
                continue
            panels, report = pipeline.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), max_side=args.max_side_image)
            target = args.output / f"{path.stem}.jpg"
            cv2.imwrite(str(target), cv2.cvtColor(panels, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            target = args.output / f"{path.stem}.mp4"
            report = pipeline.video(path, target, max_seconds=args.max_seconds, max_side=args.max_side_video)
        print(f"{path}: {report.summary()} -> {target}", flush=True)


if __name__ == "__main__":
    main()
