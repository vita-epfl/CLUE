---
name: data-mining
description: Find images or video frames where a body part (arm, hand, leg, foot, head, torso, ...) or the body shape is most uncertain, or most certain, using CLUE's per-vertex uncertainty of 3D human meshes. Use when a user asks to find, mine, rank, search or retrieve images or frames in which a body part or the body shape is uncertain, ambiguous, hard or unreliable (or certain, confident, reliable), for instance people in jackets or loose clothing, over a folder of images or a video. Also exports every person's scores as a table, for curating datasets by uncertainty.
---

# Data mining with CLUE

CLUE predicts the 3D body mesh of every person in an image and how uncertain each of its 18,439 vertices is. This skill
turns a request into two rankings of people, over a folder of images or a video file. A request such as **"find images
where the arm is most uncertain"** gives:

1. **Gap to the rest of the body**: people for whom the part stands out. Its uncertainty is far above that of every other
   body part.
2. **Part uncertainty**: people for whom the part itself is most uncertain, whatever the rest of the body does.

**"Most certain"** requests reverse both rankings: the part is far more certain than every other body part, or the part
itself is the most certain. **Body shape** requests ("whose shape is hard to estimate", "people in jackets") rank people
by how uncertain CLUE is about their body shape: how far it stands out from their overall uncertainty, and how high it
is.

For **data curation** ("keep only the images where...", "a table of every person"), `--export` writes every person's
scores to one CSV instead (section 5).

Everything runs through `mine.py` in this folder, which uses CLUE from `../ITW/inference`.

## Before the first run

- **Installing the skill.** This folder is an [Agent Skill](https://agentskills.io). An agent finds it once the folder
  is linked, under the name `data-mining`, into the folder where that agent loads skills from (see the agent's
  documentation): `ln -s "$PWD/data-mining" <skills folder>/data-mining`.
- **Link it; never copy it.** `mine.py` runs CLUE from `../ITW/inference`, the folder next to this one, so it only works
  inside the repository. A link keeps it there; a copy does not work. If `mine.py` stops because it cannot find CLUE's
  inference code, the folder was copied: tell the user to replace the copy with a link.
- `../ITW/inference` must be set up: its packages installed and its models in `../ITW/inference/assets/` (its README,
  Setup). Missing models are downloaded there on the first run, which needs a Hugging Face login with access to the two
  gated repositories named in that README.
- Use a GPU. On a CPU, CLUE takes tens of seconds per person.

## 1. Turn the request into a part, a side and a direction

**Body parts**

| the user says | `--part` | body groups used |
|---|---|---|
| arm, arms | `arm` | upper arm, forearm, hand (the whole limb) |
| upper arm | `upper_arm` | upper arm |
| forearm, elbow | `forearm` | forearm |
| hand, hands, wrist, fingers | `hand` | hand only |
| shoulder, clavicle, collarbone | `clavicle` | clavicle |
| leg, legs | `leg` | thigh, lower leg, foot (the whole limb) |
| thigh, knee | `thigh` | thigh |
| lower leg, shin, calf | `lower_leg` | lower leg |
| foot, feet, ankle, toes | `foot` | foot only |
| head, face | `head` | neck, head, jaw |
| neck | `neck` | neck |
| torso, chest, back, hips | `torso` | pelvis, spine, both clavicles |

- **"arm" means the whole limb, "hand" only the hand**; likewise "leg" versus "foot".
- **Sides.** Without a side, both sides are pooled into one part: "arm" is both arms together. "left arm" is
  `--side left`, "right hand" is `--side right`. Left and right are the **person's own** sides, not the image's. If the
  user clearly means the image side ("the arm on the left of the picture"), say so and ask which they want.
- **Several parts or vague words** ("limbs", "extremities", "upper body"): run one query per part, or ask the user.
- **A single MHR body group** (pelvis, spine, jaw, the head without neck and jaw, or one exact segment such as the left
  forearm): use `--group` with its name instead of `--part`: `root`, `c_spine`, `c_neck`, `c_head`, `c_jaw`, and per
  side (`l_` / `r_`) `clavicle`, `uparm`, `lowarm`, `hand`, `upleg`, `lowleg`, `foot`.

**Body shape**

| the user says | flag |
|---|---|
| shape, body shape, build, physique, proportions, silhouette, body size | `--shape` |
| jackets, coats, loose or baggy clothing, dresses, robes, costumes, bulky clothes | `--shape` |

- `--shape` takes no `--side` and no part: it covers the whole body, without head and hands, which have shape
  parameters of their own.
- Clothing is not detected. The gap ranking finds people whose body shape CLUE cannot pin down although it is sure of
  the rest: loose clothing is one reason, a torso hidden behind objects or seen foreshortened (from above, from behind,
  bent over) others. The shape-uncertainty ranking is led by small, distant or blurred people, who are uncertain in
  everything. Say so when you report.

**Direction**

| the user says | flag |
|---|---|
| most uncertain, ambiguous, hard, unreliable, poorly estimated, low confidence | none (the default) |
| most certain, confident, reliable, well estimated, easy, clear, high confidence | `--certain` |

"Certain with respect to the rest of the body" is the gap ranking with `--certain`; every `--certain` query gives both
rankings, as an uncertain query does.

## 2. Run the query

```bash
cd path/to/data-mining
python mine.py --input <folder of images | video file> --part <part> [--side left|right] [--certain] \
    --query "<the user's request, in their words>" [--num 5]
python mine.py --input <folder of images | video file> --group <MHR group> [--certain] \
    --query "<the user's request, in their words>" [--num 5]
python mine.py --input <folder of images | video file> --shape [--certain] \
    --query "<the user's request, in their words>" [--num 5]
```

- `--input` takes a **folder of images** (videos inside it are skipped) or **one video file**. For several videos, run
  one query per video.
- `--num`: people per ranking, 5 by default. Use the number the user asks for.
- Video: every frame is scanned; a person appears at most once per ranking within ±2 s (`--window`), so the results
  are not near-identical frames of one moment. "The same person" means the same track, or two boxes of which the
  smaller lies at least 60% inside the other (the tracker can lose a person and pick them up again under a new id, often
  with a differently sized box).
- The first query on an input scans it with CLUE and caches the result in `scans/`. Later queries on the same input, for
  any part, the shape and either direction, only rank and render. `--rescan` forces a new scan; a scan made before
  body-shape mining existed is redone automatically.

## 3. What a query produces

Every query creates a **new folder** `queries/<query>__<input>/`:

```
queries/<query>__<input>/
  query.json                    the request, mode, direction, part, side, body groups, input, people scanned, definitions
  <ranking 1>/
    001_<image>_p<person>.jpg   (video: 001_<video>_f<frame>_p<track>.jpg)
    ...
    results.csv                 rank, file, score, the query's measures, source, frame, time_s, person, box
  <ranking 2>/                  same layout
```

| query | ranking 1 | ranking 2 | measures in `results.csv` |
|---|---|---|---|
| part, uncertain | `gap_to_rest_of_body/` | `part_uncertainty/` | `level`, `gap` (percentile points) |
| part, `--certain` | `gap_to_rest_of_body/` | `part_certainty/` | `level`, `gap` (percentile points) |
| `--shape` | `gap_to_overall_uncertainty/` | `shape_uncertainty/` | `shape_level`, `overall_level`, `gap` (percentile points), `shape_sigma_cm`, `overall_sigma_cm` (cm) |
| `--shape --certain` | `gap_to_overall_uncertainty/` | `shape_certainty/` | as above |

`score` is the measure the folder is ranked on. Each image is a crop around one person: **Input | SAM 3D Body | CLUE**,
with only that person's meshes drawn. SAM 3D Body's mesh is grey, for comparison. CLUE's mesh has every vertex coloured
by its total uncertainty percentile (blue = as certain as that vertex gets, red = its most uncertain), in shape queries
too.

## 4. Report back

Give the user:
- the query folder,
- how many people were scanned,
- the top 3 of each ranking, as the image or frame (time) with its score,
- if your interface can show images, the first one or two of each ranking.

When the two rankings differ, say why:
- **Parts.** The gap ranking finds people whose part stands out from the rest of their body. The part ranking also
  returns people who are uncertain (or certain) all over.
- **Shape.** The gap ranking finds people whose body shape is more uncertain than the rest of their reconstruction:
  the candidates for loose clothing. The shape ranking also returns people who are uncertain all over, mostly small or
  distant ones.

If the best gap is zero or negative, say so: in this input the part never rises above (or, with `--certain`, never stays
below) every other body group, and the gap ranking only lists the closest cases.

## 5. Data curation: every person's scores

When the user wants to filter or curate data rather than see the top few, or asks for a table or list of everyone:

```bash
python mine.py --input <folder of images | video file> --export <table>.csv
```

One row per detected person, from the same cached scan as the queries (no model runs if the input was scanned
before):

| columns | meaning |
|---|---|
| `source`, `frame`, `time_s`, `person`, `box_x0` ... `box_y1` | the image or video, the frame and time (video only), the track id (images: detection index), the box in the scanned picture |
| `level_<group>` | the level of each of the 19 body groups (percentile points, 0-100) |
| `sigma_cm_<group>` | each group's root-mean-square sigma (cm) |
| `shape_level`, `overall_level`, `shape_sigma_cm`, `overall_sigma_cm` | the body-shape and overall measures of `--shape` |

The rankings' scores follow from the table (see "How uncertainty is measured"): a part's level is the mean of its
groups' `level_` columns, its gap that mean minus the largest `level_` of the other groups, and the shape gap
`shape_level - overall_level`. For example, the people whose left leg stands out by more than 20 points:

```python
import pandas as pd
t = pd.read_csv("<table>.csv")
leg = ["level_l_upleg", "level_l_lowleg", "level_l_foot"]
rest = [c for c in t if c.startswith("level_") and c not in leg]
t["gap"] = t[leg].mean(axis=1) - t[rest].max(axis=1)
t[t.gap > 20].sort_values("gap", ascending=False)
```

Use the thresholds the user gives. Without one, show the distribution of the score (quantiles), suggest a cut, and
say how many people and images it keeps. For a video, `frame` and `time_s` locate each person.

## How uncertainty is measured

- **Per-vertex percentile.** `u_v` is the fraction of a reference population with a lower sigma at vertex v, where sigma
  is the standard deviation of the vertex position (cm; root rotation and translation excluded). The reference is
  6,118 person-frames of in-the-wild video (`../ITW/inference/percentiles/clue.npz`), with left and right twins pooled.
- **Group level.** The mean of `u_v` over the vertices of one of the 19 body groups (`part_groups.npz`): root, spine,
  neck, head, jaw, and per side clavicle, upper arm, forearm, hand, thigh, lower leg, foot.
- **Part level** (ranking 2). The mean of the part's group levels: an arm is the mean of upper arm, forearm and hand, and
  both arms the mean of those six groups. Uncertain queries rank it highest first, `--certain` lowest first.
- **Gap** (ranking 1). Uncertain: the part level minus the highest level among all body groups outside the part.
  `--certain`: the lowest level among all body groups outside the part minus the part level. Highest first in both.
- **Body-shape sigma.** CLUE predicts a variance for each of MHR's shape parameters. The 20 body-shape parameters alone
  are propagated to the vertices, through the same closed form as the total. The body is the 14 groups other than neck,
  head, jaw and the two hands, which have shape parameters of their own. Per person, `shape_sigma_cm` is the root of
  the mean over those groups of each group's mean squared shape sigma, and `overall_sigma_cm` the same for the total
  sigma (shape and pose). The groups weigh equally, as in a part level: by vertex count, the two feet alone would weigh
  as much as the torso.
- **Shape and overall level.** The percentiles of `shape_sigma_cm` and `overall_sigma_cm` among 7,273 person-frames of
  the same in-the-wild video (`shape_percentiles.npz`). Body shape is a small part of the total (about 0.2 cm of 4.6 cm
  in the median person, mostly at the pelvis and spine), and its sigma mostly follows the overall one.
- **Shape gap** (ranking 1). Shape level minus overall level; `--certain`: overall level minus shape level. Highest first
  in both. **Shape level** (ranking 2): highest first, or lowest first with `--certain`.

## Notes

- **Images:** every detected person is a candidate, and duplicate detections of one person are removed.
- **Video:** people are tracked, and their boxes are smoothed over 5 frames.
- **A hidden limb can still rank high.** A limb hidden behind the body can be very uncertain while looking unremarkable
  in the crop. That is a correct answer, not an error.
- **Colours are relative.** They compare a vertex with its own values in the reference video; read them as "unusually
  uncertain for this vertex", not as centimetres.

## Examples

The folders in `queries/` were made from `../ITW/inference/examples/video.mp4`, a 2-minute compilation of in-the-wild
clips from Pexels:

```bash
python mine.py --input ../ITW/inference/examples/video.mp4 --part arm --side left --certain \
    --query "find frames where the left arm is most certain with respect to the rest of the body"
python mine.py --input ../ITW/inference/examples/video.mp4 --part arm --side left \
    --query "find frames where the left arm is most uncertain with respect to the rest of the body"
python mine.py --input ../ITW/inference/examples/video.mp4 --shape \
    --query "find frames where the shape is the most uncertain"
python mine.py --input ../ITW/inference/examples/video.mp4 --part leg --side right --certain \
    --query "find frames where the right leg is most certain"
python mine.py --input ../ITW/inference/examples/video.mp4 --part leg --side right \
    --query "find frames where the right leg is most uncertain"
```

The first command scans the video; the other four reuse that scan.
