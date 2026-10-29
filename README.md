# BlindTrace

Blinded manual tracing of particles in microscopy images on a tablet or any pen or touch screen, plus the analysis that turns several annotators' tracings into consensus particle sizes with agreement statistics.

BlindTrace was written to size extracellular vesicles by manual tracing in liquid-phase and negative-stain transmission electron microscopy, where automatic segmentation is unreliable. Several annotators trace every image without knowing which sample it shows, which keeps expectations about the samples out of the sizes, and the analysis combines their tracings into consensus particles.

## What it does

- **Blinding.** Images are served in a random order per annotator, under opaque random names, rotated by a random multiple of 90° and optionally mirrored. Contrast is normalised per image, and data bars and scale bars can be cropped. All images can be resampled to one pixel size. The unblinding key stays on your computer and never goes on the server.
- **Tracing.** The annotator draws the outline of each particle with a pen (or a finger), and the line is used exactly as drawn.
  - Diameters are shown live.
  - Outlines can be marked "unsure" or deleted.
  - Particles cut by the image edge can be traced past the border.
  - Fingers pan and zoom, and brightness and contrast are adjustable.
  - Each image is approved, or excluded with a reason (out of focus, empty, artefact, other).
- **Consistency.** About 10 % of images are shown a second time as hidden repeats, in a different orientation and at least 10 tasks later. In a later round, images from an earlier round can be mixed in again as anchors.
- **Practical details.**
  - Everything is saved automatically, including offline, with a queue that syncs later.
  - Annotators can stop and continue at any time.
  - Several people can share one tablet ("Switch person").
  - The server is a single Python file using only the standard library.
- **Analysis.**
  - Tracings are mapped back to the original images. This mapping is verified pixel by pixel.
  - Tracings of the same particle are matched across annotators.
  - Consensus particles are those traced by a majority.
  - Size distributions per group come with confidence intervals obtained by resampling images.
  - Agreement statistics: pairwise F1, ICC(2,1), Fleiss' kappa, and repeat consistency.

## How it fits together

```
your images + manifest.csv
        │  prepare.py
        ├──────────────► build/            tasks.json + images/<token>.png   → server
        └──────────────► private/key.json  unblinding key                     → stays with you
server.py  ◄──── tablet browser (annotators trace)
        └──────────────► state/ann/<initials>/<token>.json                    → copy back
analyze.py --key private/key.json --ann state/ann  →  out/ (CSV tables, figure, SUMMARY.md)
```

## Quick start with the demo

Needs Python 3.9 or newer.

```bash
git clone https://github.com/RueMGross/BlindTrace.git && cd BlindTrace
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python demo/make_demo.py                         # 12 synthetic images, two groups
.venv/bin/python prepare.py demo/manifest.csv --title "Demo"
.venv/bin/python server.py --host 0.0.0.0                  # prints the address
```

Open `http://<this computer's IP>:8170/` on a tablet on the same network, enter initials and trace. Then analyse:

```bash
.venv/bin/python analyze.py --key private/key.json --ann state/ann --build build
```

Analysis with two or more annotators gives the consensus and agreement statistics. With a single annotator you get their sizes and repeat consistency.

## Your own images

Write a CSV with one row per image:

| column | required | meaning |
|---|---|---|
| `image` | yes | path to the image, relative to the CSV (PNG, TIFF including 16-bit, JPEG, ...) |
| `nm_per_px` | yes | pixel size in nm |
| `group` | no | sample or condition; used only by the analysis, never shown |
| `image_id` | no | unique name (default: the path) |
| `crop_top`, `crop_bottom`, `crop_left`, `crop_right` | no | pixels to cut off, e.g. a data bar |

```bash
python prepare.py my_images.csv --title "Round 1"        # options: --repeat-frac, --nm-per-px, --stretch
```

`prepare.py` refuses to overwrite an existing key, because a new key orphans any tracings already made against the old one.

### Keeping it blind

- **Cut identifying overlays.** Crop anything that identifies the image or the sample (data bars, burned-in labels).
- **Equalise pixel and image size.** If groups were imaged at different magnifications, both can give the group away. `--nm-per-px` resamples every image to one pixel size.
- **Keep the key private.** Don't put the key on the server or share it with annotators. Don't discuss results with annotators until everyone has finished.
- **Repeat links stay on the server.** They are kept in `tasks.json` and never sent to the browser.

### More rounds

Images added later go into a new round. Mix in some images of the earlier round as anchors, so that annotators can't recognise the new images as new, and so you can measure how consistent the tracing stays across rounds:

```bash
python prepare.py round2.csv --out build_r2 --key private/key_r2.json --title "Round 2" \
                  --previous private/key.json --anchors 8 --round 2
python analyze.py --key private/key.json --key private/key_r2.json \
                  --ann state/ann --ann state_r2/ann --build build --build build_r2
```

## Running a study

Annotator instructions, as shown on the start screen:

- **Pen:** trace the outer boundary of each particle, including its dark rim if it has one. Zoom in to trace precisely.
- **Unsure or wrong outline:** tap the contour to mark it unsure or delete it.
- **Particles cut by the image edge:** trace them anyway and carry the line past the edge.
- **Approve:** when everything is traced, tap Approve. An image with no particles is approved empty. Unusable images are excluded with a reason.
- **Fingers:** drag to move, pinch to zoom.

Each annotator uses their own initials. The task order is fixed per initials, so they can stop and continue later, also on another device.

**Serving.** The server has no login, so run it only on a trusted network. To reach it over the internet, put it behind a reverse proxy with HTTPS and a password. `deploy/` has an nginx and a systemd example; run with `--prefix /blindtrace` under a sub-path. Tracings are written to `state/`; copy `state/ann` back for the analysis.

## Analysis

`analyze.py` writes to `out/`:

| file | contents |
|---|---|
| `particles_all.csv` | every outline in original image coordinates: diameter, axes, edge flag, label |
| `images.csv` | every annotator × task: status, exclusion reason, counts, time spent, `suspect_t_open` (see below) |
| `outline_drops.csv` | with `--log`: saves that removed all outlines of an image (see Quality control) |
| `annotators.csv` | per annotator × group: images, particles, median size with CI |
| `intra_rater.csv` | each image traced twice by the same annotator (repeat or cross-round anchor): F1, size difference, within-particle CV |
| `inter_rater.csv` | pairwise F1; ICC(2,1) of counts per image and of particle sizes; Fleiss' kappa of approve/exclude |
| `consensus_particles.csv` | particles traced by a majority, with mean diameter and who traced them |
| `groups.csv` | per group: D10, D50, D90 with 95 % CIs, mean, SD; D50 per annotator |
| `fig_groups.png`, `SUMMARY.md` | size distributions; key numbers and a draft methods paragraph |

**Method:**
- **Matching.** Two outlines are the same particle if their intersection over union is at least 0.3 (`--iou`). Outlines are matched one-to-one between annotators by the Hungarian algorithm, then joined across annotators as connected components.
  - Why 0.3 and not the more common 0.5: between annotators, the best overlap of each outline is bimodal, with a trough below 0.3. At 0.5, small particles traced a few nm apart are split into two.
- **Consensus.** Images approved by a majority of annotators are analysed. A consensus particle is one traced by a majority of all annotators, and its diameter is the mean of their equivalent circle diameters.
- **Edge particles.** Outlines that reach the image border count as particles but are excluded from sizes.
- **Confidence intervals.** These resample images, not particles, because particles on one image are not independent.

**Quality control.** Versions before 0.1.1 could save an empty tracing over an image that was still loading, e.g. when Back was tapped quickly several times. Version 0.1.1 prevents this. For data collected with an older version, two heuristic checks help find damaged records:
- **Load time.** Every saved record carries the time its image finished loading. A record that shares this time with another image was saved while a different image was open. `analyze.py` warns about these and marks them in `images.csv` (`suspect_t_open`).
- **Audit log.** `--log state/log.jsonl` lists every save that removed all outlines of an image (`outline_drops.csv`). It also catches damage that the load-time check misses because the image was opened again later.

An image that never had outlines and was saved empty during loading cannot be recognised by either check. Look for runs of empty saves only seconds apart in the audit log.

**Excluding records.** To leave damaged records out, list them in a CSV with columns `annotator`, `token` and `reason`, and pass it with `--exclude-records`. Listed records count as missing: the image is then decided by the other annotators.

## Data formats

- **`tasks.json`:** `{"title": ..., "tasks": [{"token", "nm_per_px", "w", "h", "after"}]}`. Here `after` is the token of the original for a hidden repeat, and `null` otherwise.
- **Tracing file `state/ann/<initials>/<token>.json`:**
  - `status`: `done`, `excluded` or `draft`;
  - `reason`;
  - `contours`: a list of `{pts: [[x, y], ...], label: "particle" | "unsure", edge, d_nm, major_nm, minor_nm}`, in served-image pixels, where pixel *i* spans [*i*, *i*+1);
  - timestamps.
- **Key:** per token, the original image, group, preparation (crop, resampling, contrast stretch), orientation (`k90`, `flip`), `repeat_of` and `anchor_of`.

## Tests

```bash
.venv/bin/pip install pytest && .venv/bin/python -m pytest tests/
```

The tests simulate three annotators tracing the known particles of the synthetic images in their served orientations, and check:
- that the analysis recovers each particle's position and size;
- the consensus, repeats and second-round anchors;
- the server API.

## Citation

If you use BlindTrace, please cite it (see `CITATION.cff`). A paper describing its use is in preparation.

## License

MIT, see `LICENSE`.
