#!/usr/bin/env python3
"""
Prepare gRefCOCO: one record per referring group, split into
train/val/testA/testB.

Pipeline
  1. read grefs(unc).json                 -> ref_id, image_id, split, sentences, ann_id
  2. resolve ann_id in COCO instances     -> bbox / segmentation / area / iscrowd
  3. attach the image captions            -> COCO captions_train2014 (5 per image)
  4. write <out_path>/grefcoco_<split>.json, keyed by ref_id

Sample unit is the ref (global key ref_id, 94050 unique). Which paraphrase to
feed the model is a loader concern, resampled every epoch:

    split = json.load(open("grefcoco_train.json"))
    ref   = random.choice(split["refs"])
    sent  = random.choice(ref["sentences"])            # paraphrase augmentation
    targets = [t for t in ref["targets"] if not t["iscrowd"]]

Inputs
  --data_path        required; holds grefs(unc).json, and usually annotations/
  --instances_file   defaults to <data_path>/instances.json when present, else
                     <data_path>/annotations/instances_train2014.json. Both give
                     identical targets (same annotation ids), the full COCO file
                     is just slower.
  --captions_file    defaults to <data_path>/annotations/captions_train2014.json,
                     mirrored into every record as "captions" (no loader join).

iscrowd targets are kept here (85 refs, one crowd target each); the official
ReLA loader skips them: `if t["iscrowd"]: continue`.

Output
{
    "split": "train",
    "num_refs": 78930,
    "image_root": "data/grefs/train2014" | null,
    "refs": [
        {
            "ref_id": 0,                       # global primary key
            "image_id": 72,
            "file_name": "COCO_train2014_000000000072.jpg",   # join with image_root
            "no_target": false,
            "image": {"height": 480, "width": 640},
            "captions": ["A man riding a wave ...", ...],   # 5 per image, COCO order
            "sentences": [
                {"sent_id": 0, "sent": "...", "raw": "...", "tokens": [...]}
            ],
            "targets": [
                {
                    "id": 600276,              # COCO annotation id, -1 when empty
                    "category_id": 25,         # -1 when empty
                    "bbox": [x, y, w, h],      # null when empty
                    "segmentation": [...],     # polygon list, or RLE dict when iscrowd == 1
                    "area": 1234.5,
                    "iscrowd": 0,
                    "empty": false
                }
            ]
        }
    ]
}

Notes
- bbox / area / category are meaningless for empty targets (null / 0 / -1):
  skip them in box normalization and IoU.
- val/testA/testB are anchored on train2014 images, one image root covers all
  splits.
- sent_id is globally unique but sparse; 13356 refs repeat a sentence text
  inside themselves, kept as-is.
- captions belong to the image and are repeated across the ~4.7 refs sharing it
  (+27 MB total). 19994/19994 images are covered.
"""

import os
import argparse
from csv import Error
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
import errno

SPLITS = ("train", "val", "testA", "testB")


class RowError(RuntimeError):
    """A single ref cannot be converted faithfully."""


def load_refs(data_path: Path):
    """Load refs from grefs(unc).json."""
    name = "grefs(unc).json"
    path = data_path / name
    if not path.exists():
        raise SystemExit(f"instances file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        refs = json.load(f)
    print(f"Loaded {len(refs)} refs from {path}")
    return refs


def load_instances(instances_file: Path):
    """Load COCO instance annotations (bbox / segmentation / area / iscrowd)."""
    if not instances_file.exists():
        raise SystemExit(f"instances file not found: {instances_file}")

    with open(instances_file, "r", encoding="utf-8") as f:
        instances = json.load(f)

    for key in ("images", "annotations"):
        if key not in instances:
            raise SystemExit(f"{instances_file}: missing '{key}' key")

    print(
        f"Loaded {len(instances['images'])} images, {len(instances['annotations'])} annotations from {instances_file}"
    )
    return instances


def load_captions(captions_file: Path):
    """Index COCO captions by image id, keeping the annotation order."""
    if not captions_file.exists():
        raise SystemExit(f"captions file not found: {captions_file}")

    with open(captions_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "annotations" not in data:
        raise SystemExit(f"{captions_file}: missing 'annotations' key")

    captions = {}
    for ann in data["annotations"]:
        captions.setdefault(ann["image_id"], []).append(ann["caption"])
    print(f"Loaded captions for {len(captions)} images from {captions_file}")
    return captions


def build_lookups(instances):
    """Index annotations and images by id."""
    anns = {a["id"]: a for a in instances["annotations"]}
    imgs = {i["id"]: i for i in instances["images"]}
    if len(anns) != len(instances["annotations"]) or len(imgs) != len(
        instances["images"]
    ):
        raise SystemExit("instances file has duplicate image or annotation ids")
    return anns, imgs


def _as_list(value):
    """gRefCOCO stores ann_id as a list; tolerate the scalar variant."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def build_ref(ref, anns, imgs, captions):
    """Convert one raw ref into an output record. Raises RowError on bad data."""
    ref_id, image_id = ref["ref_id"], ref["image_id"]

    img = imgs.get(image_id)
    if img is None:
        raise RowError(f"image_id {image_id} missing from the instances file")
    for key in ("height", "width", "file_name"):
        if key not in img:
            raise RowError(f"image_id {image_id} has no '{key}' in the instances file")

    sentences = [
        {
            "sent_id": s["sent_id"],
            "sent": s["sent"],
            "raw": s.get("raw"),
            "tokens": s["tokens"],
        }
        for s in ref.get("sentences", [])
    ]
    if not sentences:
        raise RowError("ref has no sentences")

    targets = []
    for ann_id in _as_list(ref["ann_id"]):
        if ann_id == -1:
            targets.append(
                {
                    "id": -1,
                    "category_id": -1,
                    "bbox": None,
                    "segmentation": [],
                    "area": 0,
                    "iscrowd": 0,
                    "empty": True,
                }
            )
            continue

        ann = anns.get(ann_id)
        if ann is None:
            raise RowError(f"ann_id {ann_id} missing from the instances file")
        if ann.get("image_id") != image_id:
            raise RowError(
                f"ann_id {ann_id} belongs to image {ann.get('image_id')}, not {image_id}"
            )

        bbox = ann["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise RowError(f"ann_id {ann_id} has malformed bbox: {bbox!r}")

        segmentation = ann.get("segmentation", [])
        if isinstance(segmentation, dict):
            segmentation = {
                "size": segmentation.get("size"),
                "counts": segmentation.get("counts"),
            }

        targets.append(
            {
                "id": ann_id,
                "category_id": ann["category_id"],
                "bbox": bbox,
                "segmentation": segmentation,
                "area": ann.get("area", 0),
                "iscrowd": ann.get("iscrowd", 0),
                "empty": False,
            }
        )

    return {
        "ref_id": ref_id,
        "image_id": image_id,
        "file_name": img["file_name"],
        "no_target": all(t["empty"] for t in targets),
        "image": {"height": img["height"], "width": img["width"]},
        "captions": captions.get(image_id, []),
        "sentences": sentences,
        "targets": targets,
    }


def prepare_split(refs, anns, imgs, captions, split_name, strict=True):
    """Build the records of one split."""
    split_refs = [ref for ref in refs if ref["split"] == split_name]
    print(f"\nPreparing {split_name}: {len(split_refs)} refs")
    if not split_refs:
        raise SystemExit(f"Split '{split_name}' has no refs; check --data_path")

    records, dropped = [], 0
    for ref in split_refs:
        try:
            records.append(build_ref(ref, anns, imgs, captions))
        except RowError as e:
            if strict:
                raise
            dropped += 1
            print(f"  [drop] ref {ref.get('ref_id', 'unknown')}: {e}")

    num_expr = sum(len(r["sentences"]) for r in records)
    num_tgt = sum(len(r["targets"]) for r in records)
    num_empty = sum(1 for r in records for t in r["targets"] if t["empty"])
    num_no_cap = sum(1 for r in records if not r["captions"])
    print(
        f"  {len(records)} refs / {num_expr} expressions / {num_tgt} targets ({num_empty} empty), {dropped} dropped"
        + (f", {num_no_cap} refs without captions" if num_no_cap else "")
    )
    return records


def write_json(path: Path, payload):
    """Write JSON atomically, compactly."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Prepare gRefCOCO split files")
    parser.add_argument(
        "--data_path",
        required=True,
        help="Directory with grefs(unc).json (and annotations/)",
    )
    parser.add_argument(
        "--out_path", required=True, help="Output directory for grefcoco_<split>.json"
    )
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument(
        "--instances_file",
        default=None,
        help="COCO instances file; defaults to <data_path>/instances.json, else <data_path>/annotations/instances_train2014.json",
    )
    parser.add_argument(
        "--captions_file",
        default=None,
        help="COCO captions file; defaults to <data_path>/annotations/captions_train2014.json",
    )
    parser.add_argument(
        "--image_root",
        default=None,
        help="Image directory recorded in the output as 'image_root'",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Skip refs with bad annotations instead of aborting",
    )
    args = parser.parse_args()

    data_path, out_path = Path(args.data_path), Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    refs = load_refs(data_path)
    if args.instances_file:
        instances_file = Path(args.instances_file)
    else:
        instances_file = data_path / "instances.json"
        if not instances_file.exists():
            instances_file = data_path / "annotations/instances_train2014.json"
    captions_file = (
        Path(args.captions_file)
        if args.captions_file
        else data_path / "annotations/captions_train2014.json"
    )

    instances = load_instances(instances_file)
    captions = load_captions(captions_file)

    split_counts = Counter(ref["split"] for ref in refs)
    for split_name in args.splits:
        if split_counts.get(split_name, 0) == 0:
            raise SystemExit(
                f"Split '{split_name}' has no refs. Available: {dict(sorted(split_counts.items()))}"
            )

    anns, imgs = build_lookups(instances)

    written = []
    for split_name in args.splits:
        try:
            records = prepare_split(
                refs, anns, imgs, captions, split_name, strict=args.strict
            )
        except RowError as e:
            print(
                f"Error in split '{split_name}': {e}\nRerun with --no-strict to drop bad refs.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        out_file = out_path / f"grefcoco_{split_name}.json"
        write_json(
            out_file,
            {
                "split": split_name,
                "num_refs": len(records),
                "image_root": args.image_root,
                "refs": records,
            },
        )
        print(f"  wrote {out_file}")
        written.append(out_file)

    print("\nOutput:")
    for out_file in written:
        print(f"  {out_file.name}: {out_file.stat().st_size / 2**20:.2f} MB")


if __name__ == "__main__":
    main()
