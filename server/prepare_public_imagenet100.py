"""Prepare AutoDL's mounted train subset and matching OFFICIAL validation images.

Creates new byte-preserving image files and immutable manifests. It never runs
the provider's full-ImageNet extraction script or makes a random train/val split.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, deque
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import zipfile

from PIL import Image
import scipy.io


def sha(data):
    return hashlib.sha256(data).hexdigest()


def image_record(data, destination, relative, label):
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGB").load()  # fail on a corrupt/truncated image now
    path = destination / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    content_hash = sha(data)
    if path.exists():
        if sha(path.read_bytes()) != content_hash:
            raise ValueError(f"Existing image differs from archive: {path}")
    else:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(data)
        temp.replace(path)
    return {"path": relative.as_posix(), "label": label, "sha256": content_hash,
            "id": content_hash, "bytes": len(data)}


def finalize_records(out, classes, records, provenance):
    from sepa_plan_b.config import digest
    from sepa_plan_b.data import validate_pair, write_json
    if (out / "train.json").exists() or (out / "val.json").exists():
        raise FileExistsError("Manifests already exist")
    if len(records["train"]) != provenance["expected_train"] or len(records["val"]) != 5000:
        raise ValueError("Extracted record count mismatch")
    validation_hashes = {r["sha256"] for r in records["val"]}
    excluded = [r for r in records["train"] if r["sha256"] in validation_hashes]
    records["train"] = [r for r in records["train"] if r["sha256"] not in validation_hashes]
    write_json(out / "duplicates_excluded.json", {
        "policy": "Keep the official validation set; exclude every byte-identical image from the pretraining manifest. No image files are deleted.",
        "excluded_train_records": excluded,
        "matching_validation_records": [r for r in records["val"] if r["sha256"] in {e["sha256"] for e in excluded}]})
    manifests = {}
    for role in ("train", "val"):
        payload = {"schema": 1, "role": role, "root": str(out / "images" / role), "classes": classes,
                   "records": sorted(records[role], key=lambda r: r["path"])}
        payload["fingerprint"] = digest(payload)
        manifests[role] = payload
    validate_pair(manifests["train"], manifests["val"], 100)
    for role, manifest in manifests.items():
        write_json(out / f"{role}.json", manifest)
    provenance.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat(),
                      excluded_train_duplicates=len(excluded), actual_train=len(records["train"]), actual_val=len(records["val"]),
                      fingerprints={role: m["fingerprint"] for role, m in manifests.items()},
                      train_class_counts=dict(Counter(classes[r["label"]] for r in records["train"])),
                      val_class_counts=dict(Counter(classes[r["label"]] for r in records["val"])))
    write_json(out / "preparation.json", provenance)
    print(json.dumps({"complete": True, "train": len(records["train"]), "val": len(records["val"]),
                      "excluded_train_duplicates": len(excluded), "out": str(out)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-zip", default="/root/autodl-pub/ImageNet100/imagenet100.zip")
    parser.add_argument("--val-tar", default="/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar")
    parser.add_argument("--devkit", default="/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz")
    args = parser.parse_args()
    root = Path(args.project).resolve()
    sys.path.insert(0, str(root / "src"))
    from sepa_plan_b.data import write_json
    out = root / "data/imagenet100"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() or (out / "val.json").exists():
        raise FileExistsError("Manifests already exist; inspect them before rerunning data preparation")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    provenance = {"source": "AutoDL mounted ImageNet100 training archive + ILSVRC2012 official validation/devkit",
                  "started_utc": datetime.now(timezone.utc).isoformat(), "archives": {}}
    for name in ("train_zip", "val_tar", "devkit"):
        p = Path(getattr(args, name))
        provenance["archives"][name] = {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
    with zipfile.ZipFile(args.train_zip) as archive:
        infos = [i for i in archive.infolist() if not i.is_dir()]
        for info in infos:
            parts = PurePosixPath(info.filename).parts
            if (len(parts) != 3 or parts[0] != "imagenet100" or not re.fullmatch(r"n\d{8}", parts[1])
                    or not parts[2].startswith(parts[1] + "_") or not parts[2].lower().endswith((".jpeg", ".jpg", ".png"))):
                raise ValueError(f"Unexpected train archive path: {info.filename}")
        classes = sorted({PurePosixPath(i.filename).parts[1] for i in infos})
        if len(classes) != 100:
            raise ValueError(f"Expected 100 classes, found {len(classes)}")
        if len({i.filename for i in infos}) != len(infos):
            raise ValueError("Duplicate filenames in train archive")
        class_index = {name: i for i, name in enumerate(classes)}
        # Reserve space for 40 ViT-S checkpoints and temporary checkpoint replacement.
        if shutil.disk_usage(root).free < sum(i.file_size for i in infos) + 20 * 2**30:
            raise RuntimeError("Insufficient free space for data plus experiment checkpoints")
        with tarfile.open(args.devkit, "r:gz") as devkit:
            meta_name = next(n for n in devkit.getnames() if n.endswith("/meta.mat"))
            labels_name = next(n for n in devkit.getnames() if n.endswith("/ILSVRC2012_validation_ground_truth.txt"))
            meta_bytes, label_bytes = devkit.extractfile(meta_name).read(), devkit.extractfile(labels_name).read()
        synsets = scipy.io.loadmat(io.BytesIO(meta_bytes), squeeze_me=True, struct_as_record=False)["synsets"]
        label_to_synset = {int(s.ILSVRC2012_ID): str(s.WNID) for s in synsets if int(s.num_children) == 0}
        if set(label_to_synset) != set(range(1, 1001)) or not set(classes) <= set(label_to_synset.values()):
            raise ValueError("Official synset mapping does not cover the selected classes")
        val_labels = [int(s) for s in label_bytes.decode().split()]
        if len(val_labels) != 50000 or not set(val_labels) <= set(label_to_synset):
            raise ValueError("Invalid official validation label file")
        selected_val = {i+1: label_to_synset[label] for i, label in enumerate(val_labels)
                        if label_to_synset[label] in class_index}
        if Counter(selected_val.values()) != Counter({c: 50 for c in classes}):
            raise ValueError("Expected exactly 50 official validation images per class")
        provenance.update(classes=classes, classes_sha256=sha(("\n".join(classes)+"\n").encode()),
                          meta_sha256=sha(meta_bytes), validation_labels_sha256=sha(label_bytes),
                          devkit_sha256=sha(Path(args.devkit).read_bytes()),
                          expected_train=len(infos), expected_val=len(selected_val))
        write_json(out / "preparation.json", provenance)
        (out / "classes.txt").write_text("\n".join(classes)+"\n")
        records = {"train": [], "val": []}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = deque()
            for i, info in enumerate(infos, 1):
                _, name, filename = PurePosixPath(info.filename).parts
                data = archive.read(info)  # ZIP checks CRC while reading
                pending.append(pool.submit(image_record, data, out / "images/train", Path(name) / filename, class_index[name]))
                if len(pending) >= args.workers * 2:
                    records["train"].append(pending.popleft().result())
                if i % 10000 == 0:
                    print(f"train decoded/extracted {i}/{len(infos)}", flush=True)
            records["train"].extend(f.result() for f in pending)
            pending.clear()
            with tarfile.open(args.val_tar, "r:") as archive_val:
                seen = set()
                for member in archive_val:
                    if not member.isfile():
                        continue
                    name = PurePosixPath(member.name).name
                    match = re.fullmatch(r"ILSVRC2012_val_(\d{8})\.JPEG", name)
                    if not match:
                        raise ValueError(f"Unexpected validation filename: {member.name}")
                    index = int(match.group(1))
                    if index in seen or not 1 <= index <= 50000:
                        raise ValueError("Duplicate/out-of-range validation image index")
                    seen.add(index)
                    if index not in selected_val:
                        continue
                    cls = selected_val[index]
                    data = archive_val.extractfile(member).read()
                    pending.append(pool.submit(image_record, data, out / "images/val", Path(cls) / name, class_index[cls]))
                    if len(pending) >= args.workers * 2:
                        records["val"].append(pending.popleft().result())
                if seen != set(range(1, 50001)):
                    raise ValueError("Official validation archive is incomplete")
            records["val"].extend(f.result() for f in pending)
    finalize_records(out, classes, records, provenance)


if __name__ == "__main__":
    main()
