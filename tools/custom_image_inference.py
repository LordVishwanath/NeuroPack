#!/usr/bin/env python3
"""
Utility for post-training testing with custom images.

Features:
1) Convert custom images into NeuroPack test stimulus format.
2) Read NeuroPack `.npz` output and report test accuracy/predictions.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def infer_label_from_name(path_obj):
    stem = path_obj.stem
    token = stem.split("_")[0]
    return int(token)


def image_to_active_pixels(path_obj, side, threshold):
    img = Image.open(path_obj).convert("L").resize((side, side), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    active = np.where(arr < threshold)
    pixel_ids = active[0] * side + active[1]
    return pixel_ids.tolist()


def build_stim_line(timestamp, one_based_neurons):
    serialized = ",".join(str(x) for x in one_based_neurons)
    return f"{timestamp} - {serialized}"


def convert_images_to_stim(image_dir, output_file, input_num, output_num, threshold):
    side = int(np.sqrt(input_num))
    if side * side != input_num:
        raise ValueError("input_num must be a perfect square for image conversion.")

    image_paths = sorted(
        [p for p in Path(image_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    )
    if not image_paths:
        raise ValueError("No images found in image_dir.")

    lines = []
    for idx, image_path in enumerate(image_paths):
        active = image_to_active_pixels(image_path, side, threshold)
        # pixels are indexed in the first input_num neurons, 1-based for stim files
        spike_ids = [pix + 1 for pix in active]
        if output_num > 0:
            label = infer_label_from_name(image_path)
            if label < 0 or label >= output_num:
                raise ValueError(f"Label {label} out of range [0, {output_num - 1}] for file {image_path.name}.")
            spike_ids.append(input_num + label + 1)
        lines.append(build_stim_line(idx, spike_ids))

    with open(output_file, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")

    return len(image_paths)


def evaluate_npz(npz_file):
    data = np.load(npz_file)
    if "testPredictions" not in data or "testLabels" not in data:
        raise ValueError("NPZ file does not contain testPredictions/testLabels.")
    preds = data["testPredictions"]
    labels = data["testLabels"]
    if len(preds) == 0:
        raise ValueError("No predictions found in NPZ.")
    accuracy = float(np.mean(preds == labels))
    return accuracy, preds, labels


def main():
    parser = argparse.ArgumentParser(description="Custom image testing helper for NeuroPack.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    convert_parser = sub.add_parser("convert", help="Convert images into test stim txt.")
    convert_parser.add_argument("--image-dir", required=True, help="Directory of custom images.")
    convert_parser.add_argument("--output-file", required=True, help="Output stim txt file.")
    convert_parser.add_argument("--input-num", type=int, required=True, help="Input neuron count (e.g., 484 for 22x22).")
    convert_parser.add_argument("--output-num", type=int, default=10, help="Output classes.")
    convert_parser.add_argument("--threshold", type=float, default=0.55, help="Pixel activation threshold in [0,1].")

    eval_parser = sub.add_parser("evaluate", help="Evaluate test accuracy from NeuroPack npz.")
    eval_parser.add_argument("--npz-file", required=True, help="Saved NeuroPack npz output.")

    args = parser.parse_args()

    if args.cmd == "convert":
        count = convert_images_to_stim(
            image_dir=args.image_dir,
            output_file=args.output_file,
            input_num=args.input_num,
            output_num=args.output_num,
            threshold=args.threshold,
        )
        print(f"Generated {args.output_file} with {count} samples from {os.path.abspath(args.image_dir)}.")
    elif args.cmd == "evaluate":
        accuracy, preds, labels = evaluate_npz(args.npz_file)
        print(f"Accuracy: {accuracy * 100:.2f}% ({np.sum(preds == labels)}/{len(labels)})")
        print("Predictions:", preds.tolist())
        print("Labels:", labels.tolist())


if __name__ == "__main__":
    main()
