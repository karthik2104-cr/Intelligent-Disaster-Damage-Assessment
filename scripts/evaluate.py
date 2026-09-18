"""Part 3: validation-only evaluation of existing best.pt checkpoints.

Does not train, does not tune, does not use the test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import load_config  # noqa: E402
from ml.data.augmentation import build_eval_pipeline  # noqa: E402
from ml.engine.dataloaders import build_dataloader  # noqa: E402
from ml.engine.metrics import summarise_predictions  # noqa: E402
from ml.engine.reproducibility import seed_everything  # noqa: E402
from ml.engine.train_config import load_train_config, resolve_model_settings  # noqa: E402
from ml.eval.benchmark import benchmark_inference  # noqa: E402
from ml.eval.figures import overlay_heatmap, plot_confidence, plot_confusion_matrices, plot_example_grid  # noqa: E402
from ml.eval.gradcam import GradCAM, tensor_to_image  # noqa: E402
from ml.eval.patterns import (  # noqa: E402
    confidence_breakdown,
    mild_error_patterns,
    select_indices,
)
from ml.eval.predict import (  # noqa: E402
    CLASS_NAMES,
    compare_to_reference,
    collect_predictions,
    load_eval_model,
    parameter_count,
)

MODELS = ("baseline", "custom_cnn", "resnet50", "efficientnet_b0")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _latest_experiment(directory: Path, model: str) -> dict:
    matches = sorted(directory.glob(f"{model}_20*.json"))
    if not matches:
        raise FileNotFoundError(f"no Part 2 experiment JSON for {model}")
    return json.loads(matches[-1].read_text(encoding="utf-8"))


def _display_image(dataset, index: int, preprocessor) -> object:
    tensor, _ = dataset[index]
    hwc = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return preprocessor.denormalise(hwc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="val", choices=["val"], help="development analysis uses val only")
    parser.add_argument("--atol", type=float, default=0.002)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_config = load_config()
    train_config = load_train_config()
    seed_everything(train_config.seed)
    device = _device()
    print(f"device: {device}")
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    figures = Path("reports/figures")
    error_dir = Path("reports/error_analysis")
    eval_dir = Path("reports/evaluation")
    figures.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = build_eval_pipeline(data_config).preprocessor
    all_metrics = {}
    all_match = {}
    all_errors = {}
    all_conf = {}
    param_counts = {}
    infer = {}

    for name in MODELS:
        settings = resolve_model_settings(train_config, name)
        loader = build_dataloader(
            data_config,
            train_config,
            args.split,
            batch_size=settings["batch_size"],
            device=device,
        )
        ckpt = train_config.models_dir / name / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing checkpoint {ckpt}")
        model = load_eval_model(name, ckpt, device)
        param_counts[name] = parameter_count(model)
        print(f"\n=== {name} params={param_counts[name]} ckpt={ckpt} ===")
        bundle = collect_predictions(model, loader, device, use_amp=False)
        summary = summarise_predictions(bundle.y_true, bundle.y_pred, class_names=list(CLASS_NAMES))
        part2 = _latest_experiment(train_config.experiments_dir, name)["val_metrics"]
        match = compare_to_reference(summary, part2, atol=args.atol)
        all_metrics[name] = summary
        all_match[name] = match
        print(json.dumps({k: v for k, v in match.items()}, indent=2))
        if not all(item["ok"] for item in match.values()):
            print("WARNING: reproduced metrics differ from Part 2 beyond atol; retrying with AMP")
            bundle = collect_predictions(model, loader, device, use_amp=True)
            summary = summarise_predictions(bundle.y_true, bundle.y_pred, class_names=list(CLASS_NAMES))
            match = compare_to_reference(summary, part2, atol=args.atol)
            all_metrics[name] = summary
            all_match[name] = match
            print("AMP retry", json.dumps(match, indent=2))

        plot_confusion_matrices(bundle, figures / name)
        if name in ("resnet50", "efficientnet_b0"):
            all_errors[name] = mild_error_patterns(bundle)
            all_conf[name] = confidence_breakdown(bundle)
            plot_confidence(bundle, figures / f"{name}_confidence.png", f"{name} validation confidence")
            (eval_dir / f"{name}_predictions.npz").parent.mkdir(parents=True, exist_ok=True)
            import numpy as np

            np.savez_compressed(
                eval_dir / f"{name}_predictions.npz",
                y_true=bundle.y_true,
                y_pred=bundle.y_pred,
                probs=bundle.probs,
            )

        if name in ("resnet50", "efficientnet_b0"):
            infer[name] = benchmark_inference(model, loader, device)
            print("inference", infer[name])

        dataset = loader.dataset
        if name == "efficientnet_b0":
            _write_error_gallery(bundle, dataset, preprocessor, error_dir, figures)
            _write_gradcam_panel(model, bundle, dataset, preprocessor, device, figures / "efficientnet_b0_gradcam.png", "EfficientNet-B0")
        if name == "resnet50":
            _write_gradcam_panel(model, bundle, dataset, preprocessor, device, figures / "resnet50_gradcam.png", "ResNet-50")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison = []
    for name in MODELS:
        m = all_metrics[name]
        row = {
            "model": name,
            "accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "mild_f1": m["per_class"]["mild"]["f1"],
            "parameters": param_counts[name],
        }
        if name in infer:
            row["ms_per_image"] = infer[name]["ms_per_image"]
            row["images_per_second"] = infer[name]["images_per_second"]
        comparison.append(row)

    payload = {
        "split": args.split,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "class_names": list(CLASS_NAMES),
        "metrics": all_metrics,
        "part2_match": all_match,
        "mild_errors": all_errors,
        "confidence": all_conf,
        "comparison": comparison,
        "inference_benchmark": infer,
        "notes": (
            "Validation split only. best.pt checkpoints. No training. "
            "Test set was not used for this analysis."
        ),
    }
    dest = eval_dir / "part3_validation.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", dest)
    mismatches = [
        f"{model}.{key}"
        for model, report in all_match.items()
        for key, item in report.items()
        if not item["ok"]
    ]
    if mismatches:
        print("MISMATCHES:", mismatches)
        return 1
    print("all reproduced metrics within atol", args.atol)
    return 0


def _write_error_gallery(bundle, dataset, preprocessor, error_dir: Path, figures: Path) -> None:
    names = list(CLASS_NAMES)
    mild, none, severe = names.index("mild"), names.index("little_or_none"), names.index("severe")
    groups = [
        ("A_mild_to_little_or_none", mild, none),
        ("B_mild_to_severe", mild, severe),
        ("C_little_or_none_to_mild", none, mild),
        ("D_severe_to_mild", severe, mild),
    ]
    images, captions, cases = [], [], []
    for label, true_i, pred_i in groups:
        idxs = select_indices(bundle, true_label=true_i, pred_label=pred_i, limit=3)
        for idx in idxs:
            images.append(_display_image(dataset, idx, preprocessor))
            conf = float(bundle.confidence[idx])
            captions.append(
                f"{label.split('_', 1)[0]}\ntrue {names[true_i]}\npred {names[pred_i]}\nconf {conf:.3f}"
            )
            cases.append(
                {
                    "group": label,
                    "val_index": idx,
                    "true": names[true_i],
                    "predicted": names[pred_i],
                    "confidence": conf,
                }
            )
    high_wrong = select_indices(bundle, correct=False, highest_confidence=True, limit=4)
    for idx in high_wrong:
        images.append(_display_image(dataset, idx, preprocessor))
        conf = float(bundle.confidence[idx])
        t, p = names[int(bundle.y_true[idx])], names[int(bundle.y_pred[idx])]
        captions.append(f"E high-conf wrong\ntrue {t}\npred {p}\nconf {conf:.3f}")
        cases.append(
            {
                "group": "E_high_confidence_incorrect",
                "val_index": idx,
                "true": t,
                "predicted": p,
                "confidence": conf,
            }
        )
    plot_example_grid(
        images,
        captions,
        figures / "efficientnet_b0_error_examples.png",
        title="EfficientNet-B0 validation error examples (no training)",
        ncols=4,
    )
    (error_dir / "efficientnet_b0_cases.json").write_text(
        json.dumps(
            {
                "model": "efficientnet_b0",
                "split": "val",
                "checkpoint": "models/efficientnet_b0/best.pt",
                "cases": cases,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_gradcam_panel(model, bundle, dataset, preprocessor, device, path: Path, title: str) -> None:
    names = list(CLASS_NAMES)
    mild, severe = names.index("mild"), names.index("severe")
    specs = [
        ("correct mild", select_indices(bundle, true_label=mild, pred_label=mild, limit=1)),
        ("incorrect mild", select_indices(bundle, true_label=mild, correct=False, limit=1)),
        ("correct severe", select_indices(bundle, true_label=severe, pred_label=severe, limit=1)),
        ("incorrect", select_indices(bundle, correct=False, highest_confidence=True, limit=1)),
    ]
    images, captions = [], []
    with GradCAM(model) as cam:
        for label, idxs in specs:
            if not idxs:
                continue
            idx = idxs[0]
            tensor, _ = dataset[idx]
            batch = tensor.unsqueeze(0).to(device)
            pred = int(bundle.y_pred[idx])
            heatmap = cam(batch, pred)
            rgb = tensor_to_image(tensor, preprocessor.denormalise)
            overlay = overlay_heatmap(rgb, heatmap)
            images.append(overlay)
            conf = float(bundle.confidence[idx])
            captions.append(
                f"{label}\ntrue {names[int(bundle.y_true[idx])]} / pred {names[pred]}\nconf {conf:.3f}"
            )
    if images:
        plot_example_grid(images, captions, path, title=f"Grad-CAM ({title}) — predicted class", ncols=2)


if __name__ == "__main__":
    raise SystemExit(main())
