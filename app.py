import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from materials_data_lab.demo import create_demo
from materials_data_lab.model_artifact import ArtifactError, load_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Materials Data Lab Local Demo (load-only artifacts)")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Path to pre-trained Schema v2 models directory (default: models).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to raw CSV strictly for SHA-256 verification (never triggers training).",
    )
    args = parser.parse_args()

    try:
        models, meta = load_artifact(
            models_dir=args.models_dir,
            csv_path=args.csv,
            verify_csv_hash=args.csv is not None,
        )
    except ArtifactError as exc:
        print(f"MODEL NOT READY ({type(exc).__name__}): {exc}", file=sys.stderr)
        print(
            "Explicit training command: python -m materials_data_lab.model_artifact --csv \"data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv\"",
            file=sys.stderr,
        )
        return 1

    print("Loaded existing model artifact.")
    if meta.get("version_mismatches"):
        print(f"Runtime library version note: {meta['version_mismatches']}")

    demo = create_demo(models, meta)
    # Launch strictly locally without share=True
    demo.launch(inbrowser=True, share=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
