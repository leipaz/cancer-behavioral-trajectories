import argparse
from . import constants as c
from .inference import generate_profiles

def main():
    parser = argparse.ArgumentParser(description="Generate oncology VQ-VAE profile pickles.")
    parser.add_argument("--data-csv", default=str(c.DEFAULT_DATA_CSV))
    parser.add_argument("--models-dir", default=str(c.DEFAULT_MODELS_DIR))
    parser.add_argument("--scaler", default=str(c.DEFAULT_SCALER))
    parser.add_argument("--output", default=str(c.DEFAULT_OUTPUT))
    parser.add_argument("--modes", nargs="+", default=list(c.DEFAULT_MODES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=c.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=c.DEFAULT_SEED)
    args = parser.parse_args()

    generate_profiles(
        data_csv=args.data_csv,
        models_dir=args.models_dir,
        scaler_path=args.scaler,
        output_path=args.output,
        modes=tuple(args.modes),
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
