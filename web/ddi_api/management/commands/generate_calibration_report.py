"""
Generate a calibration QA report (ECE, Brier, confidence intervals) from CSV data.

Expected CSV columns by default:
- label
- raw_score
- calibrated_score

Example:
    python manage.py generate_calibration_report \
        --input-csv data/calibration_eval.csv \
        --output-json reports/calibration_report.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from ddi_api.services.calibration_metrics import generate_calibration_report


class Command(BaseCommand):
    help = "Generate calibration QA report (ECE/Brier/CI) from a labeled CSV"

    def add_arguments(self, parser):
        parser.add_argument(
            "--input-csv",
            required=True,
            help="Path to CSV file containing labels and prediction scores",
        )
        parser.add_argument(
            "--output-json",
            default="reports/calibration_report.json",
            help="Output path for report JSON (default: reports/calibration_report.json)",
        )
        parser.add_argument(
            "--label-column",
            default="label",
            help="Column name for binary labels (0/1)",
        )
        parser.add_argument(
            "--raw-column",
            default="raw_score",
            help="Column name for raw model scores",
        )
        parser.add_argument(
            "--calibrated-column",
            default="calibrated_score",
            help="Column name for calibrated model scores",
        )
        parser.add_argument(
            "--bins",
            type=int,
            default=10,
            help="Number of confidence bins for ECE (default: 10)",
        )
        parser.add_argument(
            "--bootstrap",
            type=int,
            default=1000,
            help="Bootstrap sample count for CIs (default: 1000)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for bootstrap (default: 42)",
        )
        parser.add_argument(
            "--delimiter",
            default=",",
            help="CSV delimiter (default: ,)",
        )

    def handle(self, *args, **options):
        input_path = Path(options["input_csv"]).expanduser().resolve()
        output_path = Path(options["output_json"]).expanduser()
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()

        if not input_path.exists():
            raise CommandError(f"Input CSV does not exist: {input_path}")

        label_col = options["label_column"]
        raw_col = options["raw_column"]
        calibrated_col = options["calibrated_column"]

        self.stdout.write(self.style.SUCCESS("=" * 64))
        self.stdout.write(self.style.SUCCESS("Calibration QA Report"))
        self.stdout.write(self.style.SUCCESS("=" * 64))
        self.stdout.write(f"Input: {input_path}")

        try:
            df = pd.read_csv(input_path, delimiter=options["delimiter"])
        except Exception as exc:
            raise CommandError(f"Failed to read CSV: {exc}") from exc

        missing_cols = [c for c in [label_col, raw_col, calibrated_col] if c not in df.columns]
        if missing_cols:
            raise CommandError(
                "Missing required columns: " + ", ".join(missing_cols)
            )

        working = df[[label_col, raw_col, calibrated_col]].copy()
        before_count = len(working)

        working[label_col] = pd.to_numeric(working[label_col], errors="coerce")
        working[raw_col] = pd.to_numeric(working[raw_col], errors="coerce")
        working[calibrated_col] = pd.to_numeric(working[calibrated_col], errors="coerce")
        working = working.dropna()

        dropped_count = before_count - len(working)
        if working.empty:
            raise CommandError("No valid rows remain after numeric parsing and NaN filtering")

        labels = working[label_col].astype(int).tolist()
        raw_scores = working[raw_col].astype(float).tolist()
        calibrated_scores = working[calibrated_col].astype(float).tolist()

        try:
            report = generate_calibration_report(
                labels=labels,
                raw_scores=raw_scores,
                calibrated_scores=calibrated_scores,
                n_bins=options["bins"],
                n_bootstrap=options["bootstrap"],
                seed=options["seed"],
            )
        except Exception as exc:
            raise CommandError(f"Failed to compute calibration metrics: {exc}") from exc

        report["meta"].update(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_csv": str(input_path),
                "output_json": str(output_path),
                "rows_read": int(before_count),
                "rows_used": int(len(working)),
                "rows_dropped": int(dropped_count),
                "label_column": label_col,
                "raw_column": raw_col,
                "calibrated_column": calibrated_col,
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

        self.stdout.write(self.style.SUCCESS(f"Report written: {output_path}"))
        self.stdout.write(
            f"Samples: {report['meta']['n_samples']} | Positive rate: {report['meta']['positive_rate']:.4f}"
        )
        self.stdout.write(
            f"Brier improvement: {report['delta']['brier_improvement']:.6f} | "
            f"ECE improvement: {report['delta']['ece_improvement']:.6f}"
        )
