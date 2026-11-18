"""Orquesta la comparación, toma muestras de recursos y exporta resultados."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
from statistics import mean, stdev
import subprocess
import sys
from time import sleep

import psutil


PHASES = ("prep_s", "training_s", "validation_s", "total_s")


def capture_gpu():
    if not shutil.which("nvidia-smi"):
        return {}
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total",
         "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True, timeout=5,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"gpu_error": result.stderr.strip() or result.stdout.strip()}
    keys = ("gpu", "gpu_util_pct", "gpu_memory_util_pct", "gpu_memory_mib", "gpu_total_mib")
    return dict(zip(keys, (value.strip() for value in result.stdout.strip().split(",")), strict=True))


def monitored(command, path, label):
    print(f"\n--- {label} ---", flush=True)
    with path.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(command)
        psutil.cpu_percent(percpu=True)
        try:
            while proc.poll() is None:
                sleep(1)
                sample = {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "phase": label,
                    "cpu_pct_by_core": psutil.cpu_percent(percpu=True),
                    **capture_gpu(),
                }
                out.write(json.dumps(sample) + "\n")
                out.flush()
                # Muestra legible en vivo para las capturas del notebook.
                if proc.poll() is None:
                    print(json.dumps(sample, ensure_ascii=False), flush=True)
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)


def summarize(rows, modes):
    summary = []
    for phase in PHASES:
        values = [[row[phase] for row in rows if row["mode"] == mode] for mode in modes]
        record = {"phase": phase}
        for mode, times in zip(modes, values, strict=True):
            record[f"{mode}_mean_s"] = mean(times)
            record[f"{mode}_sd_s"] = stdev(times) if len(times) > 1 else None
        if len(modes) == 2:
            record["speedup"] = mean(values[0]) / mean(values[1])
        summary.append(record)
    return summary


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/metrics/split_manifest.csv"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--cpu-only", action="store_true", help="Prueba local sin GPU.")
    parser.add_argument("--gpu-first", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.epochs < 1:
        parser.error("Las repeticiones y épocas deben ser positivas.")
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else psutil.cpu_count()
    if not args.cpu_only and (cores < 2 or "gpu" not in capture_gpu()):
        parser.error("La comparación completa requiere al menos dos núcleos y una GPU visible en nvidia-smi.")

    args.out.mkdir(parents=True, exist_ok=False)
    device, *_ = platform.processor(), platform.machine()
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        device = next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                       if line.startswith("model name")), device)
    meta = {
        "cpu": device, "available_cores": cores, "platform": platform.platform(),
        "python": platform.python_version(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "repetitions": args.repetitions, "epochs": args.epochs, **capture_gpu(),
    }
    (args.out / "environment.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    modes = ["cpu"] if args.cpu_only else ["cpu", "cuda"]
    records = []
    expected_hash = None
    arrays = args.out / "arrays"
    for repeat in range(1, args.repetitions + 1):
        reverse = (repeat % 2 == 0) != args.gpu_first
        for mode in modes[::-1] if reverse else modes:
            tag = f"{repeat:02d}_{mode}"
            workers = 1 if mode == "cpu" else min(4, cores)
            monitored(
                [sys.executable, "-m", "p3.data", "--manifest", str(args.manifest), "--out", str(arrays),
                 "--workers", str(workers)],
                args.out / f"{tag}_prep_resources.jsonl", f"{tag} preparación {workers} CPU",
            )
            prep = json.loads((arrays / "preparation.json").read_text(encoding="utf-8"))
            expected_hash = expected_hash or prep["arrays_sha256"]
            if prep["arrays_sha256"] != expected_hash:
                raise RuntimeError("Las versiones no prepararon los mismos datos.")
            result = args.out / f"{tag}_training.json"
            monitored(
                [sys.executable, "-m", "p3.train", "--arrays", str(arrays), "--mode", mode,
                 "--epochs", str(args.epochs), "--output", str(result)],
                args.out / f"{tag}_train_resources.jsonl", f"{tag} entrenamiento",
            )
            run = json.loads(result.read_text(encoding="utf-8"))
            records.append({
                "repeat": repeat, "mode": mode, "workers": workers, "examples": run["examples"],
                "epochs": run["epochs"], "prep_s": prep["prep_s"], "training_s": run["training_s"],
                "validation_s": run["validation_s"],
                "total_s": prep["prep_s"] + run["training_s"] + run["validation_s"],
                "h2d_s": run["h2d_s"], "compute_s": run["compute_s"], "d2h_s": run["d2h_s"],
                "imgs_per_s": run["imgs_per_s"], "val_accuracy": run["history"][-1]["accuracy"],
                "val_loss": run["history"][-1]["loss"], "arrays_sha256": expected_hash,
            })
            write_csv(args.out / "runs.csv", records)
    write_csv(args.out / "summary.csv", summarize(records, modes))
    print((args.out / "summary.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
