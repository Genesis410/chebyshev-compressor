"""Reproducible Polynomial Order Ablation Study (K = 4, 8, 16, 32).

Runs deterministic signal reconstruction, Option A distillation on held-out
C-MAPSS FD001 engines, C99 export, and native streaming latency benchmarks.
"""

import argparse
import json
from pathlib import Path

from chebyshev.benchmark import run_order_ablation


def format_markdown_table(results: list) -> str:
    header = (
        "| Order K | Recon Rel L2 Error | Teacher Val RMSE (cycles) | Student Val RMSE (cycles) | "
        "Persistent State (B) | Stack Scratch (B) | Total SRAM (B) | C99 Step Latency (us) |"
    )
    sep = "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, sep]
    for r in results:
        rows.append(
            f"| {r['order_k']} | {r['recon_rel_l2_error']:.4e} | {r['teacher_val_rmse_cycles']:.2f} | "
            f"{r['student_val_rmse_cycles']:.2f} | {r['persistent_state_bytes']} | "
            f"{r['stack_scratch_bytes']} | {r['total_sram_bytes']} | {r['c_latency_us']:.3f} |"
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Chebyshev order K ablation study.")
    parser.add_argument("--orders", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-engines", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="cheb_out/ablation_results.json")
    args = parser.parse_args()

    results = run_order_ablation(
        orders=args.orders,
        window_size=args.window_size,
        epochs=args.epochs,
        seed=args.seed,
        c_iterations=args.iterations,
        max_engines=args.max_engines,
    )

    print("\n=== CHEBYSHEV ORDER ABLATION RESULTS ===")
    print(format_markdown_table(results))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ablation JSON to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
