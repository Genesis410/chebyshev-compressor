import json
from pathlib import Path
from chebyshev.models import StudentModel
from chebyshev.exporter import export_to_c_header
from chebyshev.benchmark import run_c_benchmark

results = {}
for K in [4, 8, 16, 32]:
    hidden = 16 if K == 4 else (32 if K <= 16 else 64)
    model = StudentModel(num_channels=14, order=K, hidden_dim=hidden, out_features=1)
    
    # Export header
    header_path = Path("chebyshev_engine_bench.h")
    export_to_c_header(model, header_path)
    
    # Run benchmark
    bench = run_c_benchmark(header_path, iterations=100000, seq_len=50)
    results[f"K={K}"] = bench
    print(f"K={K} (H={hidden}):")
    print(f"  Single-step latency: {bench['latency_us']:.4f} us ({bench['latency_ns']:.1f} ns)")
    print(f"  Window latency (T=50): {bench['window_latency_us']:.4f} us")
    print(f"  Throughput: {bench['throughput_steps_sec']:,.0f} steps/sec")
    print(f"  Persistent State RAM: {bench['persistent_state_bytes']} B")
    print(f"  Stack Scratch RAM: {bench['stack_scratch_bytes']} B")
    print(f"  Total SRAM: {bench['total_sram_bytes']} B")

if Path("chebyshev_engine_bench.h").exists():
    Path("chebyshev_engine_bench.h").unlink()

Path("benchmark_results_host.json").write_text(json.dumps(results, indent=2))
