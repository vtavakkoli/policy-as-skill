# Physical edge benchmark protocol

The repository can collect edge measurements, but a paper must only claim a physical edge evaluation when the experiment is actually run on the stated target hardware.

## Measurements

For each target device record:

- exact device/model and OS;
- configured LLM and quantization where applicable;
- CPU count and available RAM;
- GPU model/driver and memory when available;
- warm-up/cold-start latency;
- steady-state mean, median, p95 and p99 latency;
- throughput in tasks/s;
- process resource usage;
- GPU utilization, memory utilization, temperature and power when `nvidia-smi` exposes them;
- Jetson device-tree model and one `tegrastats` sample when available.

## Run

```bash
python scripts/edge_benchmark.py \
  --method Policy-as-Skill \
  --warmup 5 \
  --tasks 50 \
  --output result/edge_benchmark.json
```

Run the same command for the strongest comparison methods if the paper makes relative efficiency claims.

## Repetitions

For publication-quality latency, use multiple independent runs after device temperature reaches a stable range. Report the number of repetitions and the distribution; do not report only the fastest run.

## Network boundary

If Ollama runs on another machine or a cloud endpoint, the measurement is not a fully local edge-inference benchmark. Report the network topology and treat latency as end-to-end system latency. If the LLM runs directly on the edge device, state that explicitly.

## Power

`edge_benchmark.py` records power only when a supported counter is available. For stronger energy claims, use device-native telemetry or an external power meter and report the measurement method. Missing power counters are represented as missing data, never estimated.

## Interpretation

An edge feasibility claim can be supported by measured resource/latency data. It does not by itself establish production safety, legal compliance, or operational suitability. Those remain separate validity questions.
