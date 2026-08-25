from __future__ import annotations

import json
import os
import platform
import resource
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run(cmd: list[str], timeout: float = 2.0) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return (proc.stdout or proc.stderr or "").strip()
    except Exception:
        return ""


def _linux_memory() -> dict[str, Any]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            values[key] = int(raw.strip().split()[0])
        except (ValueError, IndexError):
            continue
    return {"mem_total_kib": values.get("MemTotal"), "mem_available_kib": values.get("MemAvailable")}


def network_snapshot() -> dict[str, Any]:
    """Host network counters for end-to-end bandwidth accounting when available."""
    path = Path("/proc/net/dev")
    if not path.exists():
        return {}
    interfaces = {}
    rx_total = tx_total = 0
    for line in path.read_text(errors="ignore").splitlines()[2:]:
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        parts = raw.split()
        if len(parts) < 16:
            continue
        try:
            rx, tx = int(parts[0]), int(parts[8])
        except ValueError:
            continue
        interface = name.strip()
        interfaces[interface] = {"rx_bytes": rx, "tx_bytes": tx}
        if interface != "lo":
            rx_total += rx
            tx_total += tx
    return {"network_rx_bytes": rx_total, "network_tx_bytes": tx_total, "network_interfaces": interfaces}


def power_supply_snapshot() -> dict[str, Any]:
    """Read battery/energy telemetry exposed through Linux power_supply sysfs."""
    root = Path("/sys/class/power_supply")
    if not root.exists():
        return {}
    supplies = []
    for supply in sorted(root.iterdir()):
        row: dict[str, Any] = {"name": supply.name}
        for field in ["type", "status", "capacity", "energy_now", "energy_full", "power_now", "charge_now", "charge_full", "voltage_now", "current_now"]:
            p = supply / field
            if p.exists():
                value = p.read_text(errors="ignore").strip()
                if value:
                    try:
                        row[field] = int(value)
                    except ValueError:
                        row[field] = value
        if len(row) > 1:
            supplies.append(row)
    return {"power_supplies": supplies} if supplies else {}


def _jetson_info() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in [Path("/proc/device-tree/model"), Path("/sys/firmware/devicetree/base/model")]:
        if p.exists():
            model = p.read_text(errors="ignore").replace("\x00", "").strip()
            if model:
                out["jetson_or_device_tree_model"] = model
                break
    if shutil.which("tegrastats"):
        sample = _run(["tegrastats", "--interval", "100", "--count", "1"], timeout=1.5)
        if sample:
            out["tegrastats_sample"] = sample
    return out


def _nvidia_info() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {}
    query = "name,driver_version,memory.total,memory.used,power.draw,utilization.gpu,utilization.memory,temperature.gpu"
    text = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 8:
            rows.append({"name": parts[0], "driver_version": parts[1], "memory_total_mib": parts[2], "memory_used_mib": parts[3], "power_draw_w": parts[4], "gpu_utilization_percent": parts[5], "memory_utilization_percent": parts[6], "temperature_c": parts[7]})
    return {"nvidia_gpus": rows} if rows else {}


def process_resource_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"max_rss_raw": usage.ru_maxrss, "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime, "minor_page_faults": usage.ru_minflt, "major_page_faults": usage.ru_majflt, "voluntary_context_switches": usage.ru_nvcsw, "involuntary_context_switches": usage.ru_nivcsw}


def collect_system_profile() -> dict[str, Any]:
    profile: dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version(), "cpu_count": os.cpu_count(), "resource": process_resource_snapshot()}
    profile.update(_linux_memory())
    profile.update(network_snapshot())
    profile.update(power_supply_snapshot())
    profile.update(_jetson_info())
    profile.update(_nvidia_info())
    return profile


def write_system_profile(path: Path) -> dict[str, Any]:
    payload = collect_system_profile()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
