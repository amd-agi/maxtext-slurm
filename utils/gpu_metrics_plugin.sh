#!/usr/bin/env bash
# gpu_metrics_plugin.sh — Prometheus metrics plugin: GPU hardware (AMD + NVIDIA).
#
# AMD:    Collects per-GPU temperature, power, clocks, VRAM, and RAS errors
#         via sysfs (zero subprocess overhead — plain file reads from
#         /sys/class/hwmon, /sys/class/drm, and /sys/bus/pci/drivers/amdgpu).
# NVIDIA: Collects the same metrics via a single `nvidia-smi --query-gpu` call.
#
# Always collects PCIe AER error counters for both vendors via sysfs.
# AMD-specific: XGMI/UMC/GFX/MMHUB/SDMA RAS counters via sysfs.
#
# Called by metrics_exporter.sh — outputs Prometheus text to stdout.
#
# Metrics (all prefixed hw_ for grouping in Prometheus UI):
#
#   --- Common (AMD + NVIDIA) ---
#   hw_gpu_temperature_celsius{gpu,host}           Junction temperature (°C)
#   hw_gpu_power_watts{gpu,host}                   Current power draw (W)
#   hw_gpu_clock_mhz{gpu,host,type=sclk|mclk}     Core / memory clock (MHz)
#   hw_gpu_vram_used_bytes{gpu,host}                VRAM currently used (bytes)
#   hw_gpu_vram_total_bytes{gpu,host}               VRAM total capacity (bytes)
#   hw_gpu_pcie_correctable_total{gpu,host}          PCIe correctable AER errors
#   hw_gpu_pcie_nonfatal_total{gpu,host}             PCIe non-fatal AER errors
#   hw_gpu_pcie_fatal_total{gpu,host}                PCIe fatal AER errors
#
#   --- NVIDIA only ---
#   hw_gpu_utilization_pct{gpu,host}               GPU compute utilization (%)
#   hw_gpu_ras_ecc_ce_total{gpu,host}              Correctable ECC errors
#   hw_gpu_ras_ecc_ue_total{gpu,host}              Uncorrectable ECC errors
#
#   --- AMD only (fine-grained RAS via sysfs) ---
#   hw_gpu_ras_umc_{ue,ce}_total{gpu,host}         HBM memory ECC errors
#   hw_gpu_ras_xgmi_{ue,ce}_total{gpu,host}        XGMI/WAFL link errors
#   hw_gpu_ras_gfx_{ue,ce}_total{gpu,host}         Compute engine errors
#   hw_gpu_ras_mmhub_{ue,ce}_total{gpu,host}       Memory hub errors
#   hw_gpu_ras_sdma_{ue,ce}_total{gpu,host}        SDMA engine errors

HOSTNAME_SHORT="${1:?Usage: gpu_metrics_plugin.sh <hostname>}"

python3 - "$HOSTNAME_SHORT" <<'PYEOF'
import os, re, sys
from pathlib import Path

hostname = sys.argv[1]
lines = []

def add(line):
    lines.append(line)

def read_int(path):
    """Read a single integer from a sysfs file, return None on failure."""
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None

# =========================================================================
# Discover AMD GPUs via sysfs hwmon (no subprocess, no driver ioctls)
# =========================================================================
# Each amdgpu device exposes a hwmon directory with temperature, power,
# and clock files.  We sort by PCI bus address to assign GPU indices 0..N
# matching the order ROCm uses.
# On NVIDIA systems this returns an empty list and the NVIDIA path below
# takes over (nvidia-smi subprocess).

def discover_amd_gpus():
    """Return list of (gpu_index, hwmon_path) sorted by PCI bus address."""
    hwmon_root = Path('/sys/class/hwmon')
    if not hwmon_root.exists():
        return []

    gpus = []  # (pci_addr, hwmon_path)
    for hwdir in hwmon_root.iterdir():
        name_file = hwdir / 'name'
        if not name_file.exists():
            continue
        try:
            name = name_file.read_text().strip()
        except Exception:
            continue
        if name != 'amdgpu':
            continue

        # Resolve PCI bus address from the device symlink
        dev_path = (hwdir / 'device').resolve()
        pci_match = re.findall(r'[0-9a-f]+:[0-9a-f]+:[0-9a-f]+\.[0-9a-f]+', str(dev_path))
        pci_addr = pci_match[-1] if pci_match else str(hwdir)
        gpus.append((pci_addr, str(hwdir)))

    # Sort by PCI address → GPU index 0, 1, 2, ...
    gpus.sort(key=lambda x: x[0])
    return [(idx, path) for idx, (_, path) in enumerate(gpus)]

# =========================================================================
# GPU: temperature, power, clocks, VRAM  (sysfs reads — no subprocess)
# =========================================================================
add('# HELP hw_gpu_temperature_celsius GPU temperature in Celsius.')
add('# TYPE hw_gpu_temperature_celsius gauge')
add('# HELP hw_gpu_power_watts GPU power draw in Watts.')
add('# TYPE hw_gpu_power_watts gauge')
add('# HELP hw_gpu_clock_mhz GPU clock speed in MHz.')
add('# TYPE hw_gpu_clock_mhz gauge')
add('# HELP hw_gpu_vram_used_bytes GPU VRAM currently used in bytes.')
add('# TYPE hw_gpu_vram_used_bytes gauge')
add('# HELP hw_gpu_vram_total_bytes GPU VRAM total capacity in bytes.')
add('# TYPE hw_gpu_vram_total_bytes gauge')
add('# HELP hw_gpu_utilization_pct GPU compute utilization percentage.')
add('# TYPE hw_gpu_utilization_pct gauge')
add('# HELP hw_gpu_ras_ecc_ce_total GPU correctable ECC errors (NVIDIA).')
add('# TYPE hw_gpu_ras_ecc_ce_total counter')
add('# HELP hw_gpu_ras_ecc_ue_total GPU uncorrectable ECC errors (NVIDIA).')
add('# TYPE hw_gpu_ras_ecc_ue_total counter')
# NOTE: AMD sysfs gpu_busy_percent returns 0 on MI355 OAM with current
# ROCm drivers; on NVIDIA, utilization is collected via nvidia-smi above.

amd_gpus = discover_amd_gpus()

if amd_gpus:
    for gpu_id, hwpath in amd_gpus:
        hw = Path(hwpath)
        lb = f'gpu="{gpu_id}",host="{hostname}"'

        # Temperature: prefer junction (temp2), fall back to mem (temp3),
        # then try temp1.  Values are in millidegrees C.
        for temp_file in ('temp2_input', 'temp3_input', 'temp1_input'):
            val = read_int(hw / temp_file)
            if val is not None and val > 0:
                add(f'hw_gpu_temperature_celsius{{{lb}}} {val / 1000.0:.1f}')
                break

        # Power: power1_input is in microwatts.
        pval = read_int(hw / 'power1_input')
        if pval is not None:
            add(f'hw_gpu_power_watts{{{lb}}} {pval / 1e6:.1f}')

        # Clocks: freq1_input (sclk) and freq2_input (mclk) are in Hz.
        sclk = read_int(hw / 'freq1_input')
        if sclk is not None:
            add(f'hw_gpu_clock_mhz{{{lb},type="sclk"}} {sclk / 1e6:.0f}')
        mclk = read_int(hw / 'freq2_input')
        if mclk is not None:
            add(f'hw_gpu_clock_mhz{{{lb},type="mclk"}} {mclk / 1e6:.0f}')

        # VRAM: mem_info_vram_used / _total are in the device directory (bytes).
        dev = (hw / 'device').resolve()
        vram_used = read_int(dev / 'mem_info_vram_used')
        if vram_used is not None:
            add(f'hw_gpu_vram_used_bytes{{{lb}}} {vram_used}')
        vram_total = read_int(dev / 'mem_info_vram_total')
        if vram_total is not None:
            add(f'hw_gpu_vram_total_bytes{{{lb}}} {vram_total}')

else:
    # =====================================================================
    # Discover NVIDIA GPUs via nvidia-smi (fallback when no AMD GPUs found)
    # =====================================================================
    nvidia_gpus = []
    try:
        import subprocess as _sp
        _nv = _sp.run(
            ['nvidia-smi',
             '--query-gpu=index,temperature.gpu,power.draw,clocks.sm,clocks.mem,'
             'memory.used,memory.total,utilization.gpu,'
             'ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if _nv.returncode == 0 and _nv.stdout.strip():
            for _line in _nv.stdout.strip().splitlines():
                parts = [p.strip() for p in _line.split(',')]
                if len(parts) < 8:
                    continue
                gpu_id = parts[0]
                nvidia_gpus.append(gpu_id)
                lb = f'gpu="{gpu_id}",host="{hostname}"'

                def _safe_float(s):
                    try:
                        return float(s) if s not in ('[N/A]', 'N/A', '[Not Supported]', '') else None
                    except ValueError:
                        return None

                temp = _safe_float(parts[1])
                if temp is not None:
                    add(f'hw_gpu_temperature_celsius{{{lb}}} {temp:.1f}')

                power = _safe_float(parts[2])
                if power is not None:
                    add(f'hw_gpu_power_watts{{{lb}}} {power:.1f}')

                sclk = _safe_float(parts[3])
                if sclk is not None:
                    add(f'hw_gpu_clock_mhz{{{lb},type="sclk"}} {int(sclk)}')
                mclk = _safe_float(parts[4])
                if mclk is not None:
                    add(f'hw_gpu_clock_mhz{{{lb},type="mclk"}} {int(mclk)}')

                # nvidia-smi reports memory in MiB
                vram_used = _safe_float(parts[5])
                if vram_used is not None:
                    add(f'hw_gpu_vram_used_bytes{{{lb}}} {int(vram_used * 1048576)}')
                vram_total = _safe_float(parts[6])
                if vram_total is not None:
                    add(f'hw_gpu_vram_total_bytes{{{lb}}} {int(vram_total * 1048576)}')

                util = _safe_float(parts[7])
                if util is not None:
                    add(f'hw_gpu_utilization_pct{{{lb}}} {int(util)}')

                if len(parts) >= 10:
                    ecc_ce = _safe_float(parts[8])
                    if ecc_ce is not None:
                        add(f'hw_gpu_ras_ecc_ce_total{{{lb}}} {int(ecc_ce)}')
                    ecc_ue = _safe_float(parts[9])
                    if ecc_ue is not None:
                        add(f'hw_gpu_ras_ecc_ue_total{{{lb}}} {int(ecc_ue)}')

    except FileNotFoundError:
        pass  # nvidia-smi not available
    except Exception as e:
        print(f'[gpu_plugin] nvidia-smi: {e}', file=sys.stderr)

    if not nvidia_gpus:
        print('[gpu_plugin] No AMD or NVIDIA GPUs found', file=sys.stderr)

# =========================================================================
# GPU RAS error counters  (sysfs — AMD only, zero overhead)
# =========================================================================
# The amdgpu driver exposes per-block RAS counters via sysfs.
# NVIDIA ECC errors are already collected above via nvidia-smi.
# These AMD-specific RAS blocks provide finer granularity:
#   aca_umc        — HBM memory ECC  (equivalent to check_ecc.sh / rocm-smi)
#   aca_xgmi_wafl  — XGMI/WAFL inter-GPU link errors
#   aca_gfx        — Compute engine (shader) errors
#   aca_mmhub      — Memory hub / VRAM controller errors
#   aca_sdma       — SDMA (DMA copy engine) errors
#
# Each block reports ue (uncorrectable) and ce (correctable) counts.
# A non-zero ue in any block means the GPU has a hardware fault and the
# node should be drained.

RAS_BLOCKS = [
    ('aca_umc',       'umc'),
    ('aca_xgmi_wafl', 'xgmi'),
    ('aca_gfx',       'gfx'),
    ('aca_mmhub',     'mmhub'),
    ('aca_sdma',      'sdma'),
]

for _, short in RAS_BLOCKS:
    add(f'# HELP hw_gpu_ras_{short}_ue_total GPU {short.upper()} uncorrectable RAS errors.')
    add(f'# TYPE hw_gpu_ras_{short}_ue_total counter')
    add(f'# HELP hw_gpu_ras_{short}_ce_total GPU {short.upper()} correctable RAS errors.')
    add(f'# TYPE hw_gpu_ras_{short}_ce_total counter')

# Build PCI → GPU index mapping once (reuse discover_amd_gpus result).
pci_to_gpu = {}
for idx, hwpath in amd_gpus:
    hw_pci = re.findall(r'[0-9a-f]+:[0-9a-f]+:[0-9a-f]+\.[0-9a-f]+',
                        str(Path(hwpath, 'device').resolve()))
    if hw_pci:
        pci_to_gpu[hw_pci[-1]] = idx

try:
    for dev in sorted(Path('/sys/bus/pci/drivers/amdgpu').iterdir()):
        if not dev.name.startswith('0000:'):
            continue
        gpu_id = pci_to_gpu.get(dev.name)
        if gpu_id is None:
            continue

        lb = f'gpu="{gpu_id}",host="{hostname}"'
        ras_dir = dev / 'ras'
        for sysfs_name, short in RAS_BLOCKS:
            ras_file = ras_dir / sysfs_name
            if not ras_file.exists():
                continue
            text = ras_file.read_text()
            for line in text.splitlines():
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if key == 'ue':
                        add(f'hw_gpu_ras_{short}_ue_total{{{lb}}} {val}')
                    elif key == 'ce':
                        add(f'hw_gpu_ras_{short}_ce_total{{{lb}}} {val}')
except Exception as e:
    print(f'[gpu_plugin] GPU RAS: {e}', file=sys.stderr)

# =========================================================================
# PCIe AER error counters  (sysfs — Advanced Error Reporting)
# =========================================================================
# PCIe AER errors are a major source of GPU disconnects and hangs.  The
# kernel exposes per-device totals via:
#   aer_dev_correctable  — recoverable (retried) errors
#   aer_dev_nonfatal     — uncorrectable but not fatal (device usable)
#   aer_dev_fatal        — uncorrectable fatal (device unusable)
#
# Each file contains named counters and a TOTAL_ERR_* line.  We export
# the totals.  A non-zero fatal count means the GPU's PCIe link failed.
# A spike in correctable errors often precedes a fatal event.

add('# HELP hw_gpu_pcie_correctable_total PCIe correctable AER errors.')
add('# TYPE hw_gpu_pcie_correctable_total counter')
add('# HELP hw_gpu_pcie_nonfatal_total PCIe non-fatal uncorrectable AER errors.')
add('# TYPE hw_gpu_pcie_nonfatal_total counter')
add('# HELP hw_gpu_pcie_fatal_total PCIe fatal uncorrectable AER errors.')
add('# TYPE hw_gpu_pcie_fatal_total counter')

def parse_aer_total(path, prefix='TOTAL_ERR'):
    """Parse a PCIe AER sysfs file and return the total error count."""
    try:
        text = Path(path).read_text()
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].startswith(prefix):
                return int(parts[1])
    except Exception:
        pass
    return None

# Scan both AMD and NVIDIA PCI drivers for AER counters.
# pci_to_gpu is populated by discover_amd_gpus() above; for NVIDIA we build
# a separate mapping from the nvidia PCI driver directory.
_nv_pci_to_gpu = {}
try:
    nv_driver = Path('/sys/bus/pci/drivers/nvidia')
    if nv_driver.exists():
        for idx, dev in enumerate(sorted(
                d for d in nv_driver.iterdir() if d.name.startswith('0000:'))):
            _nv_pci_to_gpu[dev.name] = idx
except Exception:
    pass

for driver_path, pci_map in [
    (Path('/sys/bus/pci/drivers/amdgpu'), pci_to_gpu),
    (Path('/sys/bus/pci/drivers/nvidia'), _nv_pci_to_gpu),
]:
    if not driver_path.exists():
        continue
    try:
        for dev in sorted(driver_path.iterdir()):
            if not dev.name.startswith('0000:'):
                continue
            gpu_id = pci_map.get(dev.name)
            if gpu_id is None:
                continue
            lb = f'gpu="{gpu_id}",host="{hostname}"'

            for sysfs_name, metric, prefix in [
                ('aer_dev_correctable', 'hw_gpu_pcie_correctable_total', 'TOTAL_ERR_COR'),
                ('aer_dev_nonfatal',    'hw_gpu_pcie_nonfatal_total',    'TOTAL_ERR_NONFATAL'),
                ('aer_dev_fatal',       'hw_gpu_pcie_fatal_total',       'TOTAL_ERR_FATAL'),
            ]:
                aer_file = dev / sysfs_name
                if aer_file.exists():
                    val = parse_aer_total(str(aer_file), prefix)
                    if val is not None:
                        add(f'{metric}{{{lb}}} {val}')
    except Exception as e:
        print(f'[gpu_plugin] PCIe AER ({driver_path.name}): {e}', file=sys.stderr)

# Output
print('\n'.join(lines))
PYEOF
