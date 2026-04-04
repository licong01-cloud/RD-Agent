import os, sys, re
from pathlib import Path

os.environ.setdefault('MLFLOW_TRACKING_URI', str(Path.cwd() / 'mlruns'))
import qlib, yaml
conf = yaml.safe_load(open('conf.yaml'))
qi = conf.get('qlib_init', {})
qlib.init(provider_uri=qi.get('provider_uri'), region=qi.get('region'))
from qlib.workflow import R

experiments = R.list_experiments()
latest_recorder = None
for exp in experiments:
    for rid in R.list_recorders(experiment_name=exp):
        if rid:
            rec = R.get_recorder(recorder_id=rid, experiment_name=exp)
            et = rec.info.get("end_time")
            if et and (latest_recorder is None or et > latest_recorder.info["end_time"]):
                latest_recorder = rec

print("CWD:", os.getcwd())

log_candidates = list(Path.cwd().glob("*.log")) + list(Path.cwd().glob("*.txt"))
for name in ("stdout.log", "output.log", "run.log", "exp_output.log"):
    p = Path.cwd() / name
    if p.exists() and p not in log_candidates:
        log_candidates.append(p)

all_text = ""
for lf in log_candidates:
    try:
        all_text += lf.read_text(encoding="utf-8", errors="replace") + "\n"
    except:
        pass

print("Files:", [p.name for p in log_candidates], "len=", len(all_text))

train_losses, val_losses = [], []
# Pattern 1
for m in re.finditer(r"[Ee]poch\s*(\d+).*?train[_ ]?loss[=:\s]+([\d.]+).*?val(?:id)?[_ ]?loss[=:\s]+([\d.]+)", all_text):
    train_losses.append(float(m.group(2))); val_losses.append(float(m.group(3)))
print("P1:", len(train_losses))
# Pattern 4
if not train_losses:
    for m in re.finditer(r"[Ee]poch\s*(\d+):\s*train\s+([\d.eE+-]+),?\s*valid\s+([\d.eE+-]+)", all_text):
        train_losses.append(float(m.group(2))); val_losses.append(float(m.group(3)))
    print("P4:", len(train_losses))
# Pattern 5
if not train_losses:
    for m in re.finditer(r"^train\s+([\d.eE+-]+),?\s*valid\s+([\d.eE+-]+)", all_text, re.MULTILINE):
        train_losses.append(float(m.group(1))); val_losses.append(float(m.group(2)))
    print("P5:", len(train_losses))
# Pattern 6
if not train_losses and not val_losses:
    for m in re.finditer(r"[Ee]poch\s*(\d+):\s*train\s+N/A.*?valid\s+([\d.eE+-]+)", all_text):
        val_losses.append(float(m.group(2)))
    print("P6 val:", len(val_losses))

print("FINAL: train=", len(train_losses), "val=", len(val_losses))
