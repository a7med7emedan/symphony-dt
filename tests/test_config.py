from pathlib import Path
from symphony_dt.config import load_config


def test_load_config_demo(tmp_path: Path) -> None:
    cfg_text = '''
run:
  project: "x"
  seed: 123
  outputs_root: "outputs"
io:
  input_dir: "data"
  cohort: "PDLIKE"
  baseline_filename_pdlike: "a.csv"
  clinic_filename_pdlike: "b.csv"
  baseline_filename_all: "c.csv"
  clinic_filename_all: "d.csv"
  omics_filename: "o.csv"
'''
    p = tmp_path / "cfg.yaml"
    p.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.run.seed == 123
    assert cfg.io.cohort == "PDLIKE"
