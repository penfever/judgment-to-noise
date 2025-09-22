# Judgment‑to‑Noise: Analysis Toolkit

Tools to analyze LLM judge outputs with a focus on schematic adherence, psychometric reliability, and uncertainty‑aware leaderboards. This README centers on the new analysis workflow (how to run it and generate visualizations). A smaller Arena‑Hard‑Auto section remains below for context.

## What’s New
- Reliability‑aware CI updates using factor communalities and per‑factor reliability.
- Multiple bootstrap options: standard, residual, and Bayesian hierarchical.
- End‑to‑end pipelines: fixed (`run_analysis.sh`) and flexible (`run_analysis_flexible.sh`).
- Visualization scripts for reliability, factor loadings, and quick image grids.

## Quick Start
- Environment
  - cd `judgment-to-noise`
  - python -m venv `.venv` && source `.venv/bin/activate`
  - pip install -r `requirements.txt` [-r `requirements-optional.txt`]

- Inputs
  - Place JSONL judgment files in an input directory. Each JSONL should include `question_id`, `model`, and `games` with per‑metric outcomes.

- Flexible pipeline (recommended)
  - bash `run_analysis_flexible.sh` --input-dir `<path/to/jsonl_dir>` --output-dir `<path/to/results_root>` --judge-name `<judge_id>`
  - Optional: `--suffix <tag>` to suffix outputs; `--skip-gen-subscores` if inputs are already processed.

Outputs land under `<output-dir>/tables/` with subfolders for factor scores, reliability, correlations, and more.

## Core CLI Entrypoints
- `show_result.py` — build leaderboards and confidence intervals.
  - Original CIs:
    - python `show_result.py` --output --judge-name `<JUDGE>` --judgment-dir `<INPUT_DIR>` --target-metric `score`
  - Bayesian bootstrap + updated CIs:
    - python `show_result.py` --output --judge-name `<JUDGE>` --judgment-dir `<INPUT_DIR>` --target-metric `score` --bootstrap-method `bayesian` --communalities-file `<tables/factor_analysis/factor_communalities.csv>` --reliability-file `<tables/factor_reliability/factor_reliability_metrics.csv>`

- `factor_reliability_improved.py` — compute Cronbach’s alpha, cross‑loading ratios, HTMT, and an aggregate reliability score.
  - Example:
    - python `factor_reliability_improved.py` `<INPUT_DIR>` --output-dir `<tables/factor_reliability>` --debug --skip-bootstrap

- `factor_analysis.py` — factor loadings/communalities; can also analyze per‑question structure.
  - Examples:
    - python `factor_analysis.py` `<INPUT_DIR>` --analyze-questions
    - python `factor_analysis.py` `<tables/factor_scores_updated_cis>` --output-dir `<tables/rankings_factor_analysis/updated_cis>`

- `get_corrs.py` — correlations and heatmaps among factor scores.
  - Example:
    - python `get_corrs.py` `<tables/factor_scores_original_cis>`

## Pipelines
- `run_analysis_flexible.sh` — configurable input/output; best for running on arbitrary judgment sets. See inline `--help` for options.
- `run_analysis.sh` — fixed paths for a standardized local layout; prefer the flexible script unless you already use the fixed structure.

## Visualizations
- Psychometric reliability overview: `scripts/visualize_psychometric_reliability.py`
  - Example:
    - python `scripts/visualize_psychometric_reliability.py` --input-path `<InDepthAnalysis root>` --output-dir `figures/psychometric_reliability`

- Factor image grids: `scripts/create_image_grid.py`
  - Example:
    - python `scripts/create_image_grid.py` --input-dir `figures/factor_loadings/` --output-dir `figures/grids` --columns 3 --title "Factor Loadings"
  - Also see `scripts/visualize_factors.sh` and `scripts/README.md`.

- Additional viz helpers: `scripts/ablate_pipeline_visualization.py`, `scripts/ablate_factors_visualization.py`, `scripts/average_setting1_correlations.py`, `scripts/judge_explainability_pies.py`.

## Testing
- Run tests in the toolkit: `cd judgment-to-noise && pytest -q`

## Data and Outputs
- Input judgments: put JSONL under a directory you pass to `--input-dir`.
- Tables: exported CSVs under `tables/` (factor scores, reliability, correlations, rankings analysis).
- Figures: saved under `figures/` (reliability plots, heatmaps, grids).

## Arena‑Hard‑Auto (Context)
This toolkit originated from Arena‑Hard‑Auto automation, and those utilities still exist (answer/judgment generation, basic result display, and QA browser). If you need that workflow:
- Configure models in `config/gen_answer_config.yaml` and `config/judge_config.yaml`.
- Generate answers: `python gen_answer.py`
- Generate judgments: `python gen_judgment.py`
- Browse QA: `python qa_browser.py --share`
- Paper: https://arxiv.org/abs/2406.11939
- Blog: https://lmsys.org/blog/2024-05-17-category-hard/

## Citation
If this toolkit helps your work, please cite our work (citation forthcoming) as well as Arena‑Hard and Chatbot Arena as the foundations of the data and evaluation pipeline.

```
@misc{li2024crowdsourceddatahighqualitybenchmarks,
  title={From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline},
  author={Tianle Li and Wei-Lin Chiang and Evan Frick and Lisa Dunlap and Tianhao Wu and Banghua Zhu and Joseph E. Gonzalez and Ion Stoica},
  year={2024},
  eprint={2406.11939},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
@misc{chiang2024chatbot},
  title={Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference},
  author={Wei-Lin Chiang and Lianmin Zheng and Ying Sheng and Anastasios Nikolas Angelopoulos and Tianle Li and Dacheng Li and Hao Zhang and Banghua Zhu and Michael Jordan and Joseph E. Gonzalez and Ion Stoica},
  year={2024},
  eprint={2403.04132},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```
