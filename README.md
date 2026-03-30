# Hybrid Entity Resolution for Historical Records

A research-oriented project that combines Large Language Model (LLM) reasoning with embedding-based clustering for robust entity resolution on noisy historical records.

## What This Repository Contains

- End-to-end hybrid ER pipeline code (`z_Taha_Code/hm_taha_modules`, `z_Taha_Code/run_hm_taha.py`, `z_Taha_Code/HM_Taha.ipynb`)
- Research framing and method notes (`z_Taha_Code/research_proposal.txt`, `z_Taha_Code/documentation.txt`)
- Dataset parameter and truth files (`z_Taha_Code/Data files`)
- Results organized by dataset under `z_Taha_Code/Results/`
- Included research paper PDF in `z_Taha_Code/`

## Pipeline Overview

1. Parse noisy raw strings into structured name/address fields (LLM + rule-assisted)
2. Generate semantic embeddings and perform clustering
3. Refine and merge clusters with deterministic rules and LLM fallback
4. Produce household-level views and movement analysis
5. Evaluate against truth files (precision/recall/F1 and clustering metrics)

## Repository Structure

- `z_Taha_Code/HM_Taha.ipynb` - notebook implementation
- `z_Taha_Code/hm_taha_modules/` - modularized pipeline cells/scripts
- `z_Taha_Code/Data files/` - input datasets and truth files
- `z_Taha_Code/Results/` - per-dataset outputs (`Results_S10PX`, `Results_S11PX`, ...)
- `z_Taha_Code/logs/` - run logs

## Quick Start

1. Create a Python environment (3.10+ recommended)
2. Install dependencies used by the scripts/notebook (e.g., OpenAI SDK, pandas, numpy, scikit-learn, sentence-transformers)
3. Set `OPENAI_API_KEY` in your environment or in `openai.env` (kept out of git)
4. Run notebook `z_Taha_Code/HM_Taha.ipynb` or script `z_Taha_Code/run_hm_taha.py`

## Notes on Reproducibility

- This repository includes intermediate and final output files for multiple datasets.
- Runtime logs and module output captures are retained for traceability.
- Some result folders may contain repeated experiment variants (e.g., `R1`..`R6`).

## Citation

If you use this repository in academic work, please cite using the metadata in `CITATION.cff`.
