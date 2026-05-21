# Scripts

Organized by analysis pipeline stage. Numeric prefix = execution order.

| Folder | Purpose |
|--------|---------|
| `01_import/` | Read wearables, EHR, clinical tables; merge on `patient_id` |
| `02_preprocess/` | Filtering, imputation, temporal sync, quality control |
| `03_features/` | Behavioral and physiologic trajectories; model variables |
| `04_analysis/` | Association with progression, host biologic state, prediction |
| `05_figures/` | Manuscript figures (typical output: `results/figures/`) |

Convention: one main script per analysis (e.g. `04_progression_models.R`) and helper functions in the same directory or in `scripts/utils/` as the project grows.
