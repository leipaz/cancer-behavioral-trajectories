# Remote monitoring of behavioral–physiologic trajectories predict progression and reflect biologic host state in metastatic cancer in women

This repository contains the code, analysis-ready data, and workflows used in the study on remote monitoring of behavioral and physiologic trajectories in patients with metastatic cancer.

## Authors

Leire Paz<sup>1,\*</sup>, Leonardo Garma<sup>2,\*</sup>, Sonia Pernas<sup>3,4</sup>, Juan Antonio Guerra<sup>5</sup>, Rosario García-Campelo<sup>6</sup>, Jacobo Rogado<sup>7</sup>, David Vicente-Baz<sup>8</sup>, Josefa Terrasa<sup>9</sup>, Begoña Bermejo<sup>10</sup>, Ruth Vera<sup>11</sup>, Santiago González Santiago<sup>12</sup>, Sandra Gallach<sup>13,14,15</sup>, Berta Nasarre<sup>5</sup>, Bartomeu Fullana<sup>3</sup>, Desirée Jiménez<sup>2</sup>, Cristina Reboredo-Rendo<sup>6</sup>, Oscar Padilla<sup>3</sup>, Rodrigo Oliver<sup>1</sup>, Cristina Simarro<sup>8</sup>, Antonia Perelló<sup>9</sup>, Berta Hernández-Martín<sup>10</sup>, Aída Morillas<sup>2</sup>, Silvana Mourón<sup>2</sup>, María J. Bueno<sup>2</sup>, Antonio Lopez<sup>2</sup>, Pablo Martínez Olmos<sup>1</sup>, Silvia Calabuig<sup>13,14,15</sup>, Ramon Colomer<sup>7,16</sup>, Antonio Artes<sup>1</sup>, Miguel Quintela-Fandino<sup>2,7</sup>

## Affiliations

1. Communications and Signal Theory, Escuela Politécnica Superior, Universidad Carlos III, Leganés (Madrid), Spain
2. Breast Cancer Clinical Research Unit, CNIO – Spanish National Cancer Research Center, Madrid, Spain
3. Breast Cancer Unit, Department of Medical Oncology, Institut Català d’Oncologia, L’Hospitalet de Llobregat (Barcelona), Spain
4. Institut d’Investigació Biomèdica de Bellvitge (IDIBELL), L’Hospitalet de Llobregat (Barcelona), Spain
5. Medical Oncology, Hospital Universitario de Fuenlabrada, Fuenlabrada (Madrid), Spain
6. Medical Oncology, Complejo Hospitalario Universitario de A Coruña – CHUAC, A Coruña, Spain
7. Medical Oncology, Hospital Universitario de La Princesa, Madrid, Spain
8. Medical Oncology, Hospital Universitario Virgen de la Macarena, Sevilla, Spain
9. Medical Oncology, Hospital Universitari Son Espases, Palma, Spain
10. Medical Oncology, Hospital Clínico Universitario, Valencia, Spain
11. Medical Oncology, Hospital Universitario de Navarra, Navarra, Spain
12. Medical Oncology, Hospital San Pedro de Alcántara – Complejo Hospitalario Universitario de Cáceres, Cáceres, Spain
13. Department of Pathology, Universitat de València, Valencia, Spain
14. Centro de Investigación Biomédica en Red Cáncer (CIBERONC), Madrid, Spain
15. Molecular Oncology Laboratory, Fundación Investigación Hospital General Universitario de Valencia, Valencia, Spain
16. Roche Endowed Chair of Precision and Personalized Medicine, Universidad Autónoma de Madrid, Madrid, Spain

## Repository structure

```
.
├── data/                 # Processed, anonymized data
│   ├── processed/        # Analysis-ready tables
│   │   ├── behavioral/   # Behavioral trajectories (activity, sleep, etc.)
│   │   ├── physiologic/  # Physiologic trajectories (HR, HRV, etc.)
│   │   └── clinical/     # Outcomes and clinical variables
│   └── metadata/         # Variable dictionaries, cohort definitions, code maps
├── scripts/              # Processing, analysis, and modeling code
│   ├── utils/            # Shared code across methods
│   ├── 01_preprocess/    # Per method: lda/, vq-vae/
│   ├── 02_univariate_analysis/
│   ├── 03_analysis/
│   └── 04_figures/
└── results/              # Reproducible outputs (not raw data)
    ├── lda/              # figures/, tables/, models/
    └── vq-vae/
```

See `data/README.md`, `scripts/README.md`, and `results/README.md` for folder-level detail.

---

## Analysis guide

**Environment:** `python3 -m venv env && source env/bin/activate && pip install -r requirements.txt`

### Run pipeline

```bash
python scripts/01_preprocess/lda/preprocess_daily_summaries.py
python scripts/02_univariate_analysis/lda/run_univariate_analysis.py
python scripts/03_analysis/lda/run_lda_pipeline.py              # --skip-train if model exists
python scripts/03_analysis/lda/run_decode_profiles.py
python scripts/03_analysis/lda/run_monthly_topics_heatmap.py    # --rebuild-table to regenerate monthly CSV
```

### Generated outputs (summary)

| Step | Command | Main figures / tables |
|------|---------|------------------------|
| Preprocess | `01_preprocess/lda/preprocess_daily_summaries.py` | Filtered daily summary under `data/raw/` |
| Univariate | `02_univariate_analysis/lda/run_univariate_analysis.py` | `results/univariate/figures/*`, `tables/variable_summary_full_stats_EPD.csv` |
| LDA | `03_analysis/lda/run_lda_pipeline.py` | `results/lda/figures/lda_top_terms_grid.png`, `topics_by_event.png`, `avg_topic_distribution_by_event.png`; `tables/lda_topics_top10.csv`; model under `data/processed/lda/` |
| Decode | `03_analysis/lda/run_decode_profiles.py` | `results/lda/figures/decoded_topic_plots/*.svg`, optional boxplot |
| Monthly heatmap | `03_analysis/lda/run_monthly_topics_heatmap.py` | `results/lda/figures/predominant_topics_per_month.png` (`.svg`) |

Versioned **figures** → `results/`; intermediate **models and CSVs** → `data/processed/` (gitignored). Details: [`scripts/README.md`](scripts/README.md).

---

## Data and privacy

Raw patient data are not published. Only **processed, anonymized** datasets that comply with applicable regulations and informed consent are included in this repository. Raw data remain outside version control (see `.gitignore`).

## License

Code is under the [Apache License 2.0](LICENSE). Data may have additional use conditions; see `data/README.md`.

## Citation

If you use this material, please cite the manuscript (bibliographic reference pending publication).
