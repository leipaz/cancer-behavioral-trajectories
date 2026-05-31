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
├── data/                 # Data (raw/ and processed/ contents are gitignored)
│   ├── raw/              # Source daily summaries and clinical tables
│   ├── processed/        # Intermediate analysis tables
│   ├── daily_summaries/  # Per-patient daily summaries (model input)
│   ├── output_vq_vae/    # VQ-VAE decoded embeddings
│   └── output_lda/       # LDA outputs
├── models/               # Trained models
│   ├── lda/              # LDA model + dictionary
│   └── vq-vae/
├── notebooks/            # Analysis notebooks (see pipeline below)
├── scripts/              # Processing, analysis, and modeling code
│   ├── vqvae/            # VQ-VAE package
│   ├── 01_preprocess/
│   ├── 02_univariate_analysis/
│   └── 03_analysis/      # lda/, vq-vae/
└── results/              # Reproducible outputs (figures and tables)
    ├── lda/              # figures/, tables/
    └── univariate/       # tables/
```

See `data/README.md`, `scripts/README.md`, and `results/README.md` for folder-level detail.

---

## Analysis guide

**Environment:** `python3 -m venv env && source env/bin/activate && pip install -r requirements.txt`

### Run pipeline

**Notebooks** (exploratory workflow, run in this order):

1. **`daily_to_profiles.ipynb`** — Applies the trained **VQ-VAE** to each patient's daily summaries to obtain a sequence of discrete **daily day-type profiles** (embedding IDs). *(patient-level merge step still pending.)*
2. **`profiles_to_dcabp.ipynb`** — Loads the trained **LDA** model and dictionary, builds the bag-of-words corpus from the day-type sequences, visualizes the topics (pyLDAvis + top-terms grid), exports the top-10 day-types per topic (`lda_topics_top10.csv`), and compares topic distributions between progression (PD) and non-PD patients.
3. **`profile_decodification.ipynb`** — Decodes the top-10 day-type profiles of each LDA topic back into **behavioral features** (via the VQ-VAE) and plots the feature profile of each pattern (raw and z-scored).
4. **`lda_dcabp_evolution.ipynb`** — Assigns the **predominant topic per 30-day block** for every patient and draws the per-month heatmaps (6 topics and favorable/unfavorable hypertopics), marking the real progression-event time.

The equivalent reproducible scripts and the generated figures/tables are documented in the folder-level READMEs: [`scripts/README.md`](scripts/README.md) and [`results/README.md`](results/README.md).

---

## Data and privacy

Raw patient data are not published. Only **processed, anonymized** datasets that comply with applicable regulations and informed consent are included in this repository. 

## License

Code is under the [Apache License 2.0](LICENSE). Data may have additional use conditions; see `data/README.md`.

## Citation

If you use this material, please cite the manuscript (bibliographic reference pending publication).
