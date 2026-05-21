# Remote monitoring of behavioral–physiologic trajectories predict progression and reflect biologic host state in metastatic cancer in women

Este repositorio contiene el código, los datos analizados y los flujos de trabajo utilizados en la investigación sobre el monitoreo remoto de trayectorias conductuales y fisiológicas en pacientes con cáncer metastásico en mujeres.

## Autores

<sup>\*</sup> LP y LG contribuyeron por igual a este manuscrito.

Leire Paz<sup>1,\*</sup>, Leonardo Garma<sup>2,\*</sup>, Sonia Pernas<sup>3,4</sup>, Juan Antonio Guerra<sup>5</sup>, Rosario García-Campelo<sup>6</sup>, Jacobo Rogado<sup>7</sup>, David Vicente-Baz<sup>8</sup>, Josefa Terrasa<sup>9</sup>, Begoña Bermejo<sup>10</sup>, Ruth Vera<sup>11</sup>, Santiago González Santiago<sup>12</sup>, Sandra Gallach<sup>13,14,15</sup>, Berta Nasarre<sup>5</sup>, Bartomeu Fullana<sup>3</sup>, Desirée Jiménez<sup>2</sup>, Cristina Reboredo-Rendo<sup>6</sup>, Oscar Padilla<sup>3</sup>, Rodrigo Oliver<sup>1</sup>, Cristina Simarro<sup>8</sup>, Antonia Perelló<sup>9</sup>, Berta Hernández-Martín<sup>10</sup>, Aída Morillas<sup>2</sup>, Silvana Mourón<sup>2</sup>, María J. Bueno<sup>2</sup>, Antonio Lopez<sup>2</sup>, Pablo Martínez Olmos<sup>1</sup>, Silvia Calabuig<sup>13,14,15</sup>, Ramon Colomer<sup>7,16</sup>, Antonio Artes<sup>1</sup>, Miguel Quintela-Fandino<sup>2,7</sup>

## Afiliaciones

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

## Estructura del repositorio

```
.
├── data/                 # Datos procesados y anonimizados
│   ├── processed/        # Tablas listas para análisis
│   │   ├── behavioral/   # Trayectorias conductuales (actividad, sueño, etc.)
│   │   ├── physiologic/  # Trayectorias fisiológicas (FC, HRV, etc.)
│   │   └── clinical/     # Desenlaces y variables clínicas
│   └── metadata/         # Diccionarios de variables, cohortes, códigos
├── scripts/              # Código de procesamiento, análisis y modelado
│   ├── 01_import/        # Carga y unión de fuentes
│   ├── 02_preprocess/    # Limpieza, alineación temporal, QC
│   ├── 03_features/      # Extracción de trayectorias y biomarcadores
│   ├── 04_analysis/      # Modelos estadísticos y predicción
│   └── 05_figures/       # Figuras del manuscrito
└── results/              # Salidas reproducibles (no datos crudos)
    ├── figures/          # Gráficos y figuras del paper
    ├── tables/           # Tablas suplementarias exportadas
    └── models/           # Objetos de modelos entrenados (si aplica)
```

Ver `data/README.md` para el detalle de cada subcarpeta de datos.

## Requisitos e instalación

<!-- Actualizar según el stack que uses (R, Python, MATLAB, o combinación). -->

Entorno y dependencias: pendiente de documentar.

```bash
# Ejemplo Python (descomentar y adaptar cuando exista requirements.txt)
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt

# Ejemplo R (descomentar cuando exista renv o DESCRIPTION)
# Rscript -e 'renv::restore()'
```

## Reproducibilidad

1. Colocar o generar los datos en `data/processed/` según `data/README.md`.
2. Ejecutar los scripts en orden numérico (`01_` → `05_`).
3. Las figuras y tablas del manuscrito se escriben en `results/`.

## Datos y privacidad

Los datos de pacientes no se publican en crudo. Solo se incluyen en el repositorio conjuntos **procesados y anonimizados** acordes con la normativa aplicable y el consentimiento informado. Los datos en bruto permanecen fuera del control de versiones (ver `.gitignore`).

## Licencia

Código bajo [Apache License 2.0](LICENSE). Los datos pueden tener condiciones de uso adicionales; consultar `data/README.md`.

## Cita

Si usas este material, cita el manuscrito (referencia bibliográfica pendiente de publicación).
