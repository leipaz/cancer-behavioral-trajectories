# Del diario al VQ-VAE, y dónde entra la máscara

Comprobado sobre los 116 pacientes de `decision_points_collapsed.csv` que están en `data/daily_summaries/DB1/daily_summary_checked.csv`. Modelo fijo: `models/vq-vae/vqvae_a0.pt`. Resultados de la pasada: `procedure_check/`.

## Procedimiento original

1. El diario trae 70 columnas. El cargador se queda solo con las 13 del VQ-VAE: `user`, `date_time`, `service` y las 10 variables de abajo. El resto no entra.
2. Cada variable continua fuera de su rango se pasa a vacío (`CLIP_INFO` en `scripts/vqvae/constants.py`).
3. Si un paciente tiene dos filas el mismo día, las continuas se promedian y `practiced_sport` se queda con el máximo.
4. El calendario de cada paciente va del primer día registrado al último, un día por fecha. Los huecos que ya existían se quedan vacíos. No se recorta el origen.
5. `weekend` no se lee del diario: se calcula otra vez con la fecha (sábado o domingo).
6. Escalado: `log1p` y después centro/escala en las siete variables positivas; `sleep_start` solo centro/escala. Un vacío sigue vacío.
7. La imputación del modelo a0 es un cero en ese espacio escalado. a0 no recibe el canal de máscara: el hueco entra solo como ese cero.
8. El encoder asigna un código de los 256 a cada día del calendario, con el contexto de los días vecinos (convolución de ancho 3, cuatro capas).

Orden de las 10 variables que ve el modelo:

`sleep_start`, `location_distance`, `location_time_home`, `sleep_duration`, `activity_walking`, `app_usage_total`, `location_clusters_count`, `steps_steps_total`, `weekend`, `practiced_sport`.

## Dónde se pega la máscara

Sobre la tabla diaria ya recortada a esas columnas, antes del paso 4. Solo se ocultan medidas:

`sleep_start`, `location_distance`, `location_time_home`, `sleep_duration`, `activity_walking`, `app_usage_total`, `location_clusters_count`, `steps_steps_total`, `practiced_sport`.

No se tocan `user`, `date_time`, `service` ni `weekend`. La fila se conserva, así que ocultar el primer día no mueve la fecha de origen ni acorta la serie. Después se repite el procedimiento de arriba, con los mismos pesos.

## Qué se ha comprobado

Pasada sin máscara repetida dos veces: los 116 códigos salen idénticos.

Máscara del 20% de días (semilla 101), del 20% de celdas observadas (semilla 101) y del primer día del paciente 11001:

- En los 116, la primera fecha, la última y el número de días de calendario no cambian.
- En 11001 el primer día sigue siendo 2023-04-19. Distancia, caminar, uso de app, pasos y deporte pasan a vacío. El fin de semana, que cae en miércoles, sigue en 0 porque lo pone la fecha.
- Esa ocultación del primer día cambia el código de ese día.
- Con el 20% de días ocultos, cambia el código en una mediana del 38% de los días de calendario. Con el 20% de celdas, la mediana es el 51%. El código se mueve también en días vecinos porque el encoder los ve juntos.

Comparación con `data/processed/lda/user_embeddings_from_pkl.csv`: las 116 series tienen la misma longitud. En 104 el código coincide día a día. En las otras 12 coincide en al menos el 98,6% de los días (mediana de acuerdo 1).
