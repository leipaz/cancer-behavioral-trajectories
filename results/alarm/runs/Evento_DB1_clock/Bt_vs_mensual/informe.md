# Alarma mensual colapsada y serie diaria

## 1. Análisis global del mes colapsado

Agrupación de mes colapsado sobre la base de tópicos de los 178 pacientes: un bloque LDA por mes, sin solape entre meses. Ventana de 3 meses, horizonte de 4.

\[
\mathrm{pUF}_m = P_m(3)+P_m(4)+P_m(5)
\]

\[
B_m = \frac{\mathrm{pUF}_{m-2}+\mathrm{pUF}_{m-1}+\mathrm{pUF}_m}{3} \qquad (m \ge 3)
\]

\[
S_m = 3\,B_m
\]

\[
T = 30\,m
\]

\(B_m\) es el score. Alarma si \(S_m \ge \theta\), es decir si \(B_m \ge \theta/3\).

\[
y = 1 \text{ si hay evento y } t_{\mathrm{evento}} \le T+120
\]

\[
y = 0 \text{ si el seguimiento llega a } T+120
\]

Si el seguimiento acaba antes de \(T+120\), el punto no entra.

\[
\theta = \arg\max\,(\mathrm{sens}+\mathrm{spec}-1) = 1{,}0596384140662845
\]

\[
\theta/3 = 0{,}3532128046887615
\]

Un solo máximo. Empate: el corte más bajo. Intervalos: bootstrap de 2.000 por paciente, corte fijo.

| | |
|---|---:|
| Pacientes de la base | 178 |
| Eventos en la base | 60 |
| Puntos evaluables | 1.016 |
| Pacientes con punto | 116 |
| Positivos / negativos | 156 / 860 |
| Eventos en los puntos | 45 |
| AUROC | 0,700 (0,620–0,773) |
| Sensibilidad | 0,878 (0,792–0,952) |
| Especificidad | 0,450 (0,361–0,534) |
| VPP | 0,225 (0,171–0,288) |
| VPN | 0,953 (0,917–0,981) |
| TP / FN / FP / TN | 137 / 19 / 473 / 387 |

Cotejo del mismo cálculo sobre la misma base: los 1.016 puntos coinciden.

| | |
|---|---:|
| Puntos en ambos lados | 1.016 |
| Solo en un lado | 0 |
| \(\max\|\Delta S\|\) | \(1{,}3 \times 10^{-15}\) |
| \(\max\|\Delta \mathrm{pUF}\|\) | \(4{,}4 \times 10^{-16}\) |
| Etiquetas o eventos distintos | 0 |
| TP / FN / FP / TN | 137 / 19 / 473 / 387 |

## 2. Comparación de la serie diaria con el mes colapsado

\(q(t)\) es la pUF de los 30 días que acaban en \(t\), con el mismo LDA. \(B(t)\) junta tres ventanas seguidas, sin solape:

\[
B(t) = \frac{q(t-60)+q(t-30)+q(t)}{3}
\]

En un mes completo, \(B(30m)\) ya es el score: la media de las tres pUF del mes. \(S_m\) es esa misma cantidad escrita como suma.

\[
B(30m) = \frac{\mathrm{pUF}_{m-2}+\mathrm{pUF}_{m-1}+\mathrm{pUF}_m}{3}, \qquad S_m = 3\,B(30m)
\]

| | |
|---|---:|
| Meses completos (\(L \ge T\)) | 1.009 |
| Coinciden | 1.009 |
| \(\max\|B-S/3\|\) | \(1{,}7 \times 10^{-9}\) |

## 3. Siete landmarks en el mensual y no en la diaria

Esas 7 están en el mes colapsado y no tienen \(B(T)\) en la serie diaria.

En el colapsado, si el último mes no tenía 30 días, se completaba con los últimos 30 días del perfil para no perder esos días. El punto seguía en \(T = 30m\):

\[
\mathrm{pUF}_m = q(L)
\]

En la diaria no existe el día \(T\), así que \(B(T)\) no sale. Hay que elegir:

| | Qué se hace | Puntos |
|---|---|---:|
| Conservar el mes incompleto | El mismo criterio del colapsado: score con los últimos 30 días y punto en \(T\) | 1.016 en las dos series |
| Quitar las 7 | Solo meses con \(L \ge T\) | 1.009, ya iguales |

\(S_m\) es la suma de las tres pUF, \(3B\). 22001 y 31005, las dos primeras, tienen el último mes incompleto (\(L < T\)), pero la progresión cae dentro del horizonte: día 520 frente a \(T+120 = 540\), y día 447 frente a \(T+120 = 480\). Si se cuentan esos días, las dos landmarks son válidas.

| ID | \(T\) | \(L\) | Día de fin | Evento | \(y\) | \(S_m\) | Si se quitan del mensual |
|---|---:|---:|---:|---:|---:|---:|---|
| 22001 | 420 | 408 | 520 | 1 | 1 | 2,519 | Único \(y=1\): se pierde el evento |
| 31005 | 360 | 357 | 447 | 1 | 1 | 1,080 | El mes 11 ya es \(y=1\) (\(S=0{,}584\)) y no alarma |
| 31003 | 360 | 347 | 492 | 0 | 0 | 1,094 | |
| 41022 | 150 | 132 | 400 | 0 | 0 | 1,566 | |
| 51008 | 180 | 179 | 386 | 0 | 0 | 0,048 | |
| 71005 | 90 | 83 | 286 | 0 | 0 | 0,358 | Único punto: sale el paciente |
| 71009 | 90 | 79 | 255 | 0 | 0 | 0,309 | Único punto: sale el paciente |

| | Con las 7 | Sin las 7, mismo corte | Sin las 7, Youden reestimado |
|---|---:|---:|---:|
| Puntos | 1.016 | 1.009 | 1.009 |
| Pacientes / eventos | 116 / 45 | 114 / 44 | 114 / 44 |
| Corte | 1,060 | 1,060 | 1,090 |
| Sensibilidad | 0,878 | 0,877 | 0,870 |
| Especificidad | 0,450 | 0,449 | 0,457 |
| TP / FN / FP / TN | 137 / 19 / 473 / 387 | 135 / 19 / 471 / 384 | 134 / 20 / 464 / 391 |

## 4. Granularidad con meses completos desde el día 90

Esas 7 landmarks se caen de la serie diaria si, desde el día 90, solo se cuentan ventanas completas. El calendario no se mueve: diaria 90, 91, 92…; semanal 90, 97, 104…; cada 14 días 90, 104, 118…; cada 30 días 90, 120, 150…. Hace falta \(q(t-60)\), \(q(t-30)\) y \(q(t)\). Si falta el día, ese punto no entra.

\(B(t)\) es el score. Horizonte de 120 días. Los dos cortes se dejan fijos, sin volver a estimarlos en cada cadencia:

\[
B \ge 0{,}3532128046887615 \quad (3B = 1{,}0596384140662845)
\]

\[
B \ge 0{,}3634247337467968 \quad (3B = 1{,}0902742012403903)
\]

| Cadencia | Con \(B(t)\) | Evaluables | Pacientes | Eventos | AUROC |
|---|---:|---:|---:|---:|---:|
| Diaria | 37.753 | 28.603 | 114 | 45 | 0,700 |
| Semanal | 5.461 | 4.140 | 114 | 45 | 0,700 |
| Cada 14 días | 2.765 | 2.101 | 114 | 44 | 0,702 |
| Cada 30 días | 1.327 | 1.009 | 114 | 44 | 0,699 |

Cobertura, en cada cadencia. El posterior \(q(t)\) es calculable si ese día está en el perfil: no se añade un mínimo nuevo, no se rellena con ceros y no se salta el hueco. \(B(t)\) exige los tres. Luego entran en las métricas solo los que tienen desenlace evaluable.

| Cadencia | Calendario | Falta un \(q\)* | Con \(B(t)\) | Sin desenlace | En las métricas | Pacientes | Eventos |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diaria | 40.424 | 2.671 | 37.753 | 9.150 | 28.603 | 114 | 45 |
| Semanal | 5.845 | 384 | 5.461 | 1.321 | 4.140 | 114 | 45 |
| Cada 14 días | 2.959 | 194 | 2.765 | 664 | 2.101 | 114 | 44 |
| Cada 30 días | 1.421 | 94 | 1.327 | 318 | 1.009 | 114 | 44 |

Calendario: días 90, 90+paso, … mientras el día sea anterior al fin del seguimiento. Falta un \(q\): no está \(q(t-60)\), \(q(t-30)\) o \(q(t)\). Sin desenlace: hay \(B(t)\), pero el seguimiento se acaba antes de 120 días y no hay evento dentro. En las métricas = con \(B(t)\) − sin desenlace. Pacientes y eventos son los de esos puntos incluidos. Cada 30 días, las 7 del mes incompleto están en «falta un \(q\)».

\* Falta un \(q\) solo porque el perfil ya se ha acabado y el calendario sigue. No hay días intermedios ausentes: dentro del perfil los días van seguidos. Cada \(q\) necesita los 30 días que terminan ese día. \(B(t)\) necesita tres ventanas de ese tipo, las que acaban en \(t-60\), \(t-30\) y \(t\). En la diaria esas tres ventanas se desplazan un día, no pasan a ser tres días sueltos.

Corte \(B \ge 0{,}3532\ldots\), el Youden de los 1.016:

| Cadencia | Sens | Spec | TP / FN / FP / TN |
|---|---:|---:|---|
| Diaria | 0,850 | 0,457 | 3729 / 659 / 13147 / 11068 |
| Semanal | 0,851 | 0,453 | 538 / 94 / 1918 / 1590 |
| Cada 14 días | 0,856 | 0,452 | 273 / 46 / 976 / 806 |
| Cada 30 días | 0,877 | 0,449 | 135 / 19 / 471 / 384 |

Corte \(B \ge 0{,}3634\ldots\), el Youden de los 1.009:

| Cadencia | Sens | Spec | TP / FN / FP / TN |
|---|---:|---:|---|
| Diaria | 0,842 | 0,465 | 3696 / 692 / 12951 / 11264 |
| Semanal | 0,845 | 0,462 | 534 / 98 / 1886 / 1622 |
| Cada 14 días | 0,850 | 0,459 | 271 / 48 / 964 / 818 |
| Cada 30 días | 0,870 | 0,457 | 134 / 20 / 464 / 391 |

La semanal conserva el evento de 22001 por el día 405. Cada 14 días el calendario pasa del 398 al 412, y el 412 ya no tiene perfil.
