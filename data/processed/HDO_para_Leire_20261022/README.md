# HDO: cierre de la alarma mensual de DB1
Fecha: 22 octubre 2026. Uso interno CNIO/UC3M.

La versión de esta entrega selecciona el máximo de Youden entre todos los scores
distintos. W3/H4 permanece como configuración principal; theta =
1.0596384140662845; theta/3 = 0.35321280468876148.

No mezclar estas salidas con el atlas previo del barrido de 25 cuantiles
(theta 1.003823...) ni con el corte 1.06 redondeado de Leire.

Ejecución: abrir HDO_alarma_dos_celdas.ipynb y ejecutar sus dos celdas.
Alternativamente, copiar las dos celdas del TXT, en orden, en JupyterLab.
Colocar el Excel revisado en la misma carpeta. Requiere numpy, pandas,
scipy, matplotlib y el lector Excel openpyxl ya utilizado en el entorno del usuario.
No se necesitan módulos .py propios.

Población: 178 IDs fuente; 140 con alguna carga calculable; 1.339 puntos con score,
323 de horizonte no resuelto; 116 participantes y 1.016 puntos evaluables,
156 positivos /860 negativos; 45 eventos únicos con horizonte positivo.
Matriz principal: TP137/FN19/FP473/TN387.
Los IC son descriptivos, agrupados por paciente y condicionados al corte seleccionado;
no corrigen el optimismo de su selección ni validan independientemente el modelo.

La enmienda es un BORRADOR: la reestimación se registra después de inspeccionar
los resultados de Stage 1. No hay firma, aprobación o prespecificación retrospectiva.
No se han utilizado resultados de Stage 2/3 en esta ejecución.
Cotejo individual con Leire y alcance del acceso a resultados posteriores: pendientes.

Ejemplos propuestos para principal: 91008,23003,41020,71002.
Suplemento: 11001 (FP persistentes),23004 (evento sin TP).
Alternativas: 61002 (detección tardía),62004 (oscilación, FP tempranos y TP posteriores).
Los ejemplos son seleccionados después del análisis; no son validación.
Conservar los horizontes no resueltos y distinguir primera alarma de primer TP.

Para Leire: enviar únicamente HDO_para_Leire_20261022.zip con el correo propuesto.
No es necesario enviar el atlas completo para el cotejo del método.
Los CSV contienen identificadores seudonimizados y fechas clínicas; compartir
solo por canales institucionales autorizados.
