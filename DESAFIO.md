# Desafío QualyData: narrativas y coordinación a escala

**Hackathon Argentina Space 2026** · Track Datos y Streaming · Centro de Convenciones de Salta · 11 al 13 de noviembre de 2026

> **Aviso.** Todos los datos de este repositorio son sintéticos. Las publicaciones, comentarios, cuentas, medios, lugares y empresas son ficticios y fueron generados por computadora. Cualquier parecido con hechos o personas reales es coincidencia. No hay datos personales.

QualyData es una plataforma que integra y procesa grandes volúmenes de datos de múltiples fuentes para modelar escenarios y tomar decisiones. Este desafío toma uno de los problemas más difíciles de ese trabajo y lo abre a los equipos del hackathon.

## El problema

Cuando miles de personas discuten un tema público, hay dos preguntas que importan:

1. **¿Qué narrativas circulan?** Una narrativa es una afirmación o un encuadre que se repite con distintas palabras, por ejemplo "la obra está parada porque se robaron la plata". Interesa saber cuáles hay, quién las sostiene, quién las rechaza y cómo nació y se propagó cada una.
2. **¿Qué parte es genuina y qué parte está armada?** Algunos grupos de cuentas actúan coordinados: publican lo mismo, al mismo tiempo y en los mismos lugares, para inflar una idea o instalar un rumor.

Hacerlo comentario por comentario con un modelo de IA es lento y caro. Hacerlo con palabras clave falla, porque la gente dice lo mismo de mil maneras. El desafío es construir algo que lo resuelva bien, rápido y barato.

## El desafío

Dos tareas sobre el mismo set de datos:

- **Tarea A: narrativas.** Descubrir las narrativas que circulan (no hay una lista dada de antemano), asignar cada comentario a su narrativa o marcarlo como ruido, indicar si apoya o rechaza esa narrativa y estimar cuándo apareció cada una.
- **Tarea B: coordinación.** Detectar las cuentas que actúan coordinadas y agruparlas por operación, separando la actividad armada de la opinión genuina.

## Los datos

El set de práctica está en `datos/practica/`:

| Archivo | Contenido | Columnas |
|---|---|---|
| `fuentes.csv` | 8 páginas de medios, radios, comunidades e instituciones | `fuente_id, nombre, tipo` |
| `publicaciones.csv.gz` | ~500 publicaciones en 14 días | `publicacion_id, fuente_id, fecha, titular` |
| `comentarios.csv.gz` | 200.000 comentarios | `comentario_id, publicacion_id, cuenta_id, fecha, texto, respuesta_a, me_gusta` |
| `cuentas.csv.gz` | ~16.000 cuentas activas | `cuenta_id, usuario, fecha_creacion, seguidores, siguiendo` |

Las fechas están en hora argentina (`-03:00`). `respuesta_a` tiene el `comentario_id` al que responde, o queda vacío.

**Formato.** Los archivos son CSV en UTF-8 con encabezado, comprimidos con gzip, que es el formato más universal. No imponemos base de datos ni herramienta: impórtenlos donde quieran (PostgreSQL, DuckDB, SQLite, ClickHouse, Elasticsearch, un dataframe, etc.). Elegir dónde y cómo guardar los datos es parte del desafío y lo evalúa el jurado (ver [Puntaje](#puntaje)).

**Qué hay adentro.** Como en una conversación real, el set incluye:
- Narrativas que **cambian de redacción** con los días.
- Narrativas que **aparecen a mitad del período**, incluidos rumores.
- **Ruido**: reacciones, menciones, comentarios fuera de tema.
- Campañas coordinadas de **distintos estilos**, algunas fáciles de ver y otras no.
- **Trampas**: momentos virales genuinos donde mucha gente comenta a la vez y frases que se copian sin coordinación.

**La verdad del set de práctica** está en `datos/practica/verdad/`, para que puedan medir su avance: narrativa y postura de cada comentario, campaña de cada cuenta y la descripción de cada narrativa y campaña. El set final no trae verdad.

## Por qué no se resuelve en una tarde con IA

- **Volumen:** son cientos de miles de comentarios. No entran en un prompt, y procesarlos uno por uno con un modelo es lento y caro.
- **Costo:** se puntúa el costo por cada mil comentarios procesados.
- **Redacción cambiante:** una misma narrativa aparece con palabras distintas a lo largo de los días, así que un match por palabras clave falla.
- **Coordinación invisible:** no se ve comentario por comentario. Aparece solo mirando el conjunto: textos casi idénticos, ráfagas de horario, cuentas que siempre coinciden, antigüedad de las cuentas.
- **Set final nuevo:** el día 3 se corre sobre un set con otros temas, otras narrativas y otras campañas. Gana el pipeline que generaliza, no el que se ajustó a mano al set de práctica.

Se puede usar IA, por ejemplo para nombrar los grupos o resolver casos dudosos. La clave es usarla donde suma y no como único motor.

## Cómo empezar

Requisitos: Python 3.9 o superior.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
python ejemplos/linea_base.py --datos datos/practica --entrega salida/practica
```

```bash
python evaluacion/puntuar.py --verdad datos/practica/verdad --entrega salida/practica
```

Las corridas de práctica van a `salida/`, que no se sube al repo. La carpeta `entrega/` queda reservada para el resultado del set final.

`ejemplos/linea_base.py` es una línea base mínima: agrupa con TF-IDF y k-means, asigna postura con una lista de palabras y marca cuentas que pegan el mismo texto en ráfagas. Saca alrededor de **17 de 55 puntos** automáticos en el set de práctica. Sirve para entender el formato y tener un piso; no es el camino a seguir.

`evaluacion/puntuar.py` es exactamente el script que usa el jurado para la parte automática. No necesita instalar nada.

## Cómo se entrega

1. El viernes 13 se publica el set final. Descárguenlo en `datos/final/`, que no se sube al repo.
2. Corran su pipeline sobre el set final y dejen el resultado en la carpeta `entrega/`, en la raíz de su repo.
3. Commiteen y pusheen `entrega/` antes del cierre.
4. Carguen el proyecto en la plataforma del hackathon (Hacki) con el link al repositorio público y un video de demo.

Lo que se puntúa es lo que está en el repo a la hora de cierre. Al cierre, la organización clona cada repo y anota el commit; lo que se suba después no cuenta.

## Formato de entrega

La carpeta `entrega/` lleva cuatro archivos:

**`comentarios.csv`**: una fila por comentario.

| Columna | Valor |
|---|---|
| `comentario_id` | El id del comentario. |
| `narrativa` | Una etiqueta propia para la narrativa, por ejemplo `n07`. Vacío o `ninguna` si es ruido. |
| `postura` | `apoya` si el comentario afirma o respalda la narrativa, `rechaza` si la niega o la refuta. |

**`narrativas.csv`**: una fila por narrativa detectada.

| Columna | Valor |
|---|---|
| `narrativa` | La misma etiqueta usada en `comentarios.csv`. |
| `titulo` | Una frase que resuma la afirmación. |
| `frases_clave` | Términos separados por `|`. |
| `fecha_origen` | Cuándo apareció, en formato ISO con zona horaria. Por ejemplo `2026-09-07T08:15:00-03:00`. |

**`cuentas.csv`**: una fila por cuenta.

| Columna | Valor |
|---|---|
| `cuenta_id` | El id de la cuenta. |
| `grupo` | Una etiqueta propia para la operación coordinada. Vacío si la cuenta es genuina. Las cuentas de una misma operación llevan la misma etiqueta. |

**`costo.json`**: cuánto costó procesar el set.

```json
{
  "comentarios_procesados": 150000,
  "segundos": 840,
  "tokens_ia": 120000,
  "usd_ia": 0.45,
  "usd_por_mil_comentarios": 0.003,
  "hardware": "notebook, 16 GB RAM, sin GPU",
  "almacenamiento": "DuckDB local sobre Parquet",
  "notas": "embeddings locales; IA solo para nombrar narrativas"
}
```

Detalles de postura:
- Las respuestas cortas que solo acompañan, como "Tal cual" o "Exacto", heredan la narrativa y la postura del comentario al que responden.
- Una respuesta que contradice al comentario anterior pertenece a la misma narrativa, con la postura opuesta.

## Puntaje

*Propuesta sujeta a las bases y condiciones del hackathon.*

| Criterio | Puntos | Cómo se mide |
|---|---|---|
| Calidad de narrativas | 30 | Automático sobre el set final. |
| Detección de coordinación | 25 | Automático sobre el set final. |
| Costo, escala y almacenamiento | 20 | Jurado: `costo.json` y tiempo de proceso; dónde y cómo guardan los datos, y por qué esa base; cómo escalaría a millones de comentarios por día. |
| Robustez del pipeline | 15 | Jurado: corre de punta a punta con un comando, maneja datos sucios y es reproducible. |
| Claridad y visualización | 10 | Jurado: demo y pitch. ¿Se entiende en un minuto qué circula, desde cuándo y quién lo empuja? |

**Detalle de la parte automática:**
- **Narrativas (30):**
  - 50% agrupamiento, medido con B-cubed F1 sobre los comentarios que pertenecen a alguna narrativa. No importa cómo llamen a los grupos ni cuántos armen; importa que lo que va junto quede junto.
  - 15% detección de ruido, medida con F1 sobre la clase "ninguna".
  - 20% postura, medida con macro-F1 entre apoya y rechaza.
  - 15% origen de las narrativas emergentes: puntaje completo si `fecha_origen` cae a menos de 3 horas del origen real, y cae a cero a las 48 horas.
- **Coordinación (25):**
  - 70% detección de cuentas coordinadas, medida con F1.
  - 30% agrupamiento por operación, medido con B-cubed F1.

## Reglas

**Requisitos excluyentes:**
- Entrega el día 3 en Hacki: repositorio público con `entrega/` commiteada, video de demo y pitch ante el jurado.
- El README del equipo explica cómo correr el pipeline, dónde guardan los datos y por qué eligieron esa base, y qué código o librerías existían antes del evento.
- El pipeline corre de punta a punta con un comando sobre el set final. No se permite etiquetar a mano el set final; el jurado puede volver a correrlo.
- Trabajo original hecho durante el hackathon. Se pueden usar librerías, modelos y servicios públicos.
- Solo los datos provistos o fuentes públicas, sin datos personales.
- Sin claves ni secretos en el repositorio.

**Uso de IA:** está permitido. Todo lo que se gaste se reporta en `costo.json`. Si la organización entrega créditos, el presupuesto es el mismo para todos los equipos.

**Licencia:** proponemos que el código de los equipos se publique con licencia abierta (MIT o Apache 2.0), sujeto a lo que definan las bases.

## Cronograma

| Día | Qué pasa |
|---|---|
| Miércoles 11 | Apertura y formación de equipos. Exploración de los datos y primera versión del pipeline. |
| Jueves 12 | Mejoras, detección de coordinación y visualización. Mentorías. |
| Viernes 13 | A la mañana se publica el set final; el horario se confirma en el evento. Cada equipo corre su pipeline, pushea `entrega/` y carga el proyecto en Hacki antes del cierre. Después, pitch ante el jurado. |

## Pistas

No son recetas, pero ayudan a pensar:
- Los modelos de embeddings multilingües chicos corren en una notebook y agrupan por significado, no por palabras.
- Cuántas narrativas hay no se sabe de antemano. Conviene un método que lo decida solo.
- La postura depende de la narrativa: "es mentira que X" rechaza X, aunque use las mismas palabras que X.
- Una ráfaga de comentarios no alcanza para hablar de coordinación. Lo que delata a una operación es la repetición: las mismas cuentas, juntas, una y otra vez.
- Comparen contra lo esperable. Veinte comentarios en diez minutos es normal en una publicación viral y raro en una que nadie mira.
- Las respuestas (`respuesta_a`), la antigüedad de las cuentas y los `me_gusta` también son señales.

## Preguntas frecuentes

**¿Cuántas narrativas y campañas hay?** En el set de práctica, todo está en `datos/practica/verdad/`. En el set final no se informa.

**¿El set final es parecido al de práctica?** Tiene la misma estructura de archivos y un volumen parecido (unos 150.000 comentarios en 10 días), pero otros temas, otras narrativas y otras campañas.

**¿Hace falta una interfaz?** No es obligatoria. Sí se evalúa cómo comunican los resultados: un dashboard, un notebook bien armado o una buena demo sirven.

**¿Puedo entrenar un modelo con la verdad del set de práctica?** Sí. Tengan en cuenta que en el set final los temas cambian, así que lo que aprendan tiene que generalizar.
