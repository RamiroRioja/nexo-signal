# NEXO SIGNAL

Pipeline local para descubrir narrativas, posturas y coordinación en conversaciones públicas. No llama a servicios externos ni usa IA generativa.

## Ejecución

```bash
python -m pip install -r requirements.txt
python -m nexo --datos datos/practica --salida salida/practica
python evaluacion/puntuar.py --verdad datos/practica/verdad --entrega salida/practica
```

Para preparar el modelo multilingüe local antes de una corrida sin conexión:

```bash
python -m nexo.descargar_modelo
```

La primera ejecución puede descargar `paraphrase-multilingual-MiniLM-L12-v2`; luego queda en `modelos/`. También se puede ejecutar `make eval` o `bash scripts/evaluar.sh`. La entrada y la salida se indican con `--datos` y `--salida` (por defecto `datos/practica` y `entrega`).

## Datos y almacenamiento

El programa importa los CSV comprimidos con DuckDB, guarda una copia Parquet en `salida/<corrida>/data/` y materializa tablas consultables en `nexo.duckdb`: comentarios enriquecidos, narrativas, grupos con evidencia y conteos horarios. DuckDB facilita joins y agregaciones locales reproducibles; Parquet comprime bien, conserva tipos y permite consultar los datos sin volver a leer los CSV. Los modelos y su caché de vectores quedan en `modelos/` y no se suben al repositorio.

## Dependencias y escala

DuckDB, pandas, NumPy, scikit-learn, NetworkX, PyTorch y Sentence Transformers son bibliotecas públicas preexistentes; no se incluyen servicios privados ni código externo de etiquetado. El procesamiento deduplica textos antes de generar vectores y limita pares en ventanas virales. Para crecer a millones de comentarios por día, el mismo esquema puede particionar Parquet por fecha y procesar particiones en paralelo, manteniendo DuckDB para SQL local.

## Evaluación y reproducibilidad

`make eval` puntúa la práctica y ejecuta la evaluación de generalización por tema y por últimos diez días. Las semillas están fijadas; los resultados de la corrida y los intermedios quedan en la carpeta indicada.

`python scripts/verificar_robustez.py` ejecuta una corrida pequeña para comprobar textos vacíos, fechas inválidas, respuestas huérfanas, ciclos, IDs duplicados y la unión de cuentas.

## Dashboard

```bash
streamlit run dashboard/app.py -- --datos datos/practica --salida salida/practica
```

El pipeline escribe `tiempos.json` junto con las entregas para mostrar la duración medida de cada etapa.
