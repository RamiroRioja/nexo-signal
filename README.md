# NEXO SIGNAL

Pipeline local para descubrir narrativas, posturas y coordinación en conversaciones públicas. No llama a servicios externos ni usa IA generativa.

## Ejecución

```bash
python -m pip install -r requirements.txt
python -m nexo --datos datos/practica --salida salida/practica
python evaluacion/puntuar.py --verdad datos/practica/verdad --entrega salida/practica
```

Se admite Python 3.10–3.13. Las dependencias tienen ruedas precompiladas para Windows x86_64, Linux x86_64 (glibc 2.28 o posterior) y macOS arm64 (12 o posterior). Se recomienda instalar en un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
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

La ejecución fija las semillas de Python, NumPy, scikit-learn, NetworkX y PyTorch, usa algoritmos deterministas de PyTorch cuando están disponibles y limita BLAS, DuckDB y PyTorch a 8 hilos fijos. El modelo y los vectores se guardan en cachés locales; el caché de embeddings se identifica por modelo y hash de texto y no cambia los resultados.

## Dashboard

```bash
streamlit run dashboard/app.py -- --datos datos/practica --salida salida/practica
```

El pipeline escribe `tiempos.json` junto con las entregas para mostrar la duración medida de cada etapa.

## Reproducir la entrega

En Linux/macOS:

```bash
python -m pip install -r requirements.txt
bash scripts/descargar_final.sh
python scripts/verificar_entrada.py --datos datos/final
python -m nexo --datos datos/final --salida entrega
python scripts/verificar_entrega.py --entrega entrega --datos datos/final
```

En Windows PowerShell, reemplaza el descargador Bash por su equivalente Python:

```powershell
python scripts/descargar_final.py
python scripts/verificar_entrada.py --datos datos/final
python -m nexo --datos datos/final --salida entrega
python scripts/verificar_entrega.py --entrega entrega --datos datos/final
```

El set de práctica de 200.000 comentarios se procesó en aproximadamente 2,9 minutos con el modelo ya en caché y 5 minutos en una corrida limpia que lo descarga, en una máquina con 8 CPU lógicas, 31 GB de RAM y sin GPU. La ejecución usa CPU y 8 hilos fijos para reproducir las salidas; el tiempo exacto figura en `entrega/costo.json`.
