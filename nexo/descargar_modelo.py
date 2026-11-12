"""Descarga una vez el modelo de embeddings para uso local."""

from pathlib import Path

from .pipeline import ensure_model


if __name__ == "__main__":
    model_path = ensure_model(Path(__file__).resolve().parents[1] / "modelos")
    print(f"Modelo listo en {model_path}")
