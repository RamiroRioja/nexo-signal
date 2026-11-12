.PHONY: eval entrega

eval:
	bash scripts/evaluar.sh

entrega:
	bash scripts/descargar_final.sh
	python scripts/verificar_entrada.py --datos datos/final
	python -m nexo --datos datos/final --salida entrega
	python scripts/verificar_entrega.py --entrega entrega --datos datos/final
