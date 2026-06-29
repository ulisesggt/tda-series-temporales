# Código del TFG: análisis topológico de series temporales

Reproduce todas las figuras y tablas de los capítulos 7 (experimentos
sintéticos) y 8 (datos reales).

## Estructura

- `tda_minimo.py`: implementación propia de Vietoris–Rips y la
  reducción matricial sobre `F_2` para calcular diagramas de
  persistencia. Funciona en `numpy` puro. Se usa como respaldo cuando
  `ripser` no está instalado; lenta en nubes grandes.
- `detector.py`: detector robusto de cambios basado en
  `mediana + k·MAD` con burn-in y exigencia de cruces consecutivos.
  Sustituye al criterio ingenuo `media + 3·sigma`, que producía
  falsos positivos cuando el tramo de referencia es corto.
- `senales.py`: generadores de las señales sintéticas (cosenoide,
  cosenoide con ruido, ruido blanco, dos frecuencias) y de los tres
  escenarios de cambio de régimen del capítulo 7.
- `datos_reales.py`: descarga del S&P 500 con `yfinance` (preferido),
  caída automática al S&P 500 mensual de Shiller via datahub.io, y
  como último recurso un sustituto sintético calibrado para cada
  evento. Catálogo de eventos disponibles: crash 1929, Lunes Negro
  1987, punto-com 2000, crisis 2008, COVID 2020, mercado bajista 2022.
- `ejecutar_cap7.py`: figuras y tablas del capítulo 7 (señales,
  embeddings, score SW1PerS, barrido de parámetros).
- `ejecutar_cap7_extra.py`: figuras pedagógicas adicionales del
  capítulo 7 (códigos de barras, persistencia frente al ruido,
  rejilla refinada de parámetros, proyecciones PCA).
- `ejecutar_cap8.py`: análisis del crash de 2008 en detalle.
- `ejecutar_cap8_eventos.py`: aplica el método de detección de cambios
  a los seis eventos del catálogo y genera figura comparativa.
- `ejecutar_cap8_baseline.py`: compara la señal topológica con un
  baseline clásico (varianza móvil) para los mismos eventos.

## Requisitos

- Python 3.10 o superior.
- `numpy`, `pandas`, `matplotlib` (obligatorios).
- `ripser` y `persim` (recomendados; aceleran los cálculos).
- `yfinance` (recomendado para datos reales del capítulo 8; si falla,
  se intenta datahub.io y, en último caso, el sustituto sintético).

Instalación rápida en el entorno del usuario:

```bash
pip install numpy pandas matplotlib ripser persim yfinance
```

## Ejecución

Desde el directorio `codigo/`:

```bash
python ejecutar_cap7.py
python ejecutar_cap7_extra.py
python ejecutar_cap8.py
python ejecutar_cap8_eventos.py
python ejecutar_cap8_baseline.py
```

Tiempos aproximados con `ripser` instalado: ~30 s todo. Con la
implementación propia `tda_minimo`: varios minutos para los
ejecutables del capítulo 8.

Las figuras se guardan en
`../TFG_plantilla_POLITÉCNICA_overleaf/Imágenes/capitulo7/` y
`../TFG_plantilla_POLITÉCNICA_overleaf/Imágenes/capitulo8/`.

Los CSV con resultados numéricos se guardan en `../resultados/`.

## Reproducibilidad

Todos los generadores aleatorios usan semillas fijas
(`np.random.default_rng(...)`). Cada figura es reproducible bit a bit
en la misma versión de Python y `numpy`.

## Sobre los datos del capítulo 8

El módulo `datos_reales.py` intenta tres fuentes por orden de
preferencia:

1. **`yfinance` (real, diario)**: la opción habitual cuando hay
   internet y la librería está instalada.
2. **Shiller via datahub.io (real, mensual)**: si `yfinance` falla
   pero la conexión sí funciona, se descarga el S&P 500 mensual
   recopilado por Robert Shiller (cobertura desde 1871). Resolución
   peor pero datos reales para los eventos antiguos.
3. **Sustituto sintético calibrado**: si ninguna fuente externa está
   disponible (por ejemplo en un sandbox sin red), se genera una
   serie por tramos cuyos parámetros (nivel inicial, drift y
   volatilidad por régimen, eventual salto puntual) están ajustados a
   partir de valores publicados para cada evento. La fuente concreta
   queda registrada en la columna `fuente` de cada DataFrame y se
   imprime en la cabecera de las figuras.

La conclusión metodológica no depende del origen exacto de los datos
porque el método se basa en la geometría local de los retornos y no
en su nivel absoluto. Aun así, **se recomienda al usuario reejecutar
el código en su propio entorno con `yfinance` instalado y conexión a
internet**, para que las cifras finales del TFG reflejen el dato real.

## Notas honestas

- Si `ripser` no se importa correctamente, el código avisa por consola
  y usa la implementación propia (`tda_minimo`). El resultado es el
  mismo, pero más lento; en ese caso las nubes se submuestrean a 40–60
  puntos para mantener tiempos razonables.
- Si `yfinance` falla y datahub.io no es alcanzable, las figuras se
  generan con el sustituto sintético. Estará claramente etiquetado en
  la columna `fuente` y en la cabecera de cada figura.
