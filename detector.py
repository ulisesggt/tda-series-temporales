"""
Detector robusto de cambios sobre una serie unidimensional.

El detector estándar 'media + 3*sigma' usado en una primera versión del
TFG resultó ser inestable cuando el tramo de referencia es corto: la
desviación típica se infla por colas pesadas (típico en distancias
bottleneck y en varianzas móviles de retornos), produciendo umbrales
demasiado laxos y, con ello, falsos positivos muy alejados del cambio
real.

Esta versión sustituye 'media + k*sigma' por 'mediana + k*MAD', con
las siguientes mejoras:

  - Mediana y MAD son estimadores robustos: no se contaminan por unos
    pocos valores extremos del tramo de referencia.
  - MAD se reescala con el factor 1.4826 para que, bajo hipótesis
    gaussiana, sea comparable a sigma.
  - Se exige un número mínimo de muestras de referencia
    (n_min_ref=20) para evitar dictar umbrales con poca información.
  - Se descartan las primeras 'burn_in' muestras del cálculo de la
    detección (no del cálculo del umbral) para evitar que efectos de
    arranque del pipeline cuenten como detección.
  - Para considerar válida una detección, se exige que k cruces
    consecutivos superen el umbral (parámetro 'consecutivos'). Esto
    filtra el ruido aleatorio puntual y obliga a que el cambio sea
    persistente en el tiempo.

El detector devuelve el primer índice del primer cruce confirmado, o
None si no se confirma ninguno.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


_FACTOR_MAD = 1.4826  # MAD * 1.4826 ~ sigma bajo gaussianidad


@dataclass
class ResultadoDetector:
    """Salida del detector."""
    indice_detectado: Optional[int]
    umbral: float
    mediana_ref: float
    mad_ref: float
    n_ref: int
    burn_in: int


def detector_robusto(
    senal: np.ndarray,
    *,
    k: float = 5.0,
    frac_ref: float = 0.30,
    n_min_ref: int = 20,
    burn_in: int = 5,
    consecutivos: int = 2,
) -> ResultadoDetector:
    """
    Detecta el primer cambio significativo en una señal 1-D.

    Parámetros
    ----------
    senal : np.ndarray
        Vector de valores ordenados temporalmente. Se asume no
        negativo o, al menos, con un cambio detectable hacia arriba
        (varianza móvil, distancia bottleneck, etc.).
    k : float, opcional
        Número de MADs por encima de la mediana que define el umbral.
        Valor típico: 5 (más estricto que 3, robusto frente a outliers).
    frac_ref : float, opcional
        Fracción inicial de la señal usada como tramo de referencia.
    n_min_ref : int, opcional
        Mínimo absoluto de muestras de referencia.
    burn_in : int, opcional
        Número de muestras iniciales que se ignoran en la búsqueda de
        la detección (para evitar artefactos del arranque).
    consecutivos : int, opcional
        Número de muestras consecutivas que deben superar el umbral
        para confirmar la detección.

    Devuelve
    --------
    ResultadoDetector
        Con el índice detectado (o None) y los parámetros del umbral.
    """
    senal = np.asarray(senal, dtype=float)
    n = len(senal)
    if n < n_min_ref + burn_in + consecutivos:
        return ResultadoDetector(None, np.nan, np.nan, np.nan, 0, burn_in)

    n_ref = max(n_min_ref, int(round(frac_ref * n)))
    n_ref = min(n_ref, n - burn_in - consecutivos)
    referencia = senal[:n_ref]

    mediana = float(np.median(referencia))
    mad = float(np.median(np.abs(referencia - mediana))) * _FACTOR_MAD
    if mad <= 1e-15:
        # Si la referencia es perfectamente plana, usamos un suelo
        # numérico (el 5% del rango total) para evitar un umbral cero.
        mad = max(1e-12, 0.05 * (np.max(senal) - np.min(senal)))
    umbral = mediana + k * mad

    # Búsqueda de detección a partir de max(n_ref, burn_in).
    inicio = max(n_ref, burn_in)
    indice_det = None
    racha = 0
    for i in range(inicio, n):
        if senal[i] > umbral:
            racha += 1
            if racha >= consecutivos:
                indice_det = i - consecutivos + 1
                break
        else:
            racha = 0

    return ResultadoDetector(
        indice_detectado=indice_det,
        umbral=umbral,
        mediana_ref=mediana,
        mad_ref=mad,
        n_ref=n_ref,
        burn_in=burn_in,
    )
