"""
Generación de señales sintéticas para el Capítulo 7.

Todas las funciones aceptan una `semilla` opcional para reproducibilidad
del ruido y devuelven (t, x), donde t son los instantes de muestreo y x
los valores de la señal.
"""

import numpy as np


def senal_sinusoidal(T=2000, periodo=40, amplitud=1.0):
    """Señal cosenoidal pura. Período medido en número de muestras."""
    t = np.arange(T)
    x = amplitud * np.cos(2 * np.pi * t / periodo)
    return t, x


def senal_sinusoidal_con_ruido(T=2000, periodo=40, amplitud=1.0, sigma=0.3,
                                semilla=0):
    """Cosenoide + ruido gaussiano N(0, sigma^2)."""
    rng = np.random.default_rng(semilla)
    t = np.arange(T)
    x = amplitud * np.cos(2 * np.pi * t / periodo) + sigma * rng.standard_normal(T)
    return t, x


def senal_aleatoria(T=2000, sigma=1.0, semilla=1):
    """Ruido blanco gaussiano."""
    rng = np.random.default_rng(semilla)
    t = np.arange(T)
    x = sigma * rng.standard_normal(T)
    return t, x


def senal_dos_frecuencias(T=2000, periodo1=40, periodo2=70, amplitud=1.0):
    """Suma de dos cosenoidales (genera nube cuasi-periódica)."""
    t = np.arange(T)
    x = amplitud * (np.cos(2 * np.pi * t / periodo1) +
                    np.cos(2 * np.pi * t / periodo2))
    return t, x


def senal_periodica_a_ruido(T=500, periodo=40, amplitud=1.0, sigma=0.6,
                             cambio=250, semilla=2):
    """
    Señal con cambio de régimen: primero periódica, después ruido gaussiano.

    Útil para validar la detección de cambios. `cambio` es el índice donde
    se produce la transición.
    """
    rng = np.random.default_rng(semilla)
    t = np.arange(T)
    x = np.empty(T)
    x[:cambio] = amplitud * np.cos(2 * np.pi * t[:cambio] / periodo)
    x[cambio:] = sigma * rng.standard_normal(T - cambio)
    return t, x


def senal_cambio_frecuencia(T=500, periodo1=40, periodo2=70, amplitud=1.0,
                              cambio=250):
    """Periódica con cambio brusco de período en `cambio`."""
    t = np.arange(T)
    x = np.empty(T)
    x[:cambio] = amplitud * np.cos(2 * np.pi * t[:cambio] / periodo1)
    x[cambio:] = amplitud * np.cos(2 * np.pi * t[cambio:] / periodo2)
    return t, x


def senal_ruido_a_periodica(T=500, periodo=40, amplitud=1.0, sigma=0.6,
                              cambio=250, semilla=3):
    """Primero ruido, después señal periódica."""
    rng = np.random.default_rng(semilla)
    t = np.arange(T)
    x = np.empty(T)
    x[:cambio] = sigma * rng.standard_normal(cambio)
    x[cambio:] = amplitud * np.cos(2 * np.pi * t[cambio:] / periodo)
    return t, x
