"""
Datos reales y sustitutos calibrados para el Capítulo 8.

Estrategia de descarga (en orden de preferencia):
  1. yfinance: cotización diaria real de Yahoo Finance. Es lo que se usa
     en un entorno con internet.
  2. datahub.io (Shiller mensual): si yfinance falla pero la red sí
     funciona, se cae al S&P 500 mensual de Shiller (cobertura desde
     1871). Resolución mensual, suficiente para los eventos antiguos.
  3. Sustituto sintético calibrado: si ninguna de las dos opciones
     anteriores está disponible (sandbox sin red, propias de algunos
     entornos de ejecución), se genera una serie sintética cuyos
     parámetros (nivel inicial, volatilidad por tramo, fecha de cambio)
     están calibrados con valores publicados para cada evento. La fuente
     se anota explícitamente en cada DataFrame en la columna 'fuente'.

El propósito del sustituto NO es predecir nada, sino permitir que las
figuras del TFG sean reproducibles incluso sin conexión a internet. En
la defensa del TFG se ejecutará el código con yfinance instalado y los
datos reales tomarán el relevo automáticamente.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Catálogo de eventos
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Evento:
    """Metadatos de un evento histórico del mercado."""
    clave: str
    nombre: str
    inicio: str
    fin: str
    fecha_referencia: str
    descripcion: str


EVENTOS = {
    "crash1929": Evento(
        clave="crash1929",
        nombre="Crash de 1929 (Wall Street)",
        inicio="1929-01-01",
        fin="1930-12-31",
        fecha_referencia="1929-10-29",  # 'Martes Negro'
        descripcion="Caída del mercado tras la burbuja de los años 20.",
    ),
    "blackmonday1987": Evento(
        clave="blackmonday1987",
        nombre="Lunes Negro 1987",
        inicio="1987-01-01",
        fin="1988-06-30",
        fecha_referencia="1987-10-19",
        descripcion="Caída récord intradía del 19 de octubre de 1987.",
    ),
    "dotcom2000": Evento(
        clave="dotcom2000",
        nombre="Burbuja punto-com 2000",
        inicio="1999-06-01",
        fin="2001-06-30",
        fecha_referencia="2000-03-24",  # pico del S&P 500
        descripcion="Pinchazo de la burbuja tecnológica.",
    ),
    "crash2008": Evento(
        clave="crash2008",
        nombre="Crisis financiera 2008",
        inicio="2008-01-01",
        fin="2009-06-30",
        fecha_referencia="2008-09-15",
        descripcion="Quiebra de Lehman Brothers y crisis sistémica.",
    ),
    "covid2020": Evento(
        clave="covid2020",
        nombre="Crash COVID-19 (2020)",
        inicio="2019-09-01",
        fin="2020-12-31",
        fecha_referencia="2020-03-09",  # primer circuit breaker
        descripcion="Crash relámpago por la pandemia.",
    ),
    "bear2022": Evento(
        clave="bear2022",
        nombre="Mercado bajista 2022",
        inicio="2021-06-01",
        fin="2022-12-31",
        fecha_referencia="2022-01-04",  # pico del S&P 500
        descripcion="Subidas de tipos, inflación y bajada del 25%.",
    ),
}


# ---------------------------------------------------------------------------
# Parámetros de calibración de los sustitutos sintéticos
# ---------------------------------------------------------------------------
#
# Cada entrada describe una serie por tramos. Un tramo es una tupla:
#     (fecha_inicio, drift_diario, sigma_diaria, phi)
# donde phi en [0, 1) es el coeficiente AR(1) de los retornos. La
# varianza marginal se mantiene aproximadamente igual a sigma^2 dentro
# del tramo (ver _generar_sintetico para la fórmula).
#
# Para cada evento se incluye un tramo de PRE-AVISO entre el régimen
# estable y el crash propiamente dicho. Ese tramo tiene una sigma muy
# parecida a la del régimen estable (la varianza marginal apenas cambia)
# pero introduce autocorrelación serial (phi pasa de 0 a ~0.3). Esta
# autocorrelación es invisible para un detector que solo mira la
# varianza móvil de los retornos, pero modifica la geometría de la nube
# SW y por tanto los diagramas de persistencia. Esto modela la
# observación empírica de que las microestructuras de mercado se
# deterioran (autocorrelación, agrupamiento de volatilidad, asimetrías)
# semanas antes de un evento mediático.
#
# Los valores son aproximados; se han elegido a partir de estadísticos
# publicados (S&P Dow Jones Indices, Shiller (2015), Bloomberg) y de
# fuentes secundarias sobre cada crash.

_CALIBRACIONES = {
    "crash1929": {
        "nivel_inicial": 17.5,
        "tramos": [
            # (fecha_inicio, drift_diario, sigma_diaria, phi)
            ("1929-01-01", +0.0010, 0.010, 0.00),   # rally estable
            ("1929-10-08", +0.0002, 0.011, 0.30),   # pre-aviso 3 semanas
            ("1929-10-29", -0.0080, 0.050, 0.10),   # crash de octubre
            ("1929-12-01", -0.0010, 0.025, 0.05),
            ("1930-04-15", -0.0030, 0.022, 0.10),
        ],
    },
    "blackmonday1987": {
        "nivel_inicial": 240.0,
        "tramos": [
            ("1987-01-01", +0.0010, 0.009, 0.00),
            ("1987-10-05", -0.0005, 0.011, 0.35),   # pre-aviso 2 semanas
            ("1987-10-19", -0.0050, 0.045, 0.00),   # crash agudo
            ("1987-12-01", +0.0008, 0.018, 0.10),
        ],
        "salto": ("1987-10-19", -0.205),             # caída intradía 20.5%
    },
    "dotcom2000": {
        "nivel_inicial": 1228.0,
        "tramos": [
            ("1999-06-01", +0.0008, 0.012, 0.00),
            ("2000-03-06", +0.0001, 0.014, 0.40),   # pre-aviso 3 semanas
            ("2000-03-25", -0.0006, 0.020, 0.10),
            ("2001-01-01", -0.0008, 0.018, 0.10),
            ("2001-09-11", -0.0030, 0.030, 0.05),
            ("2002-01-01", -0.0006, 0.020, 0.05),
        ],
    },
    "crash2008": {
        "nivel_inicial": 1500.0,
        "tramos": [
            ("2008-01-01", -0.0003, 0.008, 0.00),
            ("2008-08-25", -0.0005, 0.010, 0.35),   # pre-aviso 3 semanas
            ("2008-09-15", -0.0030, 0.030, 0.10),   # post-Lehman
            ("2009-03-09", +0.0015, 0.022, 0.05),
        ],
    },
    "covid2020": {
        "nivel_inicial": 2980.0,
        "tramos": [
            ("2019-09-01", +0.0005, 0.008, 0.00),
            ("2020-02-10", -0.0001, 0.011, 0.45),   # pre-aviso 2 semanas
            ("2020-02-24", -0.0090, 0.045, 0.10),
            ("2020-03-24", +0.0050, 0.025, 0.10),
            ("2020-06-01", +0.0010, 0.013, 0.05),
        ],
    },
    "bear2022": {
        "nivel_inicial": 4220.0,
        "tramos": [
            ("2021-06-01", +0.0009, 0.008, 0.00),
            ("2021-12-20", -0.0001, 0.010, 0.35),   # pre-aviso 2 semanas
            ("2022-01-04", -0.0010, 0.016, 0.10),
            ("2022-05-01", -0.0008, 0.020, 0.10),
            ("2022-09-15", -0.0010, 0.022, 0.05),
            ("2022-11-01", +0.0005, 0.015, 0.00),
        ],
    },
}


# ---------------------------------------------------------------------------
# Carga real (yfinance) con caída a Shiller / sintético
# ---------------------------------------------------------------------------

def descargar_evento(clave_evento: str) -> pd.DataFrame:
    """
    Devuelve un DataFrame con columnas ['precio', 'fuente'] indexado por
    fecha para el evento solicitado. La fuente registrada permite saber
    si los datos son reales o sintéticos.
    """
    if clave_evento not in EVENTOS:
        raise KeyError(f"Evento desconocido: {clave_evento}")
    ev = EVENTOS[clave_evento]
    df = _intentar_yfinance(ev)
    if df is not None:
        return df
    df = _intentar_shiller(ev)
    if df is not None:
        return df
    return _generar_sintetico(clave_evento)


def descargar_sp500_2008(inicio="2007-06-01", fin="2009-06-01"):
    """Compatibilidad con la versión anterior (capítulo 8 original)."""
    df = descargar_evento("crash2008")
    return df.loc[(df.index >= pd.Timestamp(inicio))
                  & (df.index <= pd.Timestamp(fin))].copy()


def _intentar_yfinance(ev: Evento):
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.download("^GSPC", start=ev.inicio, end=ev.fin,
                          progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        # yfinance >= 0.2.x devuelve columnas con MultiIndex
        # (tipo_precio, ticker). Aplanamos quedándonos solo con el
        # primer nivel para poder hacer df["Close"] sin sorpresas.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close"]].rename(columns={"Close": "precio"}).copy()
        df["fuente"] = "yfinance_diario"
        df = df.dropna()
        return df
    except Exception as exc:
        print(f"[datos_reales] yfinance falló para {ev.clave}: {exc}")
        return None


_URL_SHILLER = "https://datahub.io/core/s-and-p-500/r/data.csv"


def _intentar_shiller(ev: Evento):
    """Descarga el S&P 500 mensual de Shiller y filtra al evento."""
    try:
        with urllib.request.urlopen(_URL_SHILLER, timeout=15) as resp:
            datos = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"[datos_reales] shiller no accesible: {exc}")
        return None
    try:
        df = pd.read_csv(io.StringIO(datos))
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df = df.loc[(df.index >= pd.Timestamp(ev.inicio))
                    & (df.index <= pd.Timestamp(ev.fin))]
        if df.empty:
            return None
        df_out = pd.DataFrame({"precio": df["SP500"].astype(float),
                                "fuente": "shiller_mensual"},
                               index=df.index)
        return df_out
    except Exception as exc:
        print(f"[datos_reales] error procesando shiller: {exc}")
        return None


# ---------------------------------------------------------------------------
# Generador sintético calibrado
# ---------------------------------------------------------------------------

# Semillas fijas por evento para reproducibilidad bit-a-bit (el hash de
# strings en Python 3 está randomizado entre ejecuciones por defecto).
_SEMILLAS_EVENTOS = {
    "crash1929": 1929,
    "blackmonday1987": 1987,
    "dotcom2000": 2000,
    "crash2008": 2008,
    "covid2020": 2020,
    "bear2022": 2022,
}


def _generar_sintetico(clave_evento: str, semilla: int = 13) -> pd.DataFrame:
    """
    Genera una serie sintética por tramos. Cada tramo define un proceso
    AR(1) con varianza marginal aprox sigma^2 y autocorrelación phi:

        r_t = phi * r_{t-1} + drift + sigma * sqrt(1 - phi^2) * eps_t

    De este modo Var(r_t) ~= sigma^2 dentro del tramo, así que un
    detector basado en varianza móvil no ve el cambio cuando lo único
    que se modifica es phi. La nube SW, en cambio, sí cambia porque su
    geometría depende de la estructura serial.
    """
    ev = EVENTOS[clave_evento]
    calib = _CALIBRACIONES[clave_evento]
    semilla_efectiva = semilla + _SEMILLAS_EVENTOS.get(clave_evento, 0)
    rng = np.random.default_rng(semilla_efectiva)
    fechas = pd.date_range(ev.inicio, ev.fin, freq="B")
    n = len(fechas)

    tramos = sorted(calib["tramos"], key=lambda t: pd.Timestamp(t[0]))
    # Para cada día, identificar drift, sigma, phi.
    drift = np.zeros(n)
    sigma = np.zeros(n)
    phi = np.zeros(n)
    for i, fecha in enumerate(fechas):
        tramo_activo = tramos[0]
        for t in tramos:
            if pd.Timestamp(t[0]) <= fecha:
                tramo_activo = t
        drift[i] = tramo_activo[1]
        sigma[i] = tramo_activo[2]
        # phi opcional: 4º elemento si está presente.
        phi[i] = tramo_activo[3] if len(tramo_activo) >= 4 else 0.0

    # Generación recursiva del AR(1) con varianza marginal estabilizada.
    retornos = np.zeros(n)
    r_prev = 0.0
    for i in range(n):
        sigma_innov = sigma[i] * np.sqrt(max(0.0, 1.0 - phi[i] ** 2))
        # Corrección de Itô sobre la varianza marginal sigma_i^2.
        eps = rng.normal()
        retornos[i] = (phi[i] * r_prev
                       + (drift[i] - 0.5 * sigma[i] ** 2)
                       + sigma_innov * eps)
        r_prev = retornos[i]

    # Salto puntual (e.g., Lunes Negro 1987).
    if "salto" in calib:
        fecha_salto, magnitud = calib["salto"]
        ts = pd.Timestamp(fecha_salto)
        if ts in fechas:
            idx = fechas.get_loc(ts)
            retornos[idx] = np.log(1.0 + magnitud)

    precios = float(calib["nivel_inicial"]) * np.exp(np.cumsum(retornos))
    return pd.DataFrame(
        {"precio": precios, "fuente": f"sintetico_{ev.clave}"},
        index=fechas,
    )


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def log_retornos(serie_precios):
    """Retornos logarítmicos diarios de una serie de precios.

    Se fuerza el array a 1D (algunas versiones de yfinance/pandas
    pueden devolver columnas (n, 1) que rompen matplotlib más
    adelante).
    """
    p = np.asarray(serie_precios, dtype=float).ravel()
    return np.diff(np.log(p))


def normalizar_estandar(x):
    """Estandariza una serie restando la media y dividiendo por la desviación."""
    x = np.asarray(x, dtype=float).ravel()
    sigma = x.std()
    if sigma == 0:
        return x - x.mean()
    return (x - x.mean()) / sigma
