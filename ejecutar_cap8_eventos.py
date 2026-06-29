"""
Capítulo 8 (ampliado): aplica la detección topológica de cambios de
régimen a una colección de eventos históricos del S&P 500.

Eventos cubiertos:
  - Crash de 1929 (Wall Street)
  - Lunes Negro 1987
  - Burbuja punto-com 2000
  - Crisis financiera 2008
  - Crash COVID-19 2020
  - Mercado bajista 2022

Para cada evento se sigue el mismo procedimiento (capítulo 7,
sección de criterios de detección): retornos logarítmicos diarios,
estandarización, sliding window con (n, paso, M, tau), diagramas H_1
en cada ventana, distancia bottleneck consecutiva y umbral
mu + 3*sigma sobre el primer cuarto de la señal.

Salidas:
  - Imágenes/capitulo8/sp500_eventos_<clave>.png (panel por evento)
  - Imágenes/capitulo8/sp500_eventos_comparativa.png (resumen)
  - resultados/cap8_eventos_resumen.csv
  - resultados/cap8_eventos_<clave>.csv
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datos_reales as dr
import tda_minimo as tda
from detector import detector_robusto

try:
    from ripser import ripser
    from persim import bottleneck as bottleneck_persim
    USAR_RIPSER = True
    print("[cap8 eventos] ripser detectado.")
except Exception:
    USAR_RIPSER = False
    print("[cap8 eventos] ripser no disponible, usando tda_minimo.")


RAIZ = os.path.abspath(os.path.dirname(__file__))
DIR_FIGURAS = os.path.normpath(os.path.join(
    RAIZ, "..", "TFG_plantilla_POLITÉCNICA_overleaf",
    "Imágenes", "capitulo8"
))
DIR_RESULTADOS = os.path.normpath(os.path.join(RAIZ, "..", "resultados"))
os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)


# ---------------------------------------------------------------------------
# Wrappers de TDA (mismos que en ejecutar_cap8.py)
# ---------------------------------------------------------------------------

def calcular_dgms(nube):
    if USAR_RIPSER:
        res = ripser(nube, maxdim=1)
        dgms = res["dgms"]
        dgm0 = [tuple(p) for p in dgms[0]]
        dgm1 = [tuple(p) for p in dgms[1]] if len(dgms) > 1 else []
        return dgm0, dgm1
    return tda.diagrama_persistencia(nube, dim_max=2)


def bottleneck(dgm1, dgm2):
    if USAR_RIPSER:
        a = np.array([(b, d) for b, d in dgm1 if np.isfinite(d)]) if dgm1 \
            else np.empty((0, 2))
        b = np.array([(b, d) for b, d in dgm2 if np.isfinite(d)]) if dgm2 \
            else np.empty((0, 2))
        return float(bottleneck_persim(a, b))
    return tda.distancia_bottleneck(dgm1, dgm2)


def secuencia_diagramas(serie, n=80, paso=10, M=4, tau=3, max_pts=60):
    serie = np.asarray(serie)
    T = len(serie)
    diagramas, indices = [], []
    k = 0
    while k + n <= T:
        ventana = serie[k:k + n]
        try:
            nube = tda.embedding_sw(ventana, M=M, tau=tau)
        except ValueError:
            k += paso
            continue
        if len(nube) > max_pts:
            idx = np.linspace(0, len(nube) - 1, max_pts).astype(int)
            nube = nube[idx]
        _, dgm1 = calcular_dgms(nube)
        diagramas.append(dgm1)
        indices.append(k)
        k += paso
    return indices, diagramas


# ---------------------------------------------------------------------------
# Análisis por evento
# ---------------------------------------------------------------------------

def analizar_evento(clave_evento, n=60, paso=8, M=4, tau=3, max_pts=40):
    ev = dr.EVENTOS[clave_evento]
    print(f"\n[cap8 eventos] === {ev.nombre} ===")
    df = dr.descargar_evento(clave_evento)
    print(f"[cap8 eventos] {len(df)} observaciones. Fuente: {df['fuente'].iloc[0]}.")

    retornos = dr.log_retornos(df["precio"].values)
    if len(retornos) < n + 2:
        print("[cap8 eventos] serie demasiado corta, se omite.")
        return None
    retornos = dr.normalizar_estandar(retornos)

    indices, diagramas = secuencia_diagramas(
        retornos, n=n, paso=paso, M=M, tau=tau, max_pts=max_pts)
    if len(diagramas) < 4:
        print("[cap8 eventos] muy pocos diagramas; se omite.")
        return None

    distancias = np.array([
        bottleneck(diagramas[k], diagramas[k + 1])
        for k in range(len(diagramas) - 1)
    ])
    # Detector robusto: mediana + 5*MAD, con burn-in y exigencia de 2
    # cruces consecutivos para confirmar la detección.
    res = detector_robusto(distancias, k=4.0, frac_ref=0.25,
                            n_min_ref=15, burn_in=3, consecutivos=2)
    mu = float(res.mediana_ref)
    sigma = float(res.mad_ref)
    umbral = float(res.umbral)
    pos_det = res.indice_detectado

    fechas_diag = [df.index[indices[k]] for k in range(len(distancias))]
    fecha_det = fechas_diag[pos_det] if pos_det is not None else None
    fecha_ref = pd.Timestamp(ev.fecha_referencia)

    if fecha_det is not None:
        delta = (fecha_det - fecha_ref).days
        print(f"[cap8 eventos] detección: {fecha_det.date()} "
              f"(ref: {fecha_ref.date()}, {delta:+d} días)")
    else:
        print("[cap8 eventos] no se superó el umbral.")

    # --- Tabla detallada por evento --------------------------------------
    tabla = pd.DataFrame({
        "fecha_inicio_ventana": [f.date() for f in fechas_diag],
        "distancia_bottleneck": distancias,
        "supera_umbral": distancias > umbral,
    })
    tabla.to_csv(
        os.path.join(DIR_RESULTADOS, f"cap8_eventos_{clave_evento}.csv"),
        index=False,
    )

    # --- Figura individual por evento ------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(df.index, df["precio"].values, color="black", lw=0.9)
    axes[0].set_title(
        f"{ev.nombre} — fuente: {df['fuente'].iloc[0]}", fontsize=10)
    axes[0].set_ylabel("nivel S&P 500")
    axes[0].axvline(fecha_ref, color="darkgreen", ls=":", lw=1.0,
                     label=f"referencia: {fecha_ref.date()}")
    if fecha_det is not None:
        axes[0].axvline(fecha_det, color="orange", ls="--", lw=1.0,
                         label=f"detección: {fecha_det.date()}")
    axes[0].legend(fontsize=8, loc="best")

    axes[1].plot(fechas_diag, distancias, color="black", marker="o",
                  ms=2.5, lw=0.7)
    axes[1].axhline(umbral, color="firebrick", ls="--", lw=0.8,
                     label=f"umbral mediana+5MAD = {umbral:.4f}")
    axes[1].axvline(fecha_ref, color="darkgreen", ls=":", lw=1.0)
    if fecha_det is not None:
        axes[1].axvline(fecha_det, color="orange", ls="--", lw=1.0)
    axes[1].set_ylabel("d_B entre diagramas")
    axes[1].set_xlabel("fecha")
    axes[1].legend(fontsize=8, loc="best")

    fig.tight_layout()
    ruta = os.path.join(DIR_FIGURAS, f"sp500_eventos_{clave_evento}.png")
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[cap8 eventos] figura: {ruta}")

    return {
        "evento": clave_evento,
        "nombre": ev.nombre,
        "fuente": df["fuente"].iloc[0],
        "fecha_ref": fecha_ref.date(),
        "fecha_det": fecha_det.date() if fecha_det is not None else None,
        "delta_dias": (fecha_det - fecha_ref).days
                       if fecha_det is not None else None,
        "umbral": float(umbral),
        "mu": float(mu),
        "sigma": float(sigma),
        "n_ventanas": int(len(distancias)),
    }


# ---------------------------------------------------------------------------
# Figura comparativa
# ---------------------------------------------------------------------------

def figura_comparativa(resumenes):
    n = len(resumenes)
    cols = 2
    filas = (n + 1) // 2
    fig, axes = plt.subplots(filas, cols, figsize=(11, 2.8 * filas))
    axes = np.array(axes).reshape(-1)
    for ax, r in zip(axes, resumenes):
        if r is None:
            ax.axis("off")
            continue
        clave = r["evento"]
        ev = dr.EVENTOS[clave]
        tabla = pd.read_csv(
            os.path.join(DIR_RESULTADOS, f"cap8_eventos_{clave}.csv"))
        fechas = pd.to_datetime(tabla["fecha_inicio_ventana"])
        ax.plot(fechas, tabla["distancia_bottleneck"], color="black",
                 marker="o", ms=2, lw=0.6)
        ax.axhline(r["umbral"], color="firebrick", ls="--", lw=0.6)
        ax.axvline(pd.Timestamp(ev.fecha_referencia), color="darkgreen",
                    ls=":", lw=0.7)
        if r["fecha_det"] is not None:
            ax.axvline(pd.Timestamp(str(r["fecha_det"])), color="orange",
                        ls="--", lw=0.7)
        ax.set_title(ev.nombre, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("d_B", fontsize=8)
    for ax in axes[len(resumenes):]:
        ax.axis("off")
    fig.tight_layout()
    ruta = os.path.join(DIR_FIGURAS, "sp500_eventos_comparativa.png")
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[cap8 eventos] figura comparativa: {ruta}")


def main():
    np.random.seed(42)
    claves = [
        "crash1929", "blackmonday1987", "dotcom2000",
        "crash2008", "covid2020", "bear2022",
    ]
    resumenes = []
    for clave in claves:
        r = analizar_evento(clave)
        if r is not None:
            resumenes.append(r)

    if resumenes:
        df_resumen = pd.DataFrame(resumenes)
        df_resumen.to_csv(
            os.path.join(DIR_RESULTADOS, "cap8_eventos_resumen.csv"),
            index=False,
        )
        print(f"\n[cap8 eventos] resumen:")
        print(df_resumen[["nombre", "fuente", "fecha_ref",
                          "fecha_det", "delta_dias"]].to_string(index=False))
        figura_comparativa(resumenes)

    print("\n[cap8 eventos] hecho.")


if __name__ == "__main__":
    main()
