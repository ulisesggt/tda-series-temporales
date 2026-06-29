"""
Comparación con un baseline clásico: detección de cambio de régimen por
varianza móvil de los retornos.

Para cada evento generamos en paralelo:
  - señal topológica (distancia bottleneck entre diagramas consecutivos)
  - señal clásica (varianza móvil de ventana del mismo tamaño)
y comparamos cuándo cada una cruza su propio umbral mu+3sigma.

El objetivo no es demostrar que TDA es mejor, sino mostrar honestamente
qué información aporta cada uno y dónde coinciden o divergen.
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
except Exception:
    USAR_RIPSER = False


RAIZ = os.path.abspath(os.path.dirname(__file__))
DIR_FIGURAS = os.path.normpath(os.path.join(
    RAIZ, "..", "TFG_plantilla_POLITÉCNICA_overleaf",
    "Imágenes", "capitulo8"
))
DIR_RESULTADOS = os.path.normpath(os.path.join(RAIZ, "..", "resultados"))
os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)


def _calcular_dgms(nube):
    if USAR_RIPSER:
        res = ripser(nube, maxdim=1)
        dgms = res["dgms"]
        return [tuple(p) for p in dgms[0]], [tuple(p) for p in dgms[1]]
    return tda.diagrama_persistencia(nube, dim_max=2)


def _bottleneck(a, b):
    if USAR_RIPSER:
        x = np.array([(c, d) for c, d in a if np.isfinite(d)]) if a \
            else np.empty((0, 2))
        y = np.array([(c, d) for c, d in b if np.isfinite(d)]) if b \
            else np.empty((0, 2))
        return float(bottleneck_persim(x, y))
    return tda.distancia_bottleneck(a, b)


def pipeline_tda(retornos, n, paso, M, tau, max_pts):
    diagramas, indices = [], []
    T = len(retornos)
    k = 0
    while k + n <= T:
        ventana = retornos[k:k + n]
        try:
            nube = tda.embedding_sw(ventana, M=M, tau=tau)
        except ValueError:
            k += paso
            continue
        if len(nube) > max_pts:
            idx = np.linspace(0, len(nube) - 1, max_pts).astype(int)
            nube = nube[idx]
        _, dgm1 = _calcular_dgms(nube)
        diagramas.append(dgm1)
        indices.append(k)
        k += paso
    distancias = np.array([
        _bottleneck(diagramas[i], diagramas[i + 1])
        for i in range(len(diagramas) - 1)
    ])
    return indices[:-1], distancias


def pipeline_varianza(retornos, n, paso):
    """Varianza móvil de ventana n con el mismo paso."""
    T = len(retornos)
    vals, indices = [], []
    k = 0
    while k + n <= T:
        vals.append(np.var(retornos[k:k + n]))
        indices.append(k)
        k += paso
    return indices, np.array(vals)


def analizar_baseline(clave, n=60, paso=8, M=4, tau=3, max_pts=40):
    ev = dr.EVENTOS[clave]
    df = dr.descargar_evento(clave)
    retornos = dr.log_retornos(df["precio"].values)
    retornos_est = dr.normalizar_estandar(retornos)

    idx_tda, dist_tda = pipeline_tda(retornos_est, n, paso, M, tau, max_pts)
    idx_var, val_var = pipeline_varianza(retornos_est, n, paso)

    # MISMO detector robusto para ambas señales: comparación justa.
    res_tda = detector_robusto(dist_tda, k=4.0, frac_ref=0.25,
                                n_min_ref=15, burn_in=3, consecutivos=2)
    res_var = detector_robusto(val_var, k=4.0, frac_ref=0.25,
                                n_min_ref=15, burn_in=3, consecutivos=2)
    u_tda = res_tda.umbral
    u_var = res_var.umbral

    fechas_tda = [df.index[i] for i in idx_tda]
    fechas_var = [df.index[i] for i in idx_var]

    det_tda = (fechas_tda[res_tda.indice_detectado]
               if res_tda.indice_detectado is not None else None)
    det_var = (fechas_var[res_var.indice_detectado]
               if res_var.indice_detectado is not None else None)

    return {
        "clave": clave,
        "nombre": ev.nombre,
        "fecha_ref": pd.Timestamp(ev.fecha_referencia),
        "fechas_tda": fechas_tda, "dist_tda": dist_tda, "u_tda": u_tda,
        "fechas_var": fechas_var, "val_var": val_var, "u_var": u_var,
        "det_tda": det_tda, "det_var": det_var,
    }


def figura_baseline(resultados):
    n = len(resultados)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, resultados):
        ax2 = ax.twinx()
        ax.plot(r["fechas_tda"], r["dist_tda"], color="firebrick",
                 lw=0.9, label="d_B (topológica)")
        ax.axhline(r["u_tda"], color="firebrick", ls="--", lw=0.6)
        ax2.plot(r["fechas_var"], r["val_var"], color="steelblue",
                  lw=0.7, alpha=0.7, label="var móvil (clásica)")
        ax2.axhline(r["u_var"], color="steelblue", ls="--", lw=0.5, alpha=0.6)
        ax.axvline(r["fecha_ref"], color="darkgreen", ls=":", lw=1.0)
        if r["det_tda"]:
            ax.axvline(r["det_tda"], color="firebrick", ls="-", lw=0.6,
                        alpha=0.6)
        if r["det_var"]:
            ax.axvline(r["det_var"], color="steelblue", ls="-", lw=0.6,
                        alpha=0.6)
        ax.set_title(r["nombre"], fontsize=9)
        ax.tick_params(axis="y", labelsize=7, colors="firebrick")
        ax2.tick_params(axis="y", labelsize=7, colors="steelblue")
        ax.set_ylabel("d_B", fontsize=8, color="firebrick")
        ax2.set_ylabel("var móvil", fontsize=8, color="steelblue")
        ax.tick_params(axis="x", labelsize=7)
    axes[-1].set_xlabel("fecha")
    fig.tight_layout()
    ruta = os.path.join(DIR_FIGURAS, "sp500_baseline_comparativa.png")
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[baseline] figura: {ruta}")


def main():
    np.random.seed(11)
    claves = ["crash1929", "blackmonday1987", "dotcom2000",
              "crash2008", "covid2020", "bear2022"]
    resultados = []
    for c in claves:
        print(f"[baseline] {c}...")
        r = analizar_baseline(c)
        resultados.append(r)
        print(f"  TDA: {r['det_tda']}, var: {r['det_var']}, "
              f"ref: {r['fecha_ref'].date()}")
    figura_baseline(resultados)

    # CSV resumen
    df = pd.DataFrame([{
        "evento": r["clave"],
        "fecha_ref": r["fecha_ref"].date(),
        "det_tda": r["det_tda"].date() if r["det_tda"] else None,
        "det_var": r["det_var"].date() if r["det_var"] else None,
        "delta_tda_dias": (r["det_tda"] - r["fecha_ref"]).days
                           if r["det_tda"] else None,
        "delta_var_dias": (r["det_var"] - r["fecha_ref"]).days
                           if r["det_var"] else None,
    } for r in resultados])
    df.to_csv(os.path.join(DIR_RESULTADOS, "cap8_baseline_resumen.csv"),
              index=False)
    print(df.to_string(index=False))
    print("[baseline] hecho.")


if __name__ == "__main__":
    main()
