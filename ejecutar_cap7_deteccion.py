"""
Regenera los 12 PDFs de los tres experimentos de detección de cambios
del Capítulo 7 (sección "Detección de cambios sobre series sintéticas").

Para cada uno de los tres escenarios:
  - Experimento 1: señal periódica -> ruido (cambio en t=250)
  - Experimento 2: cambio de frecuencia (período 40 -> 70 en t=250)
  - Experimento 3: ruido -> señal periódica (cambio en t=250)

se generan cuatro figuras en formato PDF (vectorial, mejor para la
imprenta del TFG):

  exp{N}_serie.pdf       — la serie temporal con la marca del cambio.
  exp{N}_distancias.pdf  — distancias bottleneck entre diagramas
                            consecutivos a lo largo de la serie,
                            con el umbral robusto mediana+4·MAD
                            superpuesto y la fecha de detección.
  exp{N}_nube_antes.pdf  — proyección PCA a 2D de la nube SW de una
                            ventana en el régimen previo al cambio.
  exp{N}_nube_despues.pdf— proyección PCA a 2D de la nube SW de una
                            ventana en el régimen posterior al cambio.

Parámetros del análisis (los mismos que documenta el Capítulo 7):
  T = 500 muestras
  cambio = 250
  ventana n = 80
  paso entre ventanas s = 5
  embedding M = 10, retardo tau = 3

Reproducibilidad: las semillas están fijadas dentro de `senales.py`
para cada generador.

Uso:
    cd codigo/
    python ejecutar_cap7_deteccion.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import senales as sn
import tda_minimo as tda
from detector import detector_robusto

try:
    from ripser import ripser
    from persim import bottleneck as bottleneck_persim
    USAR_RIPSER = True
    print("[cap7 deteccion] ripser detectado.")
except Exception:
    USAR_RIPSER = False
    print("[cap7 deteccion] ripser no disponible, usando tda_minimo.")


RAIZ = os.path.abspath(os.path.dirname(__file__))
DIR_FIGURAS = os.path.normpath(os.path.join(
    RAIZ, "..", "TFG_plantilla_POLITÉCNICA_overleaf",
    "Imagenes", "capitulo7"
))
DIR_RESULTADOS = os.path.normpath(os.path.join(RAIZ, "..", "resultados"))
os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)


# ---------------------------------------------------------------------------
# Wrappers de TDA
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


def secuencia_diagramas(x, n=80, paso=5, M=10, tau=3, max_pts=60):
    """
    Recorre la señal con ventana n, calcula la nube SW y su diagrama H_1.
    Devuelve (indices_inicio_ventana, diagramas, nubes).
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    indices, diagramas, nubes = [], [], []
    k = 0
    while k + n <= T:
        ventana = x[k:k + n]
        try:
            nube = tda.embedding_sw(ventana, M=M, tau=tau)
        except ValueError:
            k += paso
            continue
        if len(nube) > max_pts:
            idx = np.linspace(0, len(nube) - 1, max_pts).astype(int)
            nube = nube[idx]
        _, dgm1 = calcular_dgms(nube)
        indices.append(k)
        diagramas.append(dgm1)
        nubes.append(nube)
        k += paso
    return indices, diagramas, nubes


# ---------------------------------------------------------------------------
# Utilidades de figura
# ---------------------------------------------------------------------------

def guardar_pdf(fig, nombre):
    """Guarda un fig en formato PDF (vectorial, para la imprenta)."""
    ruta = os.path.join(DIR_FIGURAS, nombre + ".pdf")
    fig.savefig(ruta, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {ruta}")


def fig_serie(t, x, cambio, titulo, nombre_pdf):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, x, color="black", lw=0.8)
    ax.axvline(cambio, color="firebrick", ls="--", lw=1.0,
                label=f"cambio en t={cambio}")
    ax.set_xlabel("t (muestras)")
    ax.set_ylabel("x(t)")
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    guardar_pdf(fig, nombre_pdf)


def fig_distancias(indices, distancias, cambio, umbral, idx_detectado,
                    titulo, nombre_pdf):
    fig, ax = plt.subplots(figsize=(6, 3))
    fechas_diag = indices[:-1]
    ax.plot(fechas_diag, distancias, color="black", marker="o", ms=3, lw=0.7)
    ax.axhline(umbral, color="firebrick", ls="--", lw=0.7,
                label=f"umbral mediana+4MAD = {umbral:.3f}")
    ax.axvline(cambio, color="darkgreen", ls=":", lw=1.0,
                label=f"cambio real en t={cambio}")
    if idx_detectado is not None:
        ax.axvline(fechas_diag[idx_detectado], color="orange",
                    ls="--", lw=1.0, label="detección")
    ax.set_xlabel("inicio de la ventana (t)")
    ax.set_ylabel(r"$d_B$ entre diagramas consecutivos")
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    guardar_pdf(fig, nombre_pdf)


def fig_nube_pca(nube, titulo, color, nombre_pdf):
    """Proyecta a 2D por PCA y dibuja la nube."""
    centro = nube - nube.mean(axis=0)
    _, _, Vt = np.linalg.svd(centro, full_matrices=False)
    proy = centro @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(proy[:, 0], proy[:, 1], "o-", ms=3, color=color, lw=0.6,
             alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(titulo, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    guardar_pdf(fig, nombre_pdf)


# ---------------------------------------------------------------------------
# Pipeline por experimento
# ---------------------------------------------------------------------------

def ejecutar_experimento(nombre_corto, titulo_serie, t, x, cambio,
                          n=80, paso=5, M=10, tau=3, max_pts=60):
    print(f"\n[cap7 deteccion] === {titulo_serie} ===")

    # 1) Figura de la serie
    fig_serie(t, x, cambio,
               f"{titulo_serie}",
               f"{nombre_corto}_serie")

    # 2) Secuencia de diagramas
    indices, diagramas, nubes = secuencia_diagramas(
        x, n=n, paso=paso, M=M, tau=tau, max_pts=max_pts)
    print(f"  {len(diagramas)} ventanas analizadas")

    # 3) Distancias entre diagramas consecutivos
    distancias = np.array([
        bottleneck(diagramas[k], diagramas[k + 1])
        for k in range(len(diagramas) - 1)
    ])

    # 4) Detector robusto
    res = detector_robusto(distancias, k=4.0, frac_ref=0.30,
                            n_min_ref=10, burn_in=2, consecutivos=2)
    idx_det = res.indice_detectado
    umbral = res.umbral
    if idx_det is not None:
        t_det = indices[idx_det]
        print(f"  detección: t = {t_det} (cambio real en t = {cambio}, "
              f"Δ = {t_det - cambio:+d})")
    else:
        print(f"  NO detectado (umbral = {umbral:.3f})")

    fig_distancias(indices, distancias, cambio, umbral, idx_det,
                    f"{titulo_serie} — distancias bottleneck",
                    f"{nombre_corto}_distancias")

    # 5) Nubes antes y después
    # Elegimos una ventana cuyo final esté bien dentro del régimen previo
    # y otra cuyo inicio esté bien dentro del régimen posterior.
    centro_idx = None
    for j, k_inicio in enumerate(indices):
        if k_inicio + n <= cambio:
            centro_idx = j
    idx_antes = centro_idx if centro_idx is not None else 0

    idx_despues = None
    for j, k_inicio in enumerate(indices):
        if k_inicio >= cambio + 20:
            idx_despues = j
            break
    if idx_despues is None:
        idx_despues = len(indices) - 1

    fig_nube_pca(nubes[idx_antes],
                  f"Nube SW (PCA 2D)\nventana iniciada en t={indices[idx_antes]}",
                  "steelblue",
                  f"{nombre_corto}_nube_antes")
    fig_nube_pca(nubes[idx_despues],
                  f"Nube SW (PCA 2D)\nventana iniciada en t={indices[idx_despues]}",
                  "firebrick",
                  f"{nombre_corto}_nube_despues")

    return {
        "experimento": nombre_corto,
        "titulo": titulo_serie,
        "n_ventanas": int(len(diagramas)),
        "umbral": float(umbral),
        "cambio_real": int(cambio),
        "detectado_t": int(indices[idx_det]) if idx_det is not None else None,
        "delta": (int(indices[idx_det]) - cambio) if idx_det is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(20260625)
    resultados = []

    # ---- Experimento 1 ----
    t, x = sn.senal_periodica_a_ruido(T=500, periodo=40, sigma=0.6,
                                        cambio=250, semilla=2)
    resultados.append(ejecutar_experimento(
        "exp1",
        "Experimento 1: de señal periódica a ruido",
        t, x, cambio=250))

    # ---- Experimento 2 ----
    t, x = sn.senal_cambio_frecuencia(T=500, periodo1=40, periodo2=70,
                                        cambio=250)
    resultados.append(ejecutar_experimento(
        "exp2",
        "Experimento 2: cambio de frecuencia",
        t, x, cambio=250))

    # ---- Experimento 3 ----
    t, x = sn.senal_ruido_a_periodica(T=500, periodo=40, sigma=0.6,
                                        cambio=250, semilla=3)
    resultados.append(ejecutar_experimento(
        "exp3",
        "Experimento 3: de ruido a señal periódica",
        t, x, cambio=250))

    # CSV resumen
    import pandas as pd
    df = pd.DataFrame(resultados)
    ruta_csv = os.path.join(DIR_RESULTADOS, "cap7_deteccion_resumen.csv")
    df.to_csv(ruta_csv, index=False)
    print(f"\n[cap7 deteccion] CSV: {ruta_csv}")
    print(df.to_string(index=False))
    print("\n[cap7 deteccion] Hecho.")


if __name__ == "__main__":
    main()
