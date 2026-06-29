"""
Figuras adicionales para el Capítulo 7:

1) Barcode (códigos de barras de persistencia) para una señal sinusoidal.
2) Curva de persistencia máxima H_1 frente al nivel de ruido aditivo.
3) Mapa de calor del score SW1PerS sobre la rejilla (M, tau) refinada.
4) Comparación de embeddings en 2D (proyección PCA) para señales
   periódicas, ruidosas y aleatorias.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import senales as sn
import tda_minimo as tda


RAIZ = os.path.abspath(os.path.dirname(__file__))
DIR_FIGURAS = os.path.normpath(os.path.join(
    RAIZ, "..", "TFG_plantilla_POLITÉCNICA_overleaf",
    "Imágenes", "capitulo7"
))
DIR_RESULTADOS = os.path.normpath(os.path.join(RAIZ, "..", "resultados"))
os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)


def guardar(fig, nombre):
    ruta = os.path.join(DIR_FIGURAS, nombre + ".png")
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {ruta}")


# ---------------------------------------------------------------------------
# 1) Barcode
# ---------------------------------------------------------------------------

def fig_barcode():
    """Códigos de barras H_0 y H_1 para una nube generada con SW."""
    x = sn.senal_sinusoidal(T=200, periodo=25)[1]
    nube = tda.embedding_sw(x, M=10, tau=4)
    if len(nube) > 60:
        idx = np.linspace(0, len(nube) - 1, 60).astype(int)
        nube = nube[idx]
    dgm0, dgm1 = tda.diagrama_persistencia(nube, dim_max=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                              gridspec_kw={"width_ratios": [1, 1]})

    # H_0
    pares0 = sorted(dgm0, key=lambda p: p[1] if np.isfinite(p[1]) else 1e9)
    for i, (b, d) in enumerate(pares0):
        if np.isfinite(d):
            axes[0].plot([b, d], [i, i], color="steelblue", lw=2)
        else:
            axes[0].plot([b, max(p[1] for p in pares0 if np.isfinite(p[1]))],
                          [i, i], color="steelblue", lw=2)
    axes[0].set_title(r"Barcode $H_0$ — componentes conexas", fontsize=10)
    axes[0].set_xlabel("escala de filtración")
    axes[0].set_ylabel("clase")

    # H_1
    pares1 = sorted(dgm1, key=lambda p: -(p[1] - p[0]))
    for i, (b, d) in enumerate(pares1):
        axes[1].plot([b, d], [i, i], color="firebrick", lw=2)
    axes[1].set_title(r"Barcode $H_1$ — agujeros 1-dim", fontsize=10)
    axes[1].set_xlabel("escala de filtración")

    fig.suptitle("Códigos de barras de la nube SW de una señal sinusoidal",
                  fontsize=11)
    fig.tight_layout()
    guardar(fig, "barcode_sinusoidal")


# ---------------------------------------------------------------------------
# 2) Persistencia H_1 vs ruido
# ---------------------------------------------------------------------------

def fig_persistencia_vs_ruido():
    """Para varias amplitudes de ruido, mide la persistencia máxima H_1."""
    niveles = np.linspace(0.0, 1.5, 12)
    persistencias = []
    rng = np.random.default_rng(0)
    for sigma in niveles:
        x = sn.senal_sinusoidal(T=300, periodo=25)[1]
        if sigma > 0:
            x = x + rng.normal(scale=sigma, size=len(x))
        nube = tda.embedding_sw(x, M=10, tau=4)
        if len(nube) > 60:
            idx = np.linspace(0, len(nube) - 1, 60).astype(int)
            nube = nube[idx]
        _, dgm1 = tda.diagrama_persistencia(nube, dim_max=2)
        persistencias.append(tda.persistencia_maxima(dgm1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(niveles, persistencias, color="black", marker="o", lw=1.0)
    ax.set_xlabel(r"desviación típica del ruido $\sigma$")
    ax.set_ylabel(r"persistencia máxima $H_1$")
    ax.set_title("Degradación de la señal topológica con el ruido")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    guardar(fig, "persistencia_vs_ruido")

    tabla = pd.DataFrame({"sigma_ruido": niveles,
                           "persistencia_H1": persistencias})
    tabla.to_csv(os.path.join(DIR_RESULTADOS, "cap7_persistencia_ruido.csv"),
                  index=False)


# ---------------------------------------------------------------------------
# 3) Mapa de calor SW1PerS sobre (M, tau)
# ---------------------------------------------------------------------------

def fig_heatmap_parametros():
    x = sn.senal_sinusoidal(T=400, periodo=20)[1]
    Ms = np.arange(4, 16)
    taus = np.arange(1, 10)
    grid = np.zeros((len(Ms), len(taus)))
    for i, M in enumerate(Ms):
        for j, tau in enumerate(taus):
            try:
                nube = tda.embedding_sw(x, M=int(M), tau=int(tau))
            except ValueError:
                grid[i, j] = np.nan
                continue
            if len(nube) > 50:
                idx = np.linspace(0, len(nube) - 1, 50).astype(int)
                nube = nube[idx]
            _, dgm1 = tda.diagrama_persistencia(nube, dim_max=2)
            grid[i, j] = tda.persistencia_maxima(dgm1)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(grid, origin="lower", aspect="auto",
                    extent=[taus[0] - 0.5, taus[-1] + 0.5,
                            Ms[0] - 0.5, Ms[-1] + 0.5],
                    cmap="viridis")
    ax.set_xlabel(r"retardo $\tau$")
    ax.set_ylabel(r"dimensión de embedding $M$")
    ax.set_title(r"Persistencia máxima $H_1$ sobre la rejilla $(M, \tau)$")
    plt.colorbar(im, ax=ax, label=r"persistencia $H_1$")
    fig.tight_layout()
    guardar(fig, "heatmap_M_tau")

    tabla = pd.DataFrame(grid, index=[f"M={m}" for m in Ms],
                          columns=[f"tau={t}" for t in taus])
    tabla.to_csv(os.path.join(DIR_RESULTADOS, "cap7_heatmap_M_tau.csv"))


# ---------------------------------------------------------------------------
# 4) Embeddings 2D por PCA
# ---------------------------------------------------------------------------

def fig_pca_embeddings():
    """Proyecta las nubes SW a 2D mediante PCA simple (centrado + SVD)."""
    señales = {
        "Sinusoidal pura": sn.senal_sinusoidal(T=300, periodo=25)[1],
        "Sinusoidal + ruido":
            sn.senal_sinusoidal_con_ruido(T=300, periodo=25, sigma=0.5)[1],
        "Dos frecuencias":
            sn.senal_dos_frecuencias(T=300, periodo1=25, periodo2=43)[1],
        "Ruido blanco": sn.senal_aleatoria(T=300)[1],
    }
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, (titulo, x) in zip(axes.flatten(), señales.items()):
        nube = tda.embedding_sw(x, M=10, tau=4)
        centro = nube - nube.mean(axis=0)
        _, _, Vt = np.linalg.svd(centro, full_matrices=False)
        proy = centro @ Vt[:2].T
        ax.plot(proy[:, 0], proy[:, 1], "o-", ms=2.5, color="firebrick",
                 lw=0.5, alpha=0.7)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Proyección PCA de las nubes SW", fontsize=11)
    fig.tight_layout()
    guardar(fig, "pca_embeddings")


def main():
    np.random.seed(2024)
    print("[cap7 extra] 1/4 barcode")
    fig_barcode()
    print("[cap7 extra] 2/4 persistencia vs ruido")
    fig_persistencia_vs_ruido()
    print("[cap7 extra] 3/4 heatmap (M, tau)")
    fig_heatmap_parametros()
    print("[cap7 extra] 4/4 PCA embeddings")
    fig_pca_embeddings()
    print("[cap7 extra] hecho.")


if __name__ == "__main__":
    main()
