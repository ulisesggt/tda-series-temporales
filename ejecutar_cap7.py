"""
Genera todas las figuras y tablas del Capítulo 7 (experimentos sintéticos).

Uso:
    python ejecutar_cap7.py

Las figuras se guardan en ../TFG_plantilla_POLITÉCNICA_overleaf/Imagenes/capitulo7/
y las tablas (CSV) en ../resultados/.

Se intenta usar ripser+persim; si no están disponibles, se cae a la
implementación propia en tda_minimo.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import senales
import tda_minimo as tda

# Backend "ripser" si está disponible; si no, usamos tda_minimo.
try:
    from ripser import ripser
    from persim import bottleneck as bottleneck_persim
    USAR_RIPSER = True
    print("[cap7] ripser detectado, se usará para los cálculos.")
except Exception:
    USAR_RIPSER = False
    print("[cap7] ripser no disponible, usando tda_minimo.")


# ---------- rutas ----------
RAIZ = os.path.abspath(os.path.dirname(__file__))
DIR_FIGURAS = os.path.normpath(os.path.join(
    RAIZ, "..", "TFG_plantilla_POLITÉCNICA_overleaf",
    "Imagenes", "capitulo7"
))
DIR_RESULTADOS = os.path.normpath(os.path.join(RAIZ, "..", "resultados"))
os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)


def guardar(fig, nombre):
    """Guarda figura en PNG con resolución razonable."""
    ruta = os.path.join(DIR_FIGURAS, nombre + ".png")
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {ruta}")


def calcular_dgms(nube):
    """Devuelve (dgm0, dgm1) usando ripser o tda_minimo según disponibilidad."""
    if USAR_RIPSER:
        res = ripser(nube, maxdim=1)
        dgms = res["dgms"]
        dgm0 = [tuple(p) for p in dgms[0]]
        dgm1 = [tuple(p) for p in dgms[1]] if len(dgms) > 1 else []
        return dgm0, dgm1
    return tda.diagrama_persistencia(nube, dim_max=2)


# ---------- 1. Señales ----------

def figura_senales():
    print("[cap7] Generando figuras de señales sintéticas...")
    t1, x1 = senales.senal_sinusoidal(T=600, periodo=40)
    t2, x2 = senales.senal_sinusoidal_con_ruido(T=600, periodo=40, sigma=0.3)
    t3, x3 = senales.senal_aleatoria(T=600, sigma=1.0)
    t4, x4 = senales.senal_dos_frecuencias(T=600, periodo1=40, periodo2=70)

    fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
    for ax, t, x, titulo in zip(
        axes,
        [t1, t2, t3, t4], [x1, x2, x3, x4],
        ["Sinusoidal pura (período 40)",
         "Sinusoidal con ruido (sigma=0.3)",
         "Ruido gaussiano",
         "Suma de dos cosenoidales (períodos 40 y 70)"],
    ):
        ax.plot(t, x, lw=0.8, color="black")
        ax.set_title(titulo, fontsize=10)
        ax.set_ylabel("x(t)")
    axes[-1].set_xlabel("t (muestras)")
    fig.tight_layout()
    guardar(fig, "senales_sinteticas")


# ---------- 2. Embedding y diagramas ----------

def figura_nube_y_diagrama(senal_func, nombre_archivo, titulo, M=8, tau=3,
                            max_puntos_nube=80):
    t, x = senal_func()
    nube = tda.embedding_sw(x, M=M, tau=tau)
    # Submuestreo para mantener el cálculo de VR puro en tiempos razonables.
    # En ripser este límite no es necesario.
    if len(nube) > max_puntos_nube:
        idx = np.linspace(0, len(nube) - 1, max_puntos_nube).astype(int)
        nube = nube[idx]
    dgm0, dgm1 = calcular_dgms(nube)

    # Proyección PCA a 2D para visualizar la nube
    nube_centrada = nube - nube.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(nube_centrada, full_matrices=False)
    proy = nube_centrada @ Vt[:2].T

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(t, x, lw=0.8, color="black")
    axes[0].set_title(f"Serie: {titulo}", fontsize=10)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x(t)")

    axes[1].scatter(proy[:, 0], proy[:, 1], s=8, color="steelblue", alpha=0.6)
    axes[1].set_title(f"Nube SW (M={M}, tau={tau}), PCA 2D", fontsize=10)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].axis("equal")

    # Diagrama de persistencia H_0 y H_1
    todos = [(b, d) for b, d in dgm0 if np.isfinite(d)] + list(dgm1)
    if todos:
        max_val = max(d for _, d in todos if np.isfinite(d))
    else:
        max_val = 1.0
    axes[2].plot([0, max_val * 1.1], [0, max_val * 1.1], color="gray", lw=0.8)
    if dgm0:
        h0 = np.array([(b, d) for b, d in dgm0 if np.isfinite(d)])
        if len(h0):
            axes[2].scatter(h0[:, 0], h0[:, 1], color="black",
                           marker="o", s=18, label="H_0")
    if dgm1:
        h1 = np.array([(b, d) for b, d in dgm1 if np.isfinite(d)])
        if len(h1):
            axes[2].scatter(h1[:, 0], h1[:, 1], color="firebrick",
                           marker="^", s=28, label="H_1")
    axes[2].set_title("Diagrama de persistencia", fontsize=10)
    axes[2].set_xlabel("nacimiento")
    axes[2].set_ylabel("muerte")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    guardar(fig, nombre_archivo)
    return dgm0, dgm1


# ---------- 3. SW1PerS comparado ----------

def figura_comparacion_sw1pers():
    print("[cap7] Comparando SW1PerS para varias señales...")
    casos = {
        "Sinusoidal": senales.senal_sinusoidal(T=400, periodo=40)[1],
        "Sin+ruido (0.1)": senales.senal_sinusoidal_con_ruido(
            T=400, periodo=40, sigma=0.1, semilla=10)[1],
        "Sin+ruido (0.3)": senales.senal_sinusoidal_con_ruido(
            T=400, periodo=40, sigma=0.3, semilla=11)[1],
        "Sin+ruido (0.6)": senales.senal_sinusoidal_con_ruido(
            T=400, periodo=40, sigma=0.6, semilla=12)[1],
        "Aleatoria": senales.senal_aleatoria(T=400, semilla=13)[1],
        "Dos frecuencias": senales.senal_dos_frecuencias(
            T=400, periodo1=40, periodo2=70)[1],
    }
    scores = {}
    M, tau = 8, 3
    max_pts = 80
    for nombre, x in casos.items():
        nube = tda.embedding_sw(x, M=M, tau=tau)
        if len(nube) > max_pts:
            idx = np.linspace(0, len(nube) - 1, max_pts).astype(int)
            nube = nube[idx]
        _, dgm1 = calcular_dgms(nube)
        p_max = tda.persistencia_maxima(dgm1)
        cota = 2 * np.sin(np.pi / (M + 2))
        scores[nombre] = p_max / cota if cota > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 4))
    nombres = list(scores.keys())
    valores = [scores[n] for n in nombres]
    ax.bar(nombres, valores, color="steelblue", edgecolor="black")
    ax.set_ylabel("SW1PerS normalizado")
    ax.set_title(
        f"Persistencia máxima en $H_1$ (normalizada) "
        f"para distintas señales (M={M}, tau={tau})", fontsize=10)
    ax.set_ylim(0, max(valores) * 1.15 + 0.05)
    for x_b, v in zip(nombres, valores):
        ax.text(x_b, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    guardar(fig, "sw1pers_comparacion")

    df = pd.DataFrame({"senal": nombres, "sw1pers_norm": valores})
    df.to_csv(os.path.join(DIR_RESULTADOS, "cap7_sw1pers.csv"), index=False)
    return df


# ---------- 4. Prueba de parámetros ----------

def figura_prueba_parametros():
    print("[cap7] Prueba de parámetros M y tau...")
    t, x = senales.senal_sinusoidal(T=400, periodo=40)
    Ms = [4, 6, 8, 10, 12]
    taus = [1, 2, 3, 5, 7]
    matriz = np.zeros((len(Ms), len(taus)))
    max_pts = 80
    for i, M in enumerate(Ms):
        for j, tau in enumerate(taus):
            try:
                nube = tda.embedding_sw(x, M=M, tau=tau)
                if len(nube) > max_pts:
                    idx = np.linspace(0, len(nube) - 1, max_pts).astype(int)
                    nube = nube[idx]
                _, dgm1 = calcular_dgms(nube)
                p_max = tda.persistencia_maxima(dgm1)
                cota = 2 * np.sin(np.pi / (M + 2))
                matriz[i, j] = p_max / cota if cota > 0 else 0.0
            except ValueError:
                matriz[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matriz, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels(taus)
    ax.set_yticks(range(len(Ms)))
    ax.set_yticklabels(Ms)
    ax.set_xlabel("retardo tau")
    ax.set_ylabel("dimensión M")
    ax.set_title(
        "SW1PerS normalizado para cos(2 pi t / 40)\n"
        "según parámetros del embedding", fontsize=10)
    for i in range(len(Ms)):
        for j in range(len(taus)):
            if not np.isnan(matriz[i, j]):
                ax.text(j, i, f"{matriz[i, j]:.2f}",
                       ha="center", va="center",
                       color="white" if matriz[i, j] < 0.5 else "black",
                       fontsize=8)
    fig.colorbar(im, ax=ax, label="SW1PerS norm.")
    fig.tight_layout()
    guardar(fig, "prueba_parametros")

    df = pd.DataFrame(matriz, index=[f"M={m}" for m in Ms],
                      columns=[f"tau={t}" for t in taus])
    df.to_csv(os.path.join(DIR_RESULTADOS, "cap7_parametros.csv"))


def main():
    np.random.seed(0)
    figura_senales()

    print("[cap7] Embedding + diagrama: sinusoidal pura")
    figura_nube_y_diagrama(
        lambda: senales.senal_sinusoidal(T=600, periodo=40),
        "nube_diagrama_sinusoidal", "cos(2 pi t / 40)")

    print("[cap7] Embedding + diagrama: sinusoidal con ruido")
    figura_nube_y_diagrama(
        lambda: senales.senal_sinusoidal_con_ruido(T=600, periodo=40,
                                                    sigma=0.3, semilla=21),
        "nube_diagrama_ruido", "cos(2 pi t / 40) + ruido")

    print("[cap7] Embedding + diagrama: aleatoria")
    figura_nube_y_diagrama(
        lambda: senales.senal_aleatoria(T=600, sigma=1.0, semilla=22),
        "nube_diagrama_aleatoria", "ruido gaussiano")

    print("[cap7] Embedding + diagrama: dos frecuencias")
    figura_nube_y_diagrama(
        lambda: senales.senal_dos_frecuencias(T=600, periodo1=40, periodo2=70),
        "nube_diagrama_dos_frec", "cos(2 pi t / 40) + cos(2 pi t / 70)")

    figura_comparacion_sw1pers()
    figura_prueba_parametros()
    print("[cap7] Listo.")


if __name__ == "__main__":
    main()
