"""
Capítulo 8: aplicación al S&P 500 en torno al crash de 2008.

Pipeline:
1. Descargar (o reconstruir) la serie de cierre del S&P 500 entre
   junio de 2007 y junio de 2009.
2. Calcular los retornos logarítmicos diarios.
3. Aplicar el embedding por ventanas deslizantes localmente: para cada
   posición de una ventana de longitud n, generar la nube SW correspondiente
   y calcular su diagrama de persistencia.
4. Calcular la distancia bottleneck entre diagramas consecutivos.
5. Detectar el cambio de régimen mediante un umbral basado en la media y la
   desviación típica de los primeros valores.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datos_reales as dr
import tda_minimo as tda

try:
    from ripser import ripser
    from persim import bottleneck as bottleneck_persim
    USAR_RIPSER = True
    print("[cap8] ripser detectado.")
except Exception:
    USAR_RIPSER = False
    print("[cap8] ripser no disponible, usando tda_minimo.")


RAIZ = os.path.abspath(os.path.dirname(__file__))
DIR_FIGURAS = os.path.normpath(os.path.join(
    RAIZ, "..", "TFG_plantilla_POLITÉCNICA_overleaf",
    "Imágenes", "capitulo8"
))
DIR_RESULTADOS = os.path.normpath(os.path.join(RAIZ, "..", "resultados"))
os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)


def guardar(fig, nombre):
    ruta = os.path.join(DIR_FIGURAS, nombre + ".png")
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {ruta}")


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


def cargar_serie():
    """Devuelve DataFrame con la serie de cierre y la fuente."""
    df = dr.descargar_sp500_2008(inicio="2007-06-01", fin="2009-06-01")
    return df


def figura_serie_y_retornos(df):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(df.index, df["precio"].values, color="black", lw=1)
    axes[0].set_title(f"S&P 500 (cierre diario) — fuente: {df['fuente'].iloc[0]}",
                       fontsize=10)
    axes[0].set_ylabel("Nivel de cierre")
    axes[0].axvline(pd.Timestamp("2008-09-15"), color="firebrick",
                    ls="--", lw=0.8, label="Lehman (15-sep-2008)")
    axes[0].legend(fontsize=8)

    retornos = dr.log_retornos(df["precio"].values)
    axes[1].plot(df.index[1:], retornos, color="steelblue", lw=0.5)
    axes[1].set_title("Retornos logarítmicos diarios", fontsize=10)
    axes[1].set_ylabel("log-retornos")
    axes[1].set_xlabel("fecha")
    axes[1].axvline(pd.Timestamp("2008-09-15"), color="firebrick",
                    ls="--", lw=0.8)
    fig.tight_layout()
    guardar(fig, "sp500_serie_y_retornos")


def secuencia_diagramas(serie, n=80, paso=5, M=4, tau=3, max_pts=60):
    """
    Recorre la serie con ventana de longitud n, calcula la nube SW dentro
    de cada ventana y devuelve la lista de diagramas (H_1) y los índices
    iniciales de cada ventana.
    """
    serie = np.asarray(serie)
    T = len(serie)
    diagramas = []
    indices = []
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


def figura_distancias(df, indices, distancias, umbral, posicion_cambio_real,
                       posicion_detectada):
    fechas_diag = [df.index[i + len(df) // 100] for i in indices[:-1]]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fechas_diag, distancias, color="black", marker="o", ms=3, lw=0.8)
    ax.axhline(umbral, color="firebrick", ls="--", lw=0.8,
                label=f"umbral = mu + 3 sigma = {umbral:.4f}")
    ax.axvline(pd.Timestamp("2008-09-15"), color="darkgreen",
                ls=":", lw=1.0, label="Lehman (15-sep-2008)")
    if posicion_detectada is not None:
        ax.axvline(fechas_diag[posicion_detectada], color="orange",
                   ls="--", lw=1.0, label="cambio detectado")
    ax.set_title(
        "Distancia bottleneck entre diagramas consecutivos",
        fontsize=10)
    ax.set_ylabel("d_B")
    ax.set_xlabel("fecha")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    guardar(fig, "sp500_distancias_bottleneck")


def figura_diagrama_antes_despues(diagramas, indices, df, posicion_cambio):
    """Compara dos diagramas: uno en régimen estable y otro tras el cambio."""
    if posicion_cambio is None:
        return
    idx_estable = max(0, posicion_cambio - 8)
    idx_critico = min(len(diagramas) - 1, posicion_cambio + 2)
    dgm_estable = [p for p in diagramas[idx_estable] if np.isfinite(p[1])]
    dgm_critico = [p for p in diagramas[idx_critico] if np.isfinite(p[1])]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, dgm, titulo, k in zip(
        axes,
        [dgm_estable, dgm_critico],
        ["Ventana en régimen previo", "Ventana en régimen crisis"],
        [idx_estable, idx_critico],
    ):
        if dgm:
            arr = np.array(dgm)
            ax.scatter(arr[:, 0], arr[:, 1], color="firebrick", marker="^",
                       s=30)
            mx = arr[:, 1].max()
        else:
            mx = 0.05
        ax.plot([0, mx * 1.1], [0, mx * 1.1], color="gray", lw=0.8)
        ax.set_xlabel("nacimiento")
        ax.set_ylabel("muerte")
        fecha = df.index[indices[k]]
        ax.set_title(f"{titulo}\n(ventana inicia en {fecha.date()})", fontsize=10)
    fig.tight_layout()
    guardar(fig, "sp500_diagramas_antes_despues")


def main():
    np.random.seed(7)
    df = cargar_serie()
    print(f"[cap8] {len(df)} días de cotización. Fuente: {df['fuente'].iloc[0]}.")
    figura_serie_y_retornos(df)

    retornos = dr.log_retornos(df["precio"].values)
    retornos_estand = dr.normalizar_estandar(retornos)

    print("[cap8] Calculando secuencia de diagramas (puede tardar)...")
    indices, diagramas = secuencia_diagramas(
        retornos_estand, n=80, paso=10, M=4, tau=3, max_pts=60)
    print(f"[cap8] {len(diagramas)} diagramas calculados.")

    distancias = [bottleneck(diagramas[k], diagramas[k + 1])
                  for k in range(len(diagramas) - 1)]
    distancias = np.array(distancias)

    # Tramo de referencia: primer 25% de la señal
    n_ref = max(5, len(distancias) // 4)
    mu = distancias[:n_ref].mean()
    sigma = distancias[:n_ref].std() + 1e-12
    umbral = mu + 3 * sigma
    print(f"[cap8] mu={mu:.5f}, sigma={sigma:.5f}, umbral={umbral:.5f}")

    superados = np.where(distancias > umbral)[0]
    posicion_detectada = int(superados[0]) if len(superados) > 0 else None
    if posicion_detectada is not None:
        fecha_det = df.index[indices[posicion_detectada]]
        print(f"[cap8] Primer cambio detectado en ventana k={posicion_detectada}, "
              f"fecha aprox: {fecha_det.date()}")
    else:
        print("[cap8] Ningún valor superó el umbral.")

    figura_distancias(df, indices, distancias, umbral,
                       posicion_cambio_real=None,
                       posicion_detectada=posicion_detectada)
    figura_diagrama_antes_despues(diagramas, indices, df,
                                   posicion_detectada)

    # Tabla de resultados
    tabla = pd.DataFrame({
        "k": np.arange(len(distancias)),
        "fecha_inicio_ventana": [df.index[indices[k]].date()
                                  for k in range(len(distancias))],
        "distancia_bottleneck": distancias,
        "supera_umbral": distancias > umbral,
    })
    tabla.to_csv(os.path.join(DIR_RESULTADOS, "cap8_distancias.csv"),
                  index=False)
    print(f"[cap8] CSV: {DIR_RESULTADOS}/cap8_distancias.csv")
    print("[cap8] Listo.")


if __name__ == "__main__":
    main()
