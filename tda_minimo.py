"""
Módulo TDA mínimo en numpy puro.

Implementa Vietoris-Rips para dimensión 1 y la reducción matricial sobre F_2
necesaria para calcular el diagrama de persistencia. La idea no es competir con
ripser ni con gudhi en velocidad, sino disponer de una implementación
autocontenida con la que ejecutar los experimentos del TFG en entornos donde
las librerías especializadas no estén instaladas. Para producción se
recomienda usar ripser (ver código principal).

Las cadenas se almacenan como tuplas de índices ordenados de vértices.
Los coeficientes son módulo 2: las cadenas se representan como conjuntos de
símplices.
"""

import numpy as np
from itertools import combinations


def matriz_distancias(puntos):
    """Calcula la matriz de distancias euclídeas entre filas de `puntos`."""
    diferencias = puntos[:, None, :] - puntos[None, :, :]
    return np.sqrt(np.sum(diferencias ** 2, axis=-1))


def filtracion_vietoris_rips(puntos, dim_max=2, radio_max=None):
    """
    Construye la filtración de Vietoris-Rips de la nube `puntos`.

    Devuelve una lista de tuplas (símplice, valor_filtración) ordenadas por
    valor de filtración creciente y, dentro de un mismo valor, por dimensión
    creciente. Un símplice se representa como una tupla ordenada de índices.

    Parámetros
    ----------
    puntos    : array (n_puntos, d), nube de puntos en R^d.
    dim_max   : dimensión máxima de los símplices a generar.
    radio_max : valor máximo de filtración considerado. Si es None, se usa el
                diámetro de la nube.
    """
    n = len(puntos)
    D = matriz_distancias(puntos)
    if radio_max is None:
        radio_max = float(D.max())

    simplices = []
    # 0-símplices (vértices) entran en filtración 0
    for i in range(n):
        simplices.append(((i,), 0.0))

    # 1-símplices: aristas, filtración = distancia
    for i, j in combinations(range(n), 2):
        if D[i, j] <= radio_max:
            simplices.append(((i, j), float(D[i, j])))

    # k-símplices para k >= 2: filtración = mayor distancia entre pares
    for k in range(2, dim_max + 1):
        for combinacion in combinations(range(n), k + 1):
            d_max = max(D[a, b] for a, b in combinations(combinacion, 2))
            if d_max <= radio_max:
                simplices.append((tuple(combinacion), float(d_max)))

    # Orden compatible con caras: primero por valor de filtración, después por
    # dimensión (importante para que toda cara aparezca antes que su cofaz).
    simplices.sort(key=lambda s: (s[1], len(s[0])))
    return simplices


def _caras_codimension_uno(sigma):
    """Devuelve las caras de codimensión uno de un símplice."""
    return [sigma[:i] + sigma[i + 1:] for i in range(len(sigma))]


def calcular_persistencia(simplices):
    """
    Implementa la reducción matricial estándar sobre F_2.

    Devuelve dos listas de pares (b, d) por dimensión: una para H_0 y otra
    para H_1. Las parejas no apareadas corresponden a clases que sobreviven
    al final de la filtración y se representan con muerte = +infinito.
    """
    indice = {sigma: i for i, (sigma, _) in enumerate(simplices)}
    dim = {sigma: len(sigma) - 1 for sigma, _ in simplices}
    valor = {sigma: f for sigma, f in simplices}

    # Cada columna se almacena como conjunto de índices (filas con 1).
    columnas = []
    for sigma, _ in simplices:
        if len(sigma) == 1:
            columnas.append(set())
        else:
            caras = _caras_codimension_uno(sigma)
            col = {indice[c] for c in caras}
            columnas.append(col)

    bajos = {}  # fila_baja -> columna que la tiene como low
    pares = {}  # nacer -> morir (índices en `simplices`)

    for j in range(len(simplices)):
        while columnas[j]:
            low = max(columnas[j])
            if low in bajos:
                k = bajos[low]
                columnas[j] ^= columnas[k]  # suma en F_2
            else:
                bajos[low] = j
                pares[low] = j
                break

    diagrama_h0 = []
    diagrama_h1 = []
    nacedores = set(pares.keys())
    morideros = set(pares.values())

    # Parejas finitas
    for nacer, morir in pares.items():
        d_nacer = dim[simplices[nacer][0]]
        b = valor[simplices[nacer][0]]
        d = valor[simplices[morir][0]]
        if d <= b:
            continue
        if d_nacer == 0:
            diagrama_h0.append((b, d))
        elif d_nacer == 1:
            diagrama_h1.append((b, d))

    # Clases inmortales: símplices cuya columna quedó vacía y que no son
    # extremo de muerte de ninguna pareja anterior.
    for i, (sigma, f) in enumerate(simplices):
        if i in nacedores or i in morideros:
            continue
        if columnas[i]:
            continue  # columna no nula tras reducción: no debería pasar
        d_sigma = dim[sigma]
        if d_sigma == 0:
            diagrama_h0.append((f, np.inf))
        elif d_sigma == 1:
            diagrama_h1.append((f, np.inf))

    return diagrama_h0, diagrama_h1


def diagrama_persistencia(puntos, dim_max=2, radio_max=None):
    """Atajo: filtra y calcula persistencia, devuelve (dgm_h0, dgm_h1)."""
    simplices = filtracion_vietoris_rips(puntos, dim_max=dim_max,
                                         radio_max=radio_max)
    return calcular_persistencia(simplices)


def persistencia_maxima(diagrama):
    """Devuelve max(d-b) sobre los puntos finitos del diagrama, o 0 si no hay."""
    finitos = [(b, d) for b, d in diagrama if np.isfinite(d)]
    if not finitos:
        return 0.0
    return max(d - b for b, d in finitos)


def distancia_bottleneck(dgm1, dgm2):
    """
    Distancia bottleneck entre dos diagramas, considerando solo puntos finitos.

    Se utiliza un algoritmo de búsqueda binaria sobre el radio crítico,
    suficiente para conjuntos pequeños como los que aparecen en los
    experimentos del TFG. Para diagramas grandes conviene usar persim.
    """
    P = [(b, d) for b, d in dgm1 if np.isfinite(d)]
    Q = [(b, d) for b, d in dgm2 if np.isfinite(d)]

    if not P and not Q:
        return 0.0
    if not P or not Q:
        otro = P if P else Q
        return max((d - b) / 2 for b, d in otro)

    # Distancias entre puntos no diagonales
    def d_inf(p, q):
        return max(abs(p[0] - q[0]), abs(p[1] - q[1]))

    # Distancia de un punto a la diagonal en norma infinito
    def d_diag(p):
        return (p[1] - p[0]) / 2

    candidatos = set()
    for p in P:
        candidatos.add(d_diag(p))
        for q in Q:
            candidatos.add(d_inf(p, q))
    for q in Q:
        candidatos.add(d_diag(q))

    candidatos = sorted(candidatos)

    def factible(r):
        # Algoritmo greedy: emparejar P con Q (o con diagonal) usando radio r
        usados_q = set()
        for p in P:
            asignado = False
            if d_diag(p) <= r:
                asignado = True
            else:
                for j, q in enumerate(Q):
                    if j in usados_q:
                        continue
                    if d_inf(p, q) <= r:
                        usados_q.add(j)
                        asignado = True
                        break
            if not asignado:
                return False
        for j, q in enumerate(Q):
            if j in usados_q:
                continue
            if d_diag(q) > r:
                return False
        return True

    lo, hi = 0, len(candidatos) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if factible(candidatos[mid]):
            hi = mid
        else:
            lo = mid + 1
    return candidatos[lo]


# -----------------------------------------------------------------------------
# Sliding Window embedding
# -----------------------------------------------------------------------------

def embedding_sw(serie, M, tau):
    """
    Embedding por ventana deslizante de la serie.

    serie : array 1D
    M     : entero, dimensión de embedding (vector resultante en R^{M+1})
    tau   : entero, retardo en número de muestras

    Devuelve una matriz (n_vectores, M+1) con las nubes de retardos.
    """
    serie = np.asarray(serie)
    N = len(serie) - M * tau
    if N <= 0:
        raise ValueError(
            f"Serie demasiado corta: len(serie)={len(serie)}, "
            f"se necesita > M*tau = {M*tau}"
        )
    return np.stack([serie[i: i + N] for i in range(0, M * tau + 1, tau)], axis=1)


def sw1pers(serie, M, tau, radio_max=None, normalizar=True):
    """
    Score SW1PerS: persistencia máxima en H_1 del embedding de la serie.

    Si `normalizar=True`, se divide por la cota teórica máxima 2 sin(pi/(M+2)),
    correspondiente al caso ideal de una circunferencia inscrita en un
    polígono regular.
    """
    nube = embedding_sw(serie, M, tau)
    _, dgm1 = diagrama_persistencia(nube, dim_max=2, radio_max=radio_max)
    p_max = persistencia_maxima(dgm1)
    if normalizar:
        cota = 2 * np.sin(np.pi / (M + 2))
        return p_max / cota if cota > 0 else 0.0
    return p_max
