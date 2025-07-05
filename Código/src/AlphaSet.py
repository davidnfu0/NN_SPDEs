import numpy as np
import itertools as itt
import pickle as pkl
from src.Alpha import Alpha
from itertools import combinations_with_replacement


def int_g(T, j, t_init, t_end):
    """
    Calcula la integral entre t_init y t_end de la función g, donde la función es el elemento j-ésimo
    de la base de L2(0, T) propuesta en el paper.

    Esta función evalúa la integral de un término específico de la serie de la base ortonormal utilizada en la
    expansión en caos de Wiener. La función `g` depende de j, que selecciona el término específico de la base.

    Parameters:
    T (float): El tiempo total de la expansión en caos de Wiener.
    j (int): El índice del término en la base de L2(0, T).
    t_init (float): El tiempo inicial para la integral.
    t_end (float): El tiempo final para la integral.

    Returns:
    float: El valor de la integral entre `t_init` y `t_end` para el j-ésimo término de la base.

    Notes:
    La integral se evalúa en el intervalo [t_init, t_end] y no en [0, T], con el objetivo de calcular
    los incrementos del movimiento Browniano en cada paso de tiempo para luego poder reconstruirlo.
    """
    if j == 1:
        return (t_end - t_init) / np.sqrt(T)
    else:
        n = j - 1
        coef = np.sqrt(2 * T) / (n * np.sqrt(np.pi))
        angle_end = np.pi * n * t_end / T
        angle_init = np.pi * n * t_init / T
        return coef * (np.sin(angle_end) - np.sin(angle_init))


class AlphaSet:
    """
    Clase que almacena y calcula las combinaciones de alphas, los valores de los polinomios de Wick,
    los incrementos del movimiento Browniano y evalúa los caminos del movimiento Browniano. La clase representa
    el conjunto J_{I, J, K} del paper y permite realizar cálculos relacionados con la expansión en caos de Wiener.

    Attributes:
    I (int): Dimensión del movimiento Browniano considerado.
    J (int): Número de términos en la expansión de caos de Wiener.
    K (int): Grado máximo de los polinomios de Wick.
    n (int): Número de puntos en el tiempo.
    T (float): Tiempo total del movimiento Browniano.
    t (np.array): Arreglo con los puntos de tiempo generados.
    alphas (dict): Diccionario que almacena las instancias de Alpha agrupadas por su grado.
    wick_values (list): Lista de diccionarios con los valores de los polinomios de Wick.
    brownian_increments (list): Lista con los incrementos del movimiento Browniano para cada normal.
    normals (list): Lista de normales con las que se evalúan los polinomios de Wick.

    Methods:
    calculate_card(): Calcula el número total de alphas generados.
    calculate_alphas(): Genera todas las combinaciones posibles de alphas.
    calculate_wick_values(): Calcula los valores de los polinomios de Wick para cada normal y alpha.
    calculate_increments(): Calcula los incrementos del movimiento Browniano para cada normal.
    evaluate_paths(): Evalúa los caminos del movimiento Browniano a partir de los incrementos.
    save_alpha_set(filename): Guarda el alpha set en un archivo.
    _build_G(): Construye y guarda la matriz G[k,j] = ∫_{t_k}^{t_{k+1}} g_j de forma vectorizada.
    """

    def __init__(self, I: int, J: int, K: int, n: int, T: float) -> None:
        """
        Inicializa la clase AlphaSet con los parámetros dados.

        Parameters:
        I (int): Dimensión del movimiento Browniano considerado.
        J (int): Número de términos en la expansión de caos de Wiener.
        K (int): Grado máximo de los polinomios de Wick.
        n (int): Número de puntos en el tiempo.
        T (float): Tiempo total del movimiento Browniano.
        """
        self.I = I
        self.J = J
        self.K = K
        self.n = n
        self.T = T
        self.t = np.linspace(0, T, n)
        self.alphas = {k: [] for k in range(K + 1)}
        self.wick_values = []  # Almacena los valores de Wick
        self.brownian_increments = []
        self.brownian_paths = []
        self.normals = []
        self._G = None # matriz G[k,j]

    def calculate_card(self):
        """
        Calcula el número total de alphas generados.

        Returns:
        int: El número total de alphas generados.
        """
        return sum([len(self.alphas[k]) for k in range(self.K + 1)])

    def calculate_alphas(self):
        """
        Genera todas las combinaciones posibles de alphas y calcula sus factoriales y constantes de Wick para cada uno de ellos, almacenándolo en el diccionario.

        Este método genera las combinaciones posibles de alphas según los valores de I, J y K, y luego
        calcula el factorial y la constante de Wick para cada combinación generada. Finalmente, almacena
        las instancias de la clase Alpha en el diccionario `self.alphas`, agrupadas por el grado del polinomio.
        """
        # Reinicializa el diccionario
        self.alphas = {k: [] for k in range(self.K + 1)}

        i, j, k = self.I, self.J, self.K

        for alpha_matrix in itt.product(range(k + 1), repeat = i * j):
            total = sum(alpha_matrix)
            if total <= k:
                alpha = Alpha(i, j, np.reshape(alpha_matrix, (i, j)))
                alpha.calculate_factorial()
                alpha.calculate_wick_constant()

                self.alphas[total].append(alpha)

        # Orden canónico para correspondencia por índice
        for k in range(k + 1):
            self.alphas[k].sort(key=lambda a: tuple(a.values.flatten()))

    def calculate_wick_values(self):
        """
        Calcula los valores de los polinomios de Wick para las realizaciones de las normales estándar y cada alpha.

        Este método recorre todas las matrices de normales y para cada matriz calcula los valores de los polinomios de Wick
        para cada alpha, agrupados por el grado del polinomio.
        Nota: En particular, cada normal es una matriz de tamaño (I, J), que permite construir los movimientos brownianos de dimensión I y número de términos(truncamiento) J.
        """
        for normal in self.normals:
            wick_values_dict = {}
            for k in range(self.K + 1):
                wick_values_dict[k] = [
                    alpha.wick_evaluate(normal[: alpha.I, : alpha.J])
                    for alpha in self.alphas[k]
                ]
            self.wick_values.append(wick_values_dict)

    def calculate_increments(self):
        """
        Calcula los incrementos del movimiento Browniano para cada normal.

        Este método calcula los incrementos del movimiento Browniano utilizando la fórmula del paper.
        Los incrementos se almacenan en `self.brownian_increments`.
        """
        for normal in self.normals:
            self.brownian_increments.append(np.zeros((self.I, self.n - 1)))
            for k in range(self.n - 1):
                g = np.array(
                    [int_g(self.T, j, self.t[k], self.t[k + 1]) for j in range(self.J)]
                )
                self.brownian_increments[-1][:, k] = np.dot(normal, g)

    def evaluate_paths(self):
        """
        Evalúa los caminos del movimiento Browniano a partir de los incrementos.

        Este método calcula los caminos del movimiento Browniano a partir de los incrementos almacenados en
        `self.brownian_increments` y los guarda en self.brownian_paths.
        """
        for increments in self.brownian_increments:
            self.brownian_paths.append(
                np.hstack([np.zeros((self.I, 1)), np.cumsum(increments, axis=1)])
            )

    def save_alpha_set(self, filename: str):
        """
        Guarda el alpha set en un archivo

        Parameters:
        filename (str): El nombre del archivo donde se guardará
        """
        with open(filename, "wb") as f:
            pkl.dump(self, f)




    # Métodos más eficientes

    def calculate_alphas_fast(self):
        
        # Reinicializa el diccionario
        self.alphas = {k: [] for k in range(self.K + 1)}
        i, j, k = self.I, self.J, self.K
        indices = np.arange(i * j, dtype=int)
        
        for deg in range(k + 1):
            alphas_deg = []
            if deg == 0:
                vals = np.zeros((i, j), dtype=int)
                alpha = Alpha(i, j, vals)
                alpha.calculate_factorial()
                alpha.calculate_wick_constant()
                alphas_deg.append(alpha)
            else:
                for combo in combinations_with_replacement(indices, deg):
                    counts = np.bincount(combo, minlength=i * j)
                    vals = counts.reshape(i, j)
                    alpha = Alpha(i, j, vals)
                    alpha.calculate_factorial()
                    alpha.calculate_wick_constant()
                    alphas_deg.append(alpha)

            # Orden canónico lexicográfico
            alphas_deg.sort(key=lambda a: tuple(a.values.flatten()))
            self.alphas[deg] = alphas_deg

    def _build_G(self):
        """
        Construye y guarda en self._G la matriz G[k,j] = ∫_{t_k}^{t_{k+1}} g_j
        de forma vectorizada (shape (n-1, J)).
        """
        if self._G is not None:
            return self._G
        g_mat = np.zeros((self.n-1, self.J))
        for k in range(self.n - 1):
            g_mat[k] = [int_g(self.T, j, self.t[k], self.t[k+1]) 
                        for j in range(self.J)]
        self._G = g_mat
        return self._G
    

    def add_normals(self, normals_batch):
        """
        Añade un iterable/array de normales shape (batch, I, J) y
        actualiza normals, wick_values, brownian_increments, brownian_paths.
        Usa cache de Hermite y G.
        """
        G = self._build_G()                   # (n-1, J)
        J = self.J
        rng_idx = np.arange(J)

        for normal in normals_batch:          # normal.shape = (I,J)
            self.normals.append(normal)

            # 1) ---- Wick values ------------
            # cache de Hermite para cada dimensión i
            # (si I>1 necesitarías bucle en i; para I=1 basta la primera fila)
            H = hermite_probabilistic(normal[0], self.K)   # shape (K+1,J)

            wick_dict = {}
            for k in range(self.K + 1):
                # α de grado k
                wick_dict[k] = []
                for alpha in self.alphas[k]:
                    orders = alpha.values[0]          # (J,)
                    hermite_factors = H[orders, rng_idx]
                    xi_alpha = alpha.wick_constant * hermite_factors.prod(
                        dtype=np.float32
                    )
                    wick_dict[k].append(xi_alpha)
            self.wick_values.append(wick_dict)

            # 2) ---- increments -------------
            inc = np.dot(normal, G.T)                # (I,J)·(J,n-1) = (I,n-1)
            self.brownian_increments.append(inc)

            # 3) ---- paths -------------------
            path = np.hstack(
                [np.zeros((self.I, 1), dtype=np.float32),
                 np.cumsum(inc, axis=1)]
            )
            self.brownian_paths.append(path)


def hermite_probabilistic(x, K):
    """
    Devuelve un array H[k, j] = h_k(x_j) para k = 0..K
    h_0=1 ; h_1=x ; h_{k}=x*h_{k-1}-(k-1)*h_{k-2}
    """
    J = x.shape[-1]
    H = np.empty((K+1, J), dtype=x.dtype)
    H[0] = 1.0
    if K >= 1:
        H[1] = x.copy()
    for k in range(2, K+1):
        H[k] = x*H[k-1] - (k-1)*H[k-2]
    return H