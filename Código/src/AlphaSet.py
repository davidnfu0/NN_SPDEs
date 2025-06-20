import numpy as np
import itertools as itt
import pickle as pkl
from src.Alpha import Alpha


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
        i, j, k = self.I, self.J, self.K
        for alpha_matrix in itt.product(range(k + 1), repeat=i * j):
            total = sum(alpha_matrix)
            if total <= k:
                alpha = Alpha(i, j, np.reshape(alpha_matrix, (i, j)))
                alpha.calculate_factorial()
                alpha.calculate_wick_constant()

                self.alphas[total].append(alpha)

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
