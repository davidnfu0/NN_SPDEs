import numpy as np
import scipy.special as scsp


class Alpha:
    """
    Clase que representa los valores de la matriz alpha, su factorial y la constante de Wick en el contexto
    de la expansión en caos de Wiener.

    Attributes:
    I (int): Dimensión del movimiento Browniano.
    J (int): Número de términos en la expansión de caos de Wiener.
    values (np.array): Matriz de tamaño (I, J) que contiene los valores de la matriz alpha.
    factorial (float): El valor factorial calculado para los valores de alpha.
    wick_constant (float): La constante de Wick, que es 1/sqrt(factorial).

    Methods:
    calculate_factorial(): Calcula el factorial de alpha.
    calculate_wick_constant(): Calcula la constante de Wick (1/sqrt(factorial)).
    wick_evaluate(normals): Evalúa el polinomio de Wick para los valores dados y los normales.
    """

    def __init__(self, I: int, J: int, values) -> None:
        """
        Inicializa la clase Alpha con los valores de la matriz alpha.
        """
        self.I = I
        self.J = J
        self.values = np.array(values, dtype=np.int64)
        self.factorial = 1
        self.wick_constant = 0

        # Verificación de que la forma de 'values' es correcta
        assert self.values.shape == (I, J), (
            f"Expected shape ({I}, {J}), but got {self.values.shape}"
        )

    def calculate_factorial(self):
        """
        Calcula el factorial de la matriz alpha según la definición del paper utilizando la función factorial.

        Este método calcula el producto de los factoriales de cada valor en la matriz `values`, de acuerdo con
        la expansión en caos de Wiener, y almacena el resultado en el atributo `factorial`.
        """
        self.factorial = np.prod(scsp.factorial(self.values), dtype=np.float32)

    def calculate_wick_constant(self):
        """
        Calcula la constante de Wick, que es 1/sqrt(factorial).

        Este método calcula la constante de Wick usando el valor del `factorial` almacenado en el objeto. Si el
        factorial no ha sido calculado aún, se invoca el método `calculate_factorial()` para calcularlo primero.
        """
        if self.factorial == 1:  # Verifica si el factorial aún no se ha calculado
            self.calculate_factorial()
        self.wick_constant = 1 / np.sqrt(self.factorial)

    def wick_evaluate(self, normals):
        """
        Evalúa el polinomio de Wick para los valores dados y las normales estándar.

        Este método utiliza los valores de la matriz alpha y las normales como entrada para calcular
        el polinomio de Wick asociado evaluado en las realizaciones. La evaluación se realiza utilizando los polinomios de Hermite, y luego el
        resultado se multiplica por la constante de Wick.

        Parameters:
        normals (np.array): Un arreglo de realizaciones de variables aleatorias normales estándar.

        Returns:
        float: El valor evaluado del polinomio de Wick.
        """
        if (
            self.wick_constant == 0
        ):  # Verifica si la constante de Wick ha sido calculada
            self.calculate_wick_constant()
        hermite = scsp.eval_hermitenorm(self.values, normals)
        return self.wick_constant * np.prod(hermite, dtype=np.float32)
