import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    """
    Clase que define una red neuronal para aproximar soluciones a ecuaciones estocásticas utilizando
    la expansión en caos de Wiener y los valores de los polinomios de Wick.

    Attributes:
    alpha_set (AlphaSet): Conjunto de alphas y valores de Wick utilizados para la expansión en caos de Wiener.
    activation (nn.Module): Función de activación de la red neuronal.
    omega (int): Índice de la normal que se está utilizando para calcular la realización de la solución.
    alpha_card (int): Número total de alphas en el conjunto.
    networks (nn.ModuleList): Lista de redes neuronales (una por cada alpha).
    t (np.array): Intervalos de tiempo.

    Methods:
    forward(input): Realiza la propagación hacia adelante a través de la red neuronal.
    save_weights(name): Guarda los pesos del modelo.
    load_weights(name): Carga los pesos guardados previamente del modelo.
    train_model(space_domain, loss_function, initial_condition, epochs=1000, optimizer=optim.Adam, lr=0.001):
        Entrena la red neuronal utilizando la función de pérdida proporcionada.
    """

    def __init__(
        self,
        space_dim,
        alpha_set,
        n_layers=1,
        wide=16,
        activation=nn.Tanh,
    ) -> None:
        """
        Inicializa la red neuronal con los parámetros proporcionados.

        Parameters:
        space_dim (int): Dimensión del espacio de entrada (tamaño de la entrada).
        alpha_set (AlphaSet): Conjunto de alphas utilizado para la expansión en caos de Wiener.
        n_layers (int): Número de capas ocultas en la red neuronal.
        wide (int): Número de neuronas por capa oculta.
        activation (nn.Module): Función de activación a utilizar (por defecto es Tanh).
        """
        super().__init__()

        self.alpha_set = alpha_set
        self.n_bpaths = len(alpha_set.brownian_paths)
        self.activation = activation
        self.t = alpha_set.t

        self.omega = 0  # Se puede cambiar para calcular una diferente realización.
        self.alpha_card = alpha_set.calculate_card()

        # Inicializa la lista de redes neuronales (una por cada alpha)
        self.networks = nn.ModuleList()
        input_dim = (
            space_dim + 1
        )  # La entrada es de tamaño `space_dim` más el término de condición inicial
        for _ in range(self.alpha_card):
            layers = []
            for n in range(n_layers):
                in_dim = (
                    input_dim if n == 0 else wide
                )  # El primer layer tiene como entrada `input_dim`, luego `wide` para los siguientes
                layers.append(nn.Linear(in_dim, wide))  # Capa densa
                layers.append(self.activation())  # Aplicar la activación
            layers.append(nn.Linear(wide, 1))  # Capa final, salida de tamaño 1
            self.networks.append(
                nn.Sequential(*layers)
            )  # Añadir la red al módulo de redes neuronales

    def forward(self, input):
        """
        Realiza la propagación hacia adelante a través de la red neuronal.
        Calcula la salida de la red para una entrada dada.

        Parameters:
        input (torch.Tensor): La entrada para la red neuronal (Puntos en el dominio del espacio).

        Returns:
        output (torch.Tensor): La salida de la red neuronal, una aproximación de la solución de la SPDE.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        output = 0
        wick_values_list = []

        # Para cada grado de los polinomios de Wick (de 0 a K), obtenemos los valores correspondientes
        for k in range(self.alpha_set.K + 1):
            wick_values_list += self.alpha_set.wick_values[self.omega][k]

        # Convertimos los valores de Wick a un tensor de PyTorch
        wick_values_tensor = torch.tensor(
            wick_values_list, dtype=input.dtype, device=device
        )

        # Para cada red en la lista de redes neuronales (una por cada alpha), calculamos su salida
        for idx, net in enumerate(self.networks):
            net = net.to(device)
            x = input  # La entrada es el vector dado al método `forward`
            out = net(
                x
            ).squeeze()  # Pasamos la entrada por la red y eliminamos la dimensión extra
            output = (
                output + out * wick_values_tensor[idx]
            )  # Acumulamos la salida ponderada por el valor de Wick

        return output

    def save_weights(self, name):
        """
        Guarda los pesos del modelo en un archivo.

        Parameters:
        name (str): Nombre del archivo para guardar los pesos.
        """
        torch.save(self.state_dict(), ("files/" + name + ".txt"))
        return

    def load_weights(self, name):
        """
        Carga los pesos del modelo desde un archivo guardado previamente.

        Parameters:
        name (str): Nombre del archivo que contiene los pesos guardados.
        """
        self.load_state_dict(torch.load(("files/" + name + ".txt")))
        return

    def train_model(
        self,
        space_domain,
        loss_function,
        initial_condition,
        epochs=1000,
        optimizer=optim.Adam,
        lr=0.001,
        model_name="best_model",
        n_normals_to_train=1,
        n_normals_batch_size=1,
    ):
        """
        Entrena el modelo utilizando el dominio del espacio, la función de pérdida y la condición inicial.

        Parameters:
        space_domain (torch.Tensor): El dominio del espacio (los puntos del espacio donde evaluamos la solución).
        loss_function (function): La función de pérdida para optimizar el modelo.
        initial_condition (function): La condición inicial del modelo (la función que representa la condición inicial en el tiempo).
        epochs (int): El número de épocas para entrenar el modelo.
        optimizer (torch.optim.Optimizer): El optimizador utilizado para actualizar los pesos (por defecto es SGD).
        lr (float): La tasa de aprendizaje para el optimizador.
        model_name (str): Nombre del modelo para guardar los pesos del mejor modelo encontrado durante el entrenamiento.
        n_normals_to_train (int): Número de normales a entrenar en cada época (por defecto es 1).
        n_normals_batch_size (int): Tamaño del batch de normales a entrenar en cada época (por defecto es 1).

        Returns:
        loss_list (list): Lista con el valor de la pérdida en cada época.
        """
        pbar = tqdm(total=epochs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        # Asegura que space_domain sea un tensor, y muévelo al device
        if not isinstance(space_domain, torch.Tensor):
            space_domain = torch.as_tensor(
                space_domain, dtype=torch.float32, device=device
            )
        else:
            space_domain = space_domain.to(device)

        self.train()  # Pone la red en modo de entrenamiento
        opt = optimizer(self.parameters(), lr=lr)  # Inicializa el optimizador
        loss_list = []
        best_loss = float("inf")

        # Entrenamos durante el número de épocas especificado
        for epoch in range(epochs):
            # Selecciona aleatoriamente n_normals_batch_size índices únicos de las normales disponibles
            indices = torch.randperm(n_normals_to_train)[:n_normals_batch_size].tolist()
            total_loss = []
            for bpath_n in indices:
                self.omega = bpath_n  # Actualiza la normal que se está utilizando
                opt.zero_grad()  # Resetea los gradientes de los parámetros
                loss = loss_function(
                    self, space_domain, initial_condition
                )  # Calcula la Pérdida
                total_loss.append(loss)  # Suma el error de cada normal
            total_loss = torch.stack(
                total_loss
            ).mean()  # Promedia las pérdidas de las normales
            total_loss.backward()  # Calcula el gradiente de la pérdida total
            opt.step()  # Actualiza los parámetros del modelo después de acumular el gradiente
            loss_list.append(
                total_loss.item()
            )  # Almacena el valor total de la pérdida del batch

            # Si se mejora la pérdida, guardamos los pesos del modelo
            if loss_list[-1] < best_loss:
                best_loss = loss_list[-1]
                self.save_weights(model_name)

            # Actualiza la barra de progreso
            if epoch % max(1, (epochs // 100)) == 0 or epoch == epochs - 1:
                pbar.set_description(
                    f"Epoch [{epoch + 1}/{epochs}] | Loss: {loss_list[-1]:.4f} | Best Loss: {best_loss:.4f}"
                )
            if epoch == epochs - 1:
                pbar.close()

        # Carga el mejor modelo guardado
        self.load_weights(model_name)
        return loss_list  # Retorna la lista de pérdidas durante el entrenamiento
