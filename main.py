"""Naive Bayes Gaussiano."""
import numpy as np


class GaussianNaiveBayes:
    def __init__(self):
        self._classes = None
        self._priors = None
        self._means = None
        self._vars = None
        self._likelihood = None

    def fit(self, X, y):
        """Estima priors, médias e variâncias de cada atributo por classe."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self._classes, counts = np.unique(y, return_counts=True)
        n_classes = len(self._classes)

        self._means = np.zeros((n_classes, n_features))
        self._vars = np.zeros((n_classes, n_features))

        for idx, _class in enumerate(self._classes):
            X_c = X[y == _class]
            self._means[idx] = np.mean(X_c, axis=0)
            self._vars[idx] = np.var(X_c, axis=0)

        self._priors = counts / len(y)
        return self

    def predict(self, X):
        """Retorna a classe de maior log-posteriori para cada amostra."""
        X = np.asarray(X, dtype=np.float64)
        posteriors = []

        for idx, _ in enumerate(self._classes):
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * self._vars[idx]) + ((X - self._means[idx]) ** 2) / self._vars[idx], axis=1)
            posteriors.append(np.log(self._priors[idx]) + log_likelihood)

        posteriors = np.array(posteriors).T
        return self._classes[np.argmax(posteriors, axis=1)]


def load_dataset(path):
    """Lê o dataset e retorna (X, y)."""
    raise NotImplementedError


def main():
    # Cada linha: área, perímetro, compacidade, comprimento, largura,
    # coeficiente de assimetria, comprimento do sulco, classe
    # 30 amostras tiradas de seeds_dataset.txt (10 de cada classe)
    data = [
        [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22, 1],
        [14.29, 14.09, 0.905, 5.291, 3.337, 2.699, 4.825, 1],
        [16.44, 15.25, 0.888, 5.884, 3.505, 1.969, 5.533, 1],
        [12.72, 13.57, 0.8686, 5.226, 3.049, 4.102, 4.914, 1],
        [13.45, 14.02, 0.8604, 5.516, 3.065, 3.531, 5.097, 1],
        [14.99, 14.56, 0.8883, 5.57, 3.377, 2.958, 5.175, 1],
        [15.38, 14.77, 0.8857, 5.662, 3.419, 1.999, 5.222, 1],
        [11.42, 12.86, 0.8683, 5.008, 2.85, 2.7, 4.607, 1],
        [12.78, 13.57, 0.8716, 5.262, 3.026, 1.176, 4.782, 1],
        [12.73, 13.75, 0.8458, 5.412, 2.882, 3.533, 5.067, 1],
        [17.63, 15.98, 0.8673, 6.191, 3.561, 4.076, 6.06, 2],
        [19.11, 16.26, 0.9081, 6.154, 3.93, 2.936, 6.079, 2],
        [17.12, 15.55, 0.8892, 5.85, 3.566, 2.858, 5.746, 2],
        [19.51, 16.71, 0.878, 6.366, 3.801, 2.962, 6.185, 2],
        [18.72, 16.34, 0.881, 6.219, 3.684, 2.188, 6.097, 2],
        [18.55, 16.22, 0.8865, 6.153, 3.674, 1.738, 5.894, 2],
        [18.14, 16.12, 0.8772, 6.059, 3.563, 3.619, 6.011, 2],
        [17.55, 15.66, 0.8991, 5.791, 3.69, 5.366, 5.661, 2],
        [15.56, 14.89, 0.8823, 5.776, 3.408, 4.972, 5.847, 2],
        [16.23, 15.18, 0.885, 5.872, 3.472, 3.769, 5.922, 2],
        [13.07, 13.92, 0.848, 5.472, 2.994, 5.304, 5.395, 3],
        [11.82, 13.4, 0.8274, 5.314, 2.777, 4.471, 5.178, 3],
        [10.79, 12.93, 0.8107, 5.317, 2.648, 5.462, 5.194, 3],
        [11.36, 13.05, 0.8382, 5.175, 2.755, 4.048, 5.263, 3],
        [11.49, 13.22, 0.8263, 5.304, 2.695, 5.388, 5.31, 3],
        [11.24, 13.0, 0.8359, 5.09, 2.715, 3.521, 5.088, 3],
        [12.21, 13.47, 0.8453, 5.357, 2.893, 1.661, 5.178, 3],
        [10.59, 12.41, 0.8648, 4.899, 2.787, 4.975, 4.794, 3],
        [12.76, 13.38, 0.8964, 5.073, 3.155, 2.828, 4.83, 3],
        [12.3, 13.34, 0.8684, 5.243, 2.974, 5.637, 5.063, 3],
    ]
    X = np.array([row[:-1] for row in data])
    y = np.array([row[-1] for row in data])

    print(f"Treino: {len(X)} amostras")

    gaussian_nb = GaussianNaiveBayes()
    gaussian_nb.fit(X, y)

    # Amostras de teste do seeds_dataset.txt fora do treino (classes reais: 1, 2, 3)
    X_test = [
        [17.08, 15.38, 0.9079, 5.832, 3.683, 2.956, 5.484],  # classe 1
        [18.81, 16.29, 0.8906, 6.272, 3.693, 3.237, 6.053],  # classe 2
        [12.19, 13.36, 0.8579, 5.24, 2.909, 4.857, 5.158],   # classe 3
    ]
    predictions = gaussian_nb.predict(X_test)

    print(predictions)


if __name__ == "__main__":
    main()
