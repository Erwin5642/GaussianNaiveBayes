"""Naive Bayes Gaussiano."""
import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        """Estima priors, médias e variâncias de cada atributo por classe."""
        raise NotImplementedError

    def predict(self, X):
        """Retorna a classe de maior log-posteriori para cada amostra."""
        raise NotImplementedError

"""Le o dataset, converte de numpy.ndarray para list nativa do python e retorna (X, y)."""
def load_dataset(path):
    X = np.loadtxt(path, usecols = [0, 1, 2, 3, 4, 5, 6], dtype = float)
    y = np.loadtxt(path, usecols = 7, dtype = int)

    X_list = X.tolist()
    y_list = y.tolist()
    return X_list, y_list


def main():
    """Faz a leitura do dataset e grava em X e y"""
    X, y = load_dataset('seeds_dataset.txt')

    print(X)
    print(y)

    # TODO: dividir treino/teste, treinar e avaliar
    pass


if __name__ == "__main__":
    main()
