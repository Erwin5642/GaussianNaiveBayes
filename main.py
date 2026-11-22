"""Naive Bayes Gaussiano."""
import numpy as np

SEED = 123
TAM_TESTE = 0.25

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
    return X, y


"""Divide o dataset em um conjunto de treinamento e teste."""
def divide_conjunto(X, y, tam_teste, seed): 
    rng = np.random.default_rng(seed)

    indices_treino = []
    indices_teste = []

    for classe in np.unique(y):
        indices_classe = np.where(y == classe)[0]
        rng.shuffle(indices_classe)

        qtd_teste = int(len(indices_classe) * tam_teste)

        indices_teste.extend(indices_classe[:qtd_teste])
        indices_treino.extend(indices_classe[qtd_teste:])

    indices_treino = np.array(indices_treino)
    indices_teste = np.array(indices_teste)
    rng.shuffle(indices_treino)
    rng.shuffle(indices_teste)

    return X[indices_treino], X[indices_teste], y[indices_treino], y[indices_teste]


"""Faz o calculo de conversao x-u"""
def converte_conjuntos(X_treino, X_teste):
    u = np.mean(X_treino, axis = 0)

    X_treino_conv = X_treino - u
    X_teste_conv = X_teste - u

    return X_treino_conv, X_teste_conv


def main():
    """Configura a impressao sem limite do numpy"""
    np.set_printoptions(threshold=np.inf)

    """Faz a leitura do dataset e grava em X e y"""
    X, y = load_dataset('seeds_dataset.txt')

    print('Dados carregados')
    print(X)
    print(y)

    X_treino, X_teste, y_treino, y_teste = divide_conjunto(X, y, TAM_TESTE, SEED)

    print('Dados separados')
    print('X_treino')
    print(X_treino)
    print('y_treino')
    print(y_treino)
    print('X_teste')
    print(X_teste)
    print('y_teste')
    print(y_teste)

    X_treino_conv, X_teste_conv = converte_conjuntos(X_treino, X_teste)

    print('Dados convertidos')
    print('X_treino_conv')
    print(X_treino_conv)
    print('X_teste_conv')
    print(X_teste_conv)

    # TODO: treinar e avaliar
    pass


if __name__ == "__main__":
    main()
