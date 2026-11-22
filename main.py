"""Naive Bayes Gaussiano."""


class GaussianNaiveBayes:
    def fit(self, X, y):
        """Estima priors, médias e variâncias de cada atributo por classe."""
        raise NotImplementedError

    def predict(self, X):
        """Retorna a classe de maior log-posteriori para cada amostra."""
        raise NotImplementedError


def load_dataset(path):
    """Lê o dataset e retorna (X, y)."""
    raise NotImplementedError


def main():
    # TODO: carregar dados, dividir treino/teste, treinar e avaliar
    pass


if __name__ == "__main__":
    main()
