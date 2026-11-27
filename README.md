# GaussianNaiveBayes

Implementação do algoritmo **Naive Bayes Gaussiano** em Python, desenvolvida para a disciplina de Inteligência Artificial.

## Ideia

O **Naive Bayes Gaussiano** é um classificador probabilístico baseado no Teorema de Bayes, projetado para dados contínuos. O algortimo opera sob a premissa de que todas as variáveis são independentes entre si e assume que a distribuição dos valores de cada variável segue uma curva gaussiana. Durante o treinamento, o modelo calcula a média e a variância de cada atributo por classe e, posteriormente, utiliza esses valores para estimar a probabilidade de novas amostras pertencerem a cada categoria e classificá-lo pela maior probabilidade.

## Estrutura

```
.
├── main.py             # Ponto de entrada e implementação do classificador
├── seeds_dataset.txt   # Dataset utilizado
├── README.md
└── .gitignore
```

## Configuração

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Execução

```bash
python main.py
```
