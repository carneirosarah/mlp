'''
Trabalho 7 - Sistemas Inteligentes
Porta XOR com MLP de 2 camadas
Sarah R. L. Carneiro
'''

import numpy as np

class MLP:

    # f(XW)
    def predict(self, X, W):
        h = np.dot(X, W)
        return self.unitStep(h)

    # funcao de ativacao - degrau unitario
    def unitStep(self, x):
        return np.where(x > 0, 1, 0)

def main():

    # matriz de entrada
    X = np.array([[-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [-1, 1, 1]])

    # primeira matriz de pesos sinapticos
    V = np.array([[-0.5, -1], [1, -1], [-1, 1]])

    # saída dos neurônios da camada escondida
    a = MLP().predict(X, V)

    # adiciona o bias
    a = np.concatenate((np.full((len(a), 1), -1.0), a), axis=1)

    # segunda matriz de pesos sinapticos
    W = np.array([[-2], [-1], [-1]])

    # saída da rede
    O = MLP().predict(a, W)

    print('Saída: \n', O)

if __name__ == "__main__":
    main()