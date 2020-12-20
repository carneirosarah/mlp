'''
Trabalho 8 - Sistemas Inteligentes
MLP - Gaussianas
Sarah R. L. Carneiro
'''

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn

class MLP:

    def __init__(self, learningRate, numberOfTimes, hiddenLayersNeurons, valRatio, n_iter_no_change):

        self.learningRate = learningRate  # taxa de apredisagem
        self.numberOfTimes = numberOfTimes  # numero de epocas
        self.hiddenLayersNeurons = hiddenLayersNeurons # número de neurônios por camada escondida
        self.valRatio = valRatio  # percentual dos vetores de treinamento que serão utilizados para validação
        self.n_iter_no_change = n_iter_no_change # critério para early stopping


    def trainMLP(self, X, T):

        return MLPClassifier(
            activation = "logistic", # função de ativação = sigmoide
            solver = "adam", # gradiente decendente
            hidden_layer_sizes=self.hiddenLayersNeurons,
            max_iter=self.numberOfTimes,
            early_stopping=True,
            learning_rate_init=self.learningRate,
            validation_fraction=self.valRatio,
            n_iter_no_change= self.n_iter_no_change # o número máximo de épocas sem melhora
        ).fit(X, T)

    @staticmethod
    def loadData():

        file = open('DATA.txt', 'r')
        data = []
        for line in file.readlines():
            data.append([el.replace('\n', '') for el in line.split(',')])

        data = np.array(data, dtype=float)
        file.close()

        file = open('TARGETS.txt', 'r')
        targets = []
        for line in file.readlines():
            targets.append([el.replace('\n', '') for el in line.split(',')])

        targets = np.array(targets, dtype=int)
        file.close()

        return data, targets

    @staticmethod
    def plotConfusionMatrix(data, output_filename):
        seaborn.set(color_codes=True)
        plt.figure(1, figsize=(9, 6))

        plt.title("Matriz de confusão")

        seaborn.set(font_scale=1.4)
        seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt="d")

        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

def main():

    print('\n\n===== MLP - Classificação de Distribuições Gaussianas =====\n\n')
    learningRate = float(input('Insira a taxa de aprendizagem:'))
    percentTrain = float(input('Insira a porcentagem de dados que serão utililizados no treinamento:'))
    percentVal = float(input('Insira a porcentagem de dados que serão utililizados para validação:'))
    numberOfTimes = int(input('Insira o número máximo de épocas:'))
    n_iter_no_change = int(input('Insira o número máximo de épocas sem melhora:'))

    data, targets = MLP.loadData()

    # cria os conjuntos de treinamento e de teste
    rand_state = np.random.RandomState(0)
    rand_state.shuffle(data)
    idx = int(percentTrain * len(data))
    trainSet = data[0:idx, :]
    testSet = data[idx + 1:len(data), :]

    rand_state.seed(0)
    rand_state.shuffle(targets)
    trainTargets = targets[0:idx, :]
    testTargets = targets[idx+1:len(targets), :]

    model = MLP(learningRate, numberOfTimes, [9], percentVal, n_iter_no_change)

    # train
    mlp = model.trainMLP(trainSet, trainTargets)

    # test
    output = mlp.predict(testSet)

    conf = confusion_matrix(testTargets.argmax(axis=1), output.argmax(axis=1))
    print(conf)
    print(classification_report(testTargets.argmax(axis=1), output.argmax(axis=1)))

    MLP.plotConfusionMatrix(conf, 'matrizConfusao')

if __name__ == '__main__':
    main()