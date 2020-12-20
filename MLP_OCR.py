'''
Trabalho 9 - Sistemas Inteligentes
MLP - OCR
Sarah R. L. Carneiro
'''

import sys
import numpy as np
from tensorflow import keras
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

class MLP:

    def __init__(self, learningRate, numberOfTimes, hiddenLayersNeurons, valRatio, n_iter_no_change):

        self.learningRate = learningRate  # taxa de apredisagem
        self.numberOfTimes = numberOfTimes  # numero de epocas
        self.hiddenLayersNeurons = hiddenLayersNeurons # número de neurônios por camada escondida
        self.valRatio = valRatio  # percentual dos vetores de treinamento que serão utilizados para validação
        self.n_iter_no_change = n_iter_no_change # critério para early stopping


    def trainMLP(self, X, T):

        return MLPClassifier(
            activation="logistic", # função de ativação = sigmoide
            solver="adam", # gradiente decendente
            hidden_layer_sizes=self.hiddenLayersNeurons,
            max_iter=self.numberOfTimes,
            early_stopping=True,
            learning_rate_init=self.learningRate,
            validation_fraction=self.valRatio,
            n_iter_no_change=self.n_iter_no_change, # o número máximo de épocas sem melhora
            shuffle=True,
            verbose=True
        ).fit(X, T)

    # Adequa a matriz de alvos
    @staticmethod
    def labelTreatment(T):

        auxT = np.zeros((len(T), 10))

        for i in range(0, len(T)):
            auxT[i][T[i]] = 1

        return auxT

    # Adequa os padrões de entrada
    @staticmethod
    def dataTreatment(trainSet, testSet):

        # converte a matriz de 3 dimensões para 2 dimensões
        trainSet = trainSet.reshape(len(trainSet), 784)
        testSet = testSet.reshape(len(testSet), 784)

        # preserva apenas as variaveis com a variancia não nula
        std = np.std(trainSet, axis=0)
        idx = list(filter(lambda i: std[i] > 0, range(len(std))))
        trainSet = trainSet[:, idx]
        testSet = testSet[:, idx]

        # normaliza os valores de luminância - originalmente valores inteiros
        # no intervalo [0,255] - para valores reais no intervalo [0,1]
        min_max_scaler = preprocessing.MinMaxScaler()
        trainSet = min_max_scaler.fit_transform(trainSet)
        testSet = min_max_scaler.fit_transform(testSet)

        return trainSet, testSet

    # Utilizo o Keras para importar o dataset
    # https://keras.io/api/datasets/mnist/
    @staticmethod
    def loadData():

        dataSet = keras.datasets.mnist
        ((trainSet, trainLabels), (testSet, testLabels)) = dataSet.load_data(path="mnist.npz")

        return trainSet, trainLabels, testSet, testLabels

    @staticmethod
    def binaryToDecimal (M):

        return np.array([9 if np.sum([m[i]*(i) for i in range(len(m))]) > 9 else np.sum([m[i]*(i) for i in range(len(m))]) for m in M])

def main():

    print('\n\n===== MLP - OCR =====\n\n')
    learningRate = float(input('Insira a taxa de aprendizagem:'))
    percentTrain = float(input('Insira a porcentagem de dados que serão utililizados no treinamento:'))
    percentVal = float(input('Insira a porcentagem de dados que serão utililizados para validação:'))
    numberOfTimes = int(input('Insira o número máximo de épocas:'))
    n_iter_no_change = int(input('Insira o número máximo de épocas sem melhora:'))

    # carrega os padroes de entrada e alvos
    trainSet, trainLabels, testSet, testLabels = MLP.loadData()

    # padroes de entrada
    trainSet, testSet = MLP.dataTreatment(trainSet, testSet)

    # alvos
    trainLabels = MLP.labelTreatment(trainLabels)

    model = MLP(learningRate, numberOfTimes, [50], percentVal, n_iter_no_change)

    # train
    mlp = model.trainMLP(trainSet, trainLabels)

    # test
    output = mlp.predict(testSet)

    output = MLP.binaryToDecimal(output)

    np.set_printoptions(threshold=sys.maxsize)
    #print(testLabels, testLabels.shape)
    #print(output, output.shape)

    #conf = confusion_matrix(testLabels, output)
    #print(conf)
    print(classification_report(testLabels, output))

if __name__ == "__main__":
    main()