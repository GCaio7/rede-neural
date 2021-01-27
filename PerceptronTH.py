import numpy as np

# Define o número de épocas e o número de amostras (q)
numEpocas = 50 # Valor para ter erro com 0
numAmostras = 2


'''
    0   0   0   0

    0   0   0   0

    0   0   0   0

    0   0   0   0
'''


# Atributos
listaMatriz = []

listaMatriz.append(np.array([1, 1]))
listaMatriz.append(np.array([1, 0]))
listaMatriz.append(np.array([1, 1]))

listaMatriz.append(np.array([0, 1]))
listaMatriz.append(np.array([1, 1]))
listaMatriz.append(np.array([0, 1]))

listaMatriz.append(np.array([0, 1]))
listaMatriz.append(np.array([1, 0]))
listaMatriz.append(np.array([0, 1]))


# Bias
bias = 1

# Entrada do perceptron
listaAtributos = np.vstack((listaMatriz))

resultadoEsperado = np.array([1, -1]) #T = 1, H = -1

# Taxa de aprendizado
taxaAprendizado = 0.1

# Define o vetor de pesos
peso = np.zeros([1, 10]) # 3 entradas + o bias

# Array para armazenar os erros
erro = np.zeros(2)

def funcaoAtivacao(valor):
    # A funcao de ativação é a degrau bipolar
    if valor < 0.0:
        return (-1)
    elif valor > 0.0:
        return (1)
    return 0

#Treinamento
for j in range(numEpocas):
    for k in range(numAmostras):

        # Inserir o Bias no vetor de entrada
        listaAtributosB = np.hstack((bias, listaAtributos[:,k])) # Empilha o bias sobre as 4 linhas da matriz listaAtributos
        # Calcula o campo induzido

        V = np.dot(peso, listaAtributosB) #Multiplicação vetorial
        # Calcula a saída do Perceptron
        valorSaida = funcaoAtivacao(V)

        # Calcula o erro: erro = (resultadoEsperado - valorSaida) ou seja, a saída que queremos MENOS a saída da rede
        erro[k] = resultadoEsperado[k] - valorSaida

        # Treinamento do Perceptron
        peso = peso + taxaAprendizado*erro[k]*listaAtributosB


print("Vetor de erros = "+str(erro))

#Teste
# Atributos
listaMatriz = []
listaMatriz.append(np.array([0]))
listaMatriz.append(np.array([1]))
listaMatriz.append(np.array([0]))

listaMatriz.append(np.array([0]))
listaMatriz.append(np.array([1]))
listaMatriz.append(np.array([0]))

listaMatriz.append(np.array([0]))
listaMatriz.append(np.array([1]))
listaMatriz.append(np.array([0]))

# Entrada do perceptron
listaAtributosTeste = np.vstack((listaMatriz))

for i in range(1):
    # Inserir o Bias no vetor de entrada
    listaAtributosBias = np.hstack((bias, listaAtributosTeste[:, i])) # Empilha o bias sobre as 4 linhas da matriz listaAtributos
    
    V = np.dot(peso, listaAtributosBias) #Multiplicação vetorial

    valorSaida = funcaoAtivacao(V)

    if valorSaida == -1:
        print('H')
    elif valorSaida == 1:
        print('T')
