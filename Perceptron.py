import numpy as np

# Define o número de épocas e o número de amostras (q)
numEpocas = 100 # Valor para ter erro com 0
numAmostras = 6

# Atributos
febre =   np.array([1, 0, 1, 1, 1, 0])
enjoo =   np.array([1, 0, 1, 0, 0, 0])
manchas = np.array([0, 1, 0, 1, 0 ,1])
dores =   np.array([1, 0, 0, 1, 1, 1])

# Bias
bias = 1

# Entrada do perceptron
listaAtributos = np.vstack((febre, enjoo, manchas, dores))
resultadoEsperado = np.array([-1, 1, 1, -1, 1, -1]) #Doente -1 e Saudável 1

# Taxa de aprendizado
taxaAprendizado = 0.1

# Define o vetor de pesos
peso = np.zeros([1,5]) # Quatro entradas + o bias

# Array para armazenar os erros
erro = np.zeros(6)

def funcaoAtivacao(valor):
    # A funcao de ativação é a degrau bipolar
    if valor<0.0:
        return (-1)
    else:
        return (1)

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
        print(peso)
        
print("Vetor de erros = "+str(erro))

#Teste
# Atributos
febre =   np.array([0, 1])
enjoo =   np.array([0, 1])
manchas = np.array([0, 1])
dores =   np.array([1, 1])

# Entrada do perceptron
listaAtributosTeste = np.vstack((febre, enjoo, manchas, dores))

for i in range(2):
    # Inserir o Bias no vetor de entrada
    listaAtributosBias = np.hstack((bias, listaAtributosTeste[:,i])) # Empilha o bias sobre as 4 linhas da matriz listaAtributos
    
    V = np.dot(peso, listaAtributosBias) #Multiplicação vetorial
    
    valorSaida = funcaoAtivacao(V)

    if valorSaida == -1:
        print('Doente')
    elif valorSaida == 1:
        print('Saudável')