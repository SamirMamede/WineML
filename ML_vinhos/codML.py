import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

#Carregando o conjunto de arquivos
arquivo = pd.read_csv('C:/Users/Anuar/Documents/pandas_machineLearning/WineML/ML_vinhos/wine_dataset.csv')

#Tratando a coluna style para que o pyhton possa calcular, string -> int
arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

#Separando as variáveis entre preditoras(x) e variável alvo(y)
y = arquivo['style']
x = arquivo.drop('style', axis=1)

#Criando o conjunto de dados de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

#Criação do modelo
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)
#Imprimindo resultados
resultado = modelo.score(x_teste, y_teste)
print(f'Acurácia: {resultado}')

#Verificando comportamento com outros dados ainda não testados
print(y_teste[400:405])
print(x_teste[400:405])
previsoes = modelo.predict(x_teste[400:405])
print(previsoes)