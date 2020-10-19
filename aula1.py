import pandas as pd, numpy, matplotlib.pyplot as plt

# fonte = 'https://github.com/alura-cursos/imersao-dados-2-2020/blob/master/MICRODADOS_ENEM_2019_SAMPLE_43278.csv?raw=true'

# forma alternativa se já baixado
fonte = './nao_enviar/MICRODADOS_ENEM_2019_SAMPLE_43278.csv'

pd.options.display.width = None  # todas as colunas horizontalmente
# pd.set_option('display.max_columns', None)


dados = pd.read_csv(fonte)
print(dados.head())
# print(dados.shape)   #  // (número de linhas, numero de colunas) -> (127380, 136)

print('1------------------------')

print(dados['NO_MUNICIPIO_NASCIMENTO'])  # escolhendo uma coluna

print('2------------------------')
print(dados.columns.values)  # nomes das colunas

print('------------------------')
# print(dados['NO_MUNICIPIO_NASCIMENTO', 'Q025'])  # dá erro
print(dados[['NO_MUNICIPIO_NASCIMENTO', 'Q025']])  # 2 colunas

print('3------------------------')
print(dados['SG_UF_RESIDENCIA'].unique())  # lista de UFs não repetidos
print(len(dados['SG_UF_RESIDENCIA'].unique()))

print('4------------------------')
print(dados['SG_UF_RESIDENCIA'].value_counts())  # conta valores

print('5------------------------')
print(dados['NU_IDADE'].value_counts())  # conta valores

print('6------------------------')
print(dados['NU_IDADE'].value_counts().sort_index())

# desafio1: proporção de inscritos por idade
# desafio2: Descobrir de quais estados são os inscritos com 13 anos.

print('7------------------------')
print(dados['NU_IDADE'].hist())   # testar no colab ipynb
print(dados['NU_IDADE'].hist(bins=20, figsize=(10, 8)))  # testar no colab ipynb
dados['NU_IDADE'].hist()
print(dados['NU_IDADE'].hist)

print('8------------------------')
# print(dados['NU_IDADE'])
# print(type(dados['NU_IDADE']))
# print(dados['NU_IDADE'].value_counts().sort_index())
# print('**')
# print(dados['NU_IDADE'].value_counts().sort_index().keys())
# print(dados['NU_IDADE'].value_counts().sort_index().keys()[0])
# print(dados['NU_IDADE'].value_counts().sort_index().values)
# print(dados['NU_IDADE'].value_counts().sort_index().values[0])
chaves = []
valores = []

for chave in dados['NU_IDADE'].value_counts().sort_index().keys():
    chaves.append(chave)

for valor in dados['NU_IDADE'].value_counts().sort_index().values:
    valores.append(valor)

print(chaves)
print(valores)

fig, ax = plt.subplots()

ax.plot(chaves, valores)

plt.show()

print('9------------------------')
"""
import matplotlib.pyplot as plt

ages = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

total_population = [27877307, 24280683, 25258169, 25899454, 24592293, 21217467, 27958147, 20859088, 28882735, 19978972]

fig, ax = plt.subplots()

ax.plot(ages, total_population)

plt.show()

"""

print('10------------------------')
print('11------------------------')
print('12------------------------')
print('13------------------------')
print('14------------------------')
