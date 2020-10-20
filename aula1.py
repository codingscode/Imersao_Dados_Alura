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
print(dados['NU_IDADE'].hist())  # testar no colab ipynb
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

# desafio3: Adicionar título no gráfico

print('9------------------------')
"""
import matplotlib.pyplot as plt

ages = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

total_population = [27877307, 24280683, 25258169, 25899454, 24592293, 21217467, 27958147, 20859088, 28882735, 19978972]

fig, ax = plt.subplots()

ax.plot(ages, total_population)

plt.show()

"""

print(dados['IN_TREINEIRO'])

print('10------------------------')
# somente os treineiros
print(dados.query('IN_TREINEIRO == 1'))

print('11------------------------')
print(dados.query('IN_TREINEIRO == 1')['NU_IDADE'])

print('12------------------------')
print(dados.query('IN_TREINEIRO == 1')['NU_IDADE'].value_counts())

print('13------------------------')
print(dados.query('IN_TREINEIRO == 1')['NU_IDADE'].value_counts().sort_index())
# desafio4: Plotar os Histogramas das idades dos treineiros e não treineiros.

print('14------------------------')
print(dados['NU_NOTA_REDACAO'].hist(bins=20, figsize=(8, 6)))

notas = []
quantidades = []

for nota in dados.query('IN_TREINEIRO == 1')['NU_IDADE'].value_counts().sort_index().keys():
    notas.append(nota)

for quantidade in dados.query('IN_TREINEIRO == 1')['NU_IDADE'].value_counts().sort_index().values:
    quantidades.append(quantidade)

print(notas)
print(quantidades)

fig, ax = plt.subplots()
ax.plot(notas, quantidades)

plt.show()

print('15------------------------')
dados['NU_NOTA_LC'].hist(bins=20, figsize=(8, 6))

notas2 = []
quantidades2 = []

for nota2 in dados.query('IN_TREINEIRO == 1')['NU_IDADE'].value_counts().sort_index().keys():
    notas2.append(nota2)

for quantidade2 in dados.query('IN_TREINEIRO == 1')['NU_IDADE'].value_counts().sort_index().values:
    quantidades2.append(quantidade2)

print(notas2)
print(quantidades2)

fig, ax = plt.subplots()
ax.plot(notas2, quantidades2)

plt.show()

print('16------------------------')
print('média: ', dados['NU_NOTA_REDACAO'].mean())

print('17------------------------')
print('desvio padrão: ', dados['NU_NOTA_REDACAO'].std())

print('18------------------------')
provas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_MT', 'NU_NOTA_LC', 'NU_NOTA_REDACAO']
print(dados[provas].describe())

print('19------------------------')
print(dados['NU_NOTA_LC'].quantile(0.9))  # 1 - 0.9 -> 10% ; 10% tem acima deste valor
print(dados['NU_NOTA_LC'].quantile(0.1))

print('20------------------------')
print(dados['NU_NOTA_LC'].plot.box(grid=True, figsize=(8, 6)))  # aparece umas grades no gráfico
print(dados['NU_NOTA_LC'].keys())
print(dados['NU_NOTA_LC'].keys()[0])
print(dados['NU_NOTA_LC'].keys()[1])
print(dados['NU_NOTA_LC'].keys()[2])
print(dados['NU_NOTA_LC'].values)

nu_nota_lc = []
valores_y = []

for nota in dados['NU_NOTA_LC'].keys():
    nu_nota_lc.append(nota)

for val_y in dados['NU_NOTA_LC'].values:
    valores_y.append(val_y)

fig, ax = plt.subplots()
ax.plot(nu_nota_lc, valores_y)

plt.show()

print('21------------------------')
print(dados[provas].boxplot(grid=True, figsize=(10, 8)))

nomes_e_notas = {'cn': [[], []], 'ch': [[], []], 'mt': [[], []], 'lc': [[], []], 'red': [[], []]}
notas3 = []
as_5 = dict()



"""
for nome_nota in dados[provas].keys():
    nome_notas.append(nota)

for nota in dados[provas].values:
    notas3.append(nota)


"""
print(len(dados[provas]))
print('1*', dados[provas])
print('nome*', dados[provas]['NU_NOTA_CN'])
print('2*', dados[provas].values)
print('3*', dados[provas].values[0])
print('4*', dados[provas].values[1])

for disciplina in provas:
    #print(disciplina)
    for nome in dados[provas][f'{disciplina}'].keys():
        pass



"""

for i in range(len(dados[provas])):
    for cada in dados[provas][i].keys():
        nome_notas.append(cada)

for i in range(len(dados[provas])):
    for cada in dados[provas][i].values:
        notas3.append(cada)

print('****', nome_notas)
print('****', notas3)

fig, ax = plt.subplots()
ax.plot(nome_notas, notas3)

plt.show()

"""





print('22------------------------')
print('23------------------------')
print('24------------------------')
print('25------------------------')
