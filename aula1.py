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
print(dados[provas].boxplot(grid=True, figsize=(8, 6)))

nomes_e_notas = {'cn': [[], []], 'ch': [[], []], 'mt': [[], []], 'lc': [[], []], 'red': [[], []]}

print(len(dados[provas]))
print('1*', dados[provas])
print('nome*', dados[provas]['NU_NOTA_CN'])
print('2*', dados[provas].values)
print('3*', dados[provas].values[0])
print('4*', dados[provas].values[1])

for disciplina in provas:

    if disciplina == 'NU_NOTA_CN':
        for nome in dados[provas[0]].keys():
            nomes_e_notas['cn'][0].append(nome)

        for valor in dados[provas[0]].values:
            nomes_e_notas['cn'][1].append(valor)

    if disciplina == 'NU_NOTA_CH':
        for nome in dados[provas[1]].keys():
            nomes_e_notas['ch'][0].append(nome)

        for valor in dados[provas[1]].values:
            nomes_e_notas['ch'][1].append(valor)

    if disciplina == 'NU_NOTA_MT':
        for nome in dados[provas[2]].keys():
            nomes_e_notas['mt'][0].append(nome)

        for valor in dados[provas[2]].values:
            nomes_e_notas['mt'][1].append(valor)

    if disciplina == 'NU_NOTA_LC':
        for nome in dados[provas[3]].keys():
            nomes_e_notas['lc'][0].append(nome)

        for valor in dados[provas[3]].values:
            nomes_e_notas['lc'][1].append(valor)

    if disciplina == 'NU_NOTA_REDACAO':
        for nome in dados[provas[4]].keys():
            nomes_e_notas['red'][0].append(nome)

        for valor in dados[provas[4]].values:
            nomes_e_notas['red'][1].append(valor)

plt.figure(figsize=(6, 4))
plt.plot(nomes_e_notas['cn'][0], nomes_e_notas['cn'][1])
plt.plot(nomes_e_notas['ch'][0], nomes_e_notas['ch'][1])
plt.plot(nomes_e_notas['mt'][0], nomes_e_notas['mt'][1])
plt.plot(nomes_e_notas['lc'][0], nomes_e_notas['lc'][1])
plt.plot(nomes_e_notas['red'][0], nomes_e_notas['red'][1])

plt.xlabel('disciplinas')
plt.ylabel('valores')
plt.grid(True)
# plt.savefig('reta-simples-duas.png')

plt.show()

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

# desafio5: Comparar as distribuições das provas em inglês e espanhol
# desafio6: Explorar as documentações e visualizações com mathplotlib ou pandas e gerar novas visualizações.

print('22------------------------')
# 2ª Aula

print(dados['NU_IDADE'])

print(dados.query('NU_IDADE == 13'))

print('23------------------------')
print(dados.query('NU_IDADE <= 14'))

print('24------------------------')
print(dados.query('NU_IDADE <= 14')['SG_UF_RESIDENCIA'])
print(
    dados.query('NU_IDADE <= 14')['SG_UF_RESIDENCIA'].value_counts(normalize=True))  # traz a proporção para cada estado

print('25------------------------')
alunos_menor_14 = dados.query('NU_IDADE <= 14')
print(alunos_menor_14['SG_UF_RESIDENCIA'].value_counts().plot.pie(figsize=(8, 6)))
print(alunos_menor_14['SG_UF_RESIDENCIA'].value_counts().keys())
print(alunos_menor_14['SG_UF_RESIDENCIA'].value_counts().values)

estados = []
partes = []

for estado in alunos_menor_14['SG_UF_RESIDENCIA'].value_counts().keys():
    estados.append(estado)

for parte in alunos_menor_14['SG_UF_RESIDENCIA'].value_counts().values:
    partes.append(parte)

plt.figure(figsize=(6, 4))
plt.plot(estados, partes)

plt.xlabel('estados')
plt.ylabel('partes')
plt.grid(True)

plt.show()

print('26------------------------')
print(alunos_menor_14['SG_UF_RESIDENCIA'].value_counts(normalize=True).plot.bar(figsize=(8, 6)))

plt.figure(figsize=(6, 4))
plt.plot(estados, partes)

plt.xlabel('estados')
plt.ylabel('partes')
plt.grid(True)

plt.show()

# Mega desafio da Thainá: Pegar a amostra completa dos alunos de 13 e 14 anos.
# Desafio do guilherme: aumentar a amostra para alunos menor de idade e comparar a proporção por estado.

print('27------------------------')

import seaborn as sns

plt.figure(figsize=(8, 6))
print(sns.boxplot(x='Q006', y='NU_NOTA_MT', data=dados))
plt.title('Boxplot das notas de matemática pela renda')

print('1-', dados['Q006'].keys())
print('2-', dados['Q006'].keys()[0])
print('3-', dados['Q006'].keys()[1])
print('4-', dados['Q006'].values)
print('5-', dados['NU_NOTA_MT'].keys())
print('6-', dados['NU_NOTA_MT'].values)

rendas = []
notas_mat = []

for renda in dados['Q006'].values:
    rendas.append(renda)

for nota in dados['NU_NOTA_MT'].values:
    notas_mat.append(nota)

plt.figure(figsize=(6, 4))
plt.plot(rendas, notas_mat)

plt.xlabel('rendas')
plt.ylabel('notas mat')
plt.grid(True)

plt.show()

print('28------------------------')
renda_ordenada = dados['Q006'].unique()
renda_ordenada.sort()

print(sns.boxplot(x='Q006', y='NU_NOTA_MT', data=dados, order=renda_ordenada))  # fica ordenado agora

rendas2 = []
notas_mat2 = []

for renda in dados['Q006'].values:
    rendas2.append(renda)

for nota in dados['NU_NOTA_MT'].values:
    notas_mat2.append(nota)

plt.figure(figsize=(6, 4))
plt.plot(rendas2, notas_mat2)

plt.xlabel('rendas')
plt.ylabel('notas mat')
plt.grid(True)

plt.show()

print('29------------------------')
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

print(dados[provas].sum())  # soma de cada disciplina
print(dados[provas].sum(axis=1))  # soma invés de coluna, a linha

# criação de coluna
dados['NU_NOTA_TOTAL'] = dados[provas].sum(axis=1)
print(dados.head())

print(sns.boxplot(x="Q006", y="NU_NOTA_TOTAL", data=dados, order=renda_ordenada))
plt.title('Boxplot das notas totais pela renda')

rendas3 = []
notas_totais = []

for renda in dados['Q006'].values:
    rendas3.append(renda)

for nota in dados['NU_NOTA_TOTAL'].values:
    notas_totais.append(nota)

plt.figure(figsize=(6, 4))
plt.plot(rendas3, notas_totais)

plt.xlabel('rendas')
plt.ylabel('notas totais')
plt.grid(True)

plt.show()


print('30------------------------')
print('31------------------------')
print('32------------------------')
print('33------------------------')
print('34------------------------')
