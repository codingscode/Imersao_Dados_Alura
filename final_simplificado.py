import pandas as pd, numpy as np, matplotlib.pyplot as plt

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
print(dados['NU_IDADE'].hist())
print(dados['NU_IDADE'].hist(bins=20, figsize=(8, 6)))
dados['NU_IDADE'].hist()

print('8------------------------')
print(dados['NU_IDADE'])
print(type(dados['NU_IDADE']))
print(dados['NU_IDADE'].value_counts().sort_index())
print('**')
print(dados['NU_IDADE'].value_counts().sort_index().keys())
print(dados['NU_IDADE'].value_counts().sort_index().keys()[0])
print(dados['NU_IDADE'].value_counts().sort_index().values)
print(dados['NU_IDADE'].value_counts().sort_index().values[0])
plt.show()
# desafio3: Adicionar título no gráfico

print('9------------------------')
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
plt.show()

print('15------------------------')
dados['NU_NOTA_LC'].hist(bins=20, figsize=(8, 6))
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
plt.show()

print('21------------------------')
print(dados[provas].boxplot(grid=True, figsize=(8, 6)))
plt.show()

print('22------------------------')
# 2ª Aula

print(dados['NU_IDADE'])
print(dados.query('NU_IDADE == 13'))

print('23------------------------')
print(dados.query('NU_IDADE <= 14'))

print('24------------------------')
print(dados.query('NU_IDADE <= 14')['SG_UF_RESIDENCIA'])
print(dados.query('NU_IDADE <= 14')['SG_UF_RESIDENCIA'].value_counts(normalize=True)) # traz a proporção para cada estado

print('25------------------------')
alunos_menor_14 = dados.query('NU_IDADE <= 14')
print(alunos_menor_14['SG_UF_RESIDENCIA'].value_counts().plot.pie(figsize=(8, 6)))
plt.show()

print('26------------------------')
print(alunos_menor_14['SG_UF_RESIDENCIA'].value_counts().plot.bar(figsize=(8, 6)))
plt.show()
# Mega desafio da Thainá: Pegar a amostra completa dos alunos de 13 e 14 anos.
# Desafio do guilherme: aumentar a amostra para alunos menor de idade e comparar a proporção por estado.

print('27------------------------')
import seaborn as sns

plt.figure(figsize=(8, 6))
print(sns.boxplot(x='Q006', y='NU_NOTA_MT', data=dados))
plt.title('Boxplot das notas de matemática pela renda')
plt.show()

print('28------------------------')
renda_ordenada = dados['Q006'].unique()
renda_ordenada.sort()

print(sns.boxplot(x='Q006', y='NU_NOTA_MT', data=dados, order=renda_ordenada))  # fica ordenado agora
plt.show()

print('29------------------------')
print('*', dados[provas].sum())  # soma de cada disciplina
print('**', dados[provas].sum(axis=1))  # soma invés de coluna, a linha

# criação de coluna
dados['NU_NOTA_TOTAL'] = dados[provas].sum(axis=1)
print(dados.head())

print(sns.boxplot(x="Q006", y="NU_NOTA_TOTAL", data=dados, order=renda_ordenada))
plt.title('Boxplot das notas totais pela renda')
plt.show()

print('30------------------------')
#desafio7: criar uma função para plotar o boxplot do seaborn
print(sns.displot(dados, x='NU_NOTA_TOTAL'))
plt.show()

print('31------------------------')
provas.append('NU_NOTA_TOTAL')
print(dados.query('NU_NOTA_TOTAL == 0'))

print('32------------------------')
print(dados[provas].query('NU_NOTA_TOTAL == 0'))  # aparece NaN
#desafio8: Verificar se quem zerou a prova foi eliminado ou não estava presente
#desafio9: quem é eliminado tira zero ou será NaN(não teve registro de notas)

print('33------------------------')
dados_sem_notas_zero = dados.query('NU_NOTA_TOTAL != 0')
print(dados_sem_notas_zero.head())

print('34------------------------')
print(sns.boxplot(x="Q006", y="NU_NOTA_TOTAL", data=dados_sem_notas_zero, hue='IN_TREINEIRO', order=renda_ordenada))
plt.title('Boxplot das notas totais pela renda')
plt.show()
# desafio8: Verificar a proporção dos participantes de rendas mais altas e mais baixas como treineiro e não treineiro.
# desafio9: fazer o mesmo boxplot olhando para a questão 25(Q025?) (tem internet ou não) e fazer uma reflexão sobre o assunto e o contexto de pandemia.

print('35------------------------')
# 3ª Aula
print(sns.histplot(dados_sem_notas_zero, x='NU_NOTA_TOTAL'))
plt.show()

print('36------------------------')
print(sns.histplot(dados_sem_notas_zero, x='NU_NOTA_MT'))
print(dados_sem_notas_zero['NU_NOTA_MT'])
plt.show()

print('37------------------------')
print('38------------------------')
print('39------------------------')
