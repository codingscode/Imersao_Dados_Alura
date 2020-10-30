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
print(
    dados.query('NU_IDADE <= 14')['SG_UF_RESIDENCIA'].value_counts(normalize=True))  # traz a proporção para cada estado

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
# desafio7: criar uma função para plotar o boxplot do seaborn
print(sns.displot(dados, x='NU_NOTA_TOTAL'))
plt.show()

print('31------------------------')
provas.append('NU_NOTA_TOTAL')
print(dados.query('NU_NOTA_TOTAL == 0'))

print('32------------------------')
print(dados[provas].query('NU_NOTA_TOTAL == 0'))  # aparece NaN
# desafio8: Verificar se quem zerou a prova foi eliminado ou não estava presente
# desafio9: quem é eliminado tira zero ou será NaN(não teve registro de notas)

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
print(sns.histplot(dados_sem_notas_zero, x='NU_NOTA_LC'))
plt.show()

print('38------------------------')
# desafio10 : Plotar as médias, medianas e moda nas notas de LC e MT(matplotlib linha vertical)

print(sns.histplot(dados_sem_notas_zero, x='NU_NOTA_TOTAL', hue='Q025', kde=True))  # kde cria uma linha
plt.show()
print(sns.histplot(dados_sem_notas_zero, x='NU_NOTA_TOTAL', hue='Q025', kde=True,
                   cumulative=True))  # parametro stat para 'probability', 'density' nao estao funcionando grafico doido
plt.show()

print('39------------------------')
sns.scatterplot(data=dados_sem_notas_zero, x='NU_NOTA_MT', y='NU_NOTA_LC')
plt.xlim((-50, 1050))  # experimentar comentar
plt.ylim((-50, 1050))  # experi mentar comentar
plt.show()

print('40------------------------')
# print(sns.pairplot(dados_sem_notas_zero[provas])) # descometar urgente
# plt.show() # descometar urgente
# pegar as 6 disciplinas e fazer comparação de gráficos em pares

print('41------------------------')
correlacao = dados_sem_notas_zero[provas].corr()
print(correlacao)  # vai de -1 a 1

print('42------------------------')
print(sns.heatmap(correlacao, cmap='Blues'))
plt.show()
print(sns.heatmap(correlacao, cmap='Blues', center=0))
plt.show()
print(sns.heatmap(correlacao, cmap='Blues', center=0, annot=True))
plt.show()

print('43------------------------')
provas_entrada = ['NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_CN', 'NU_NOTA_REDACAO']
provas_saida = 'NU_NOTA_MT'
dados_sem_notas_zero = dados_sem_notas_zero[provas].dropna()
notas_entrada = dados_sem_notas_zero[provas_entrada]
notas_saida = dados_sem_notas_zero[provas_saida]

print(notas_entrada)

print('44------------------------')
x = notas_entrada
y = notas_saida

from sklearn.model_selection import train_test_split

SEED = 4321
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.25, random_state=SEED)  # agora é constante

print(x_treino.head())
print(x_treino.shape)
print(len(y_treino))
print(x_teste.shape)
print(len(y_teste.shape))

print('45------------------------')
from sklearn.svm import LinearSVR

modelo = LinearSVR(random_state=SEED)  # há erro pois na tabela há NaN
modelo.fit(x_treino, y_treino)
predicoes_matematica = modelo.predict(x_teste)

print(y_teste[:5])

print('46------------------------')
print(sns.scatterplot(x=y_teste, y=y_teste - predicoes_matematica))
plt.xlim(-50, 1050)
plt.ylim(-50, 1050)
plt.show()

print('47------------------------')
print(sns.scatterplot(x=y_teste, y=y_teste - x_teste.mean(axis=1)))
plt.show()

print('48------------------------')
resultados = pd.DataFrame()
resultados['Real'] = y_teste
resultados['Previsao'] = predicoes_matematica
resultados['diferenca'] = resultados['Real'] - resultados['Previsao']
resultados['quadrado_diferenca'] = (resultados['Real'] - resultados['Previsao'])**2

print(resultados)

print('49------------------------')
print(resultados['quadrado_diferenca'].mean())
print(resultados['quadrado_diferenca'].mean() ** 0.5)

print('50------------------------')
from sklearn.dummy import DummyRegressor

modelo_dummy = DummyRegressor()
modelo_dummy.fit(x_treino, y_treino)
dummy_predicoes = modelo_dummy.predict(x_teste)

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_teste, dummy_predicoes))
print(mean_squared_error(y_teste, predicoes_matematica))

print('51------------------------')
# Aula 5
from sklearn.tree import DecisionTreeRegressor

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.25)
modelo_arvore = DecisionTreeRegressor(max_depth=3)
modelo_arvore.fit(x_treino, y_treino)
predicoes_matematica_arvore = modelo_arvore.predict(x_teste)

print(mean_squared_error(y_teste, predicoes_matematica_arvore))

print('52------------------------')
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

partes = KFold(n_splits=10, shuffle=True)

resultados = cross_validate(modelo_arvore, x, y, cv=partes, scoring='neg_mean_squared_error')
media = (resultados['test_score']*-1).mean()
print(media)
print(resultados['test_score']*-1)

print('53------------------------')
desvio_padrao = (resultados['test_score']*-1).std()
lim_inferior = media - (2*desvio_padrao)
lim_superior = media + (2*desvio_padrao)

print(f'Intervalo de confiança {lim_inferior} - {lim_superior}')  # traz sempre o mesmo resultado

print('54------------------------')


def regressor_arvore(nivel):
    SEED=1232
    np.random.seed(SEED)
    partes = KFold(n_splits=10, shuffle=True)
    modelo_arvore = DecisionTreeRegressor(max_depth=nivel)
    resultados = cross_validate(modelo_arvore, x, y, cv=partes, scoring='neg_mean_squared_error', return_train_score=True)
    print(f'Treino = {(resultados["train_score"] * -1).mean()} | Teste = {(resultados["test_score"] * -1).mean()}')


for i in range(1, 21):
    regressor_arvore(i)

print('55------------------------')
print('56------------------------')
print('57------------------------')
