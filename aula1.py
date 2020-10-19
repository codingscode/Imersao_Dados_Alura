
import pandas as pd

#fonte = 'https://github.com/alura-cursos/imersao-dados-2-2020/blob/master/MICRODADOS_ENEM_2019_SAMPLE_43278.csv?raw=true'

# forma alternativa se já baixado
fonte = './nao_enviar/MICRODADOS_ENEM_2019_SAMPLE_43278.csv'


pd.options.display.width = None  #  todas as colunas horizontalmente
#pd.set_option('display.max_columns', None)


dados = pd.read_csv(fonte)
print(dados.head())
#print(dados.shape)   #  // (número de linhas, numero de colunas) -> (127380, 136)

print('1------------------------')

print(dados['NO_MUNICIPIO_NASCIMENTO'])  #  escolhendo uma coluna

print('2------------------------')
print(dados.columns.values)  # nomes das colunas

print('------------------------')
#print(dados['NO_MUNICIPIO_NASCIMENTO', 'Q025'])  # dá erro
print(dados[['NO_MUNICIPIO_NASCIMENTO', 'Q025']])  # 2 colunas

print('3------------------------')
print(dados['SG_UF_RESIDENCIA'].unique())  #  lista de UFs não repetidos
print(len(dados['SG_UF_RESIDENCIA'].unique()))

print('4------------------------')
print(dados['SG_UF_RESIDENCIA'].value_counts())  #  conta valores

print('5------------------------')
print(dados['NU_IDADE'].value_counts())  #  conta valores

print('6------------------------')
print(dados['NU_IDADE'].value_counts().sort_index())

# desafio1: proporção de inscritos por idade
# desafio2: Descobrir de quais estados são os inscritos com 13 anos.

print('7------------------------')


print('8------------------------')
print('9------------------------')
print('10------------------------')
print('11------------------------')
print('12------------------------')
print('13------------------------')

