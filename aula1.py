
import pandas as pd

#fonte = 'https://github.com/alura-cursos/imersao-dados-2-2020/blob/master/MICRODADOS_ENEM_2019_SAMPLE_43278.csv?raw=true'
# forma alternativa se já baixado 
fonte = './nao_enviar/MICRODADOS_ENEM_2019_SAMPLE_43278.csv'


pd.set_option('display.max_columns', None)


dados = pd.read_csv(fonte)
print(dados.head())









