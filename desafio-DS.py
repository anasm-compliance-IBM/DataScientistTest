
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
file_sla = "/Users/anasilviamorettobraga/DataScientistTest/dataset/ticket_cientista1.csv"
## arquivo csv separado por ponto e vírgula, incluir parametro ##
df = pd.read_csv(file_sla, sep=';', low_memory=False)
##df


# In[2]:


##df.dtypes


# In[3]:


## totalização -  onTimeSolution por customer code
df_gp1 = df[['customerCode','onTimeSolution']]                        
df_gp2 =  df_gp1.groupby(['customerCode'],as_index=False).count()
##df_gp2


# In[4]:


## dataframe original com numero de colunas reduzidas
df1= df[['closeDateTime', 'customerCode','onTimeSolution','openDateTime','end_date','currentDateTime']]
##df1.head()


# In[5]:


## totalização missing data do dataframe df1 (reduzido)
missing_data1 = df1.isnull()
for column in missing_data1.columns.values.tolist(): print(missing_data1[column].value_counts())
print("")


# In[6]:


##totalização de ontimeSolution por data e por cliente
df_gp3 = df[['end_date','customerCode','onTimeSolution']]                        
df_gp4 =  df_gp3.groupby(['end_date','customerCode'],as_index=False).count()
df_gp4['end_date'] = pd.to_datetime(df_gp4['end_date'], format='%d/%m/%Y') ### mudar formato de data
##df_gp4 = df_gp4.sort_values(['end_date'])### organizar por data
##df_gp4


# In[7]:


## criação do dataframe data, cliente, OnTimeSolution = Total - step01
df_gp4['ref'] =df_gp4['onTimeSolution']
##df_gp4


# In[8]:


## criação do dataframe data, cliente, OnTimeSolution = Total - step02
df_gp41 = df_gp4
df_gp41['onTimeSolution']='T'
df_gp42= df_gp41[df_gp41.onTimeSolution == 'T']
df_gp42 = df_gp41.rename(columns={'onTimeSolution': 'onTimeT'})
df_gp42['onTimeT'] = df_gp42['ref']


# In[9]:


## criação do dataframe data, cliente, OnTimeSolution = Total - step03
df_gp43=df_gp42.drop('ref', axis=1)
##df_gp43


# In[10]:


## totalizacao por data, cliente, tipo onTimeSolution (sim ou nao)
df_gp5 = df[['end_date','customerCode','onTimeSolution','ref']]                        
df_gp6 =  df_gp5.groupby(['end_date','customerCode','onTimeSolution'],as_index=False).count()
df_gp6['end_date'] = pd.to_datetime(df_gp6['end_date'], format='%d/%m/%Y') ### mudar formato de data
##df_gp6 = df_gp6.sort_values(['end_date'])### organizar por data
##df_gp6


# In[11]:


## contatenação dos dataframes "S", "N", "T" 
frames = [df_gp41, df_gp6]
result = pd.concat(frames)


# In[12]:


## ordenando arquivo concatenado
result = result.sort_values(['end_date', 'customerCode'])
##result


# In[13]:


#result1 = result
#result1['onTimeS'] =0
#result1['onTimeN'] =0
#result1['onTimeT'] =0
##result1


# In[14]:


## criação do dataframe data, cliente, OnTimeSolution = Sim - step01
df_gp10= result[result.onTimeSolution == 'S']
##df_gp10


# In[15]:


df_gp11= df_gp6[df_gp6.onTimeSolution == 'N']
df_gp11
df_gp1102 = df_gp11.rename(columns={'onTimeSolution': 'onTimeN'})
df_gp1102['onTimeN'] = df_gp1102['ref']


# In[16]:


df_gp1103=df_gp1102.drop('ref', axis=1)
##df_gp1103


# In[17]:


a=result.loc[result['onTimeSolution'] == 'S']
a = a.reset_index()
a['onTimeS'] = a['ref']
##a


# In[18]:


b=result.loc[result['onTimeSolution'] == 'N']
b = b.reset_index()
b['onTimeN'] = b['ref']
##b


# In[19]:


c=result.loc[result['onTimeSolution'] == 'T']
c = c.reset_index()
c['onTimeT'] = c['ref']
##c


# In[20]:


frms = [a, b, c]
result2 = pd.concat(frms)
##result2


# In[21]:


result2_transposed = result2.T # or df1.transpose()
##result2_transposed


# In[22]:


result3 = result2
##result3


# In[23]:


result4=result3.drop('ref', axis=1)
result5 = result4.drop('onTimeSolution', axis=1)
##result5


# In[24]:


result6 = result5[['end_date','customerCode','onTimeS','onTimeN','onTimeT']]                        
result7 =  result6.groupby(['end_date','customerCode','onTimeS','onTimeT'],as_index=False).count()
result7['end_date'] = pd.to_datetime(result7['end_date'], format='%d/%m/%Y') ### mudar formato de data
##result7 = result7.sort_values(['end_date'])### organizar por data
##result7


# In[25]:


##result7_transposed = result7.T # or df1.transpose()
##result7_transposed


# In[26]:


df_gp12= df_gp6[df_gp6.onTimeSolution == 'S']
##df_gp12


# In[27]:


df_gp1202 = df_gp12.rename(columns={'onTimeSolution': 'onTimeS'})
df_gp1202['onTimeS'] = df_gp1202['ref']
##df_gp1202


# In[28]:


df_gp1203=df_gp1202.drop('ref', axis=1)
##df_gp1203


# In[29]:


## merging df_gp43 && df_gp1103 && df_gp1203
result8 = pd.merge(df_gp43,df_gp1103, on=['end_date', 'customerCode'])
##result8


# In[30]:


## merging df_gp43 && df_gp1103 && df_gp1203
##result9 = pd.merge(df_gp43, df_gp1203, df_gp1103, on=['end_date', 'customerCode'])
##result9


# In[31]:


## criacao da coluna onTimeS
result8['onTimeS'] = result8['onTimeT'] - result8['onTimeN']
##result8


# In[32]:


## criacao da coluna SLA_day
result8['SLA_day'] = result8['onTimeS'] / result8['onTimeT']
##result8


# In[33]:


## SLA média geral de todos os clientes
##result8['SLA_day'].mean()


# In[34]:


df_result01 = result8
##df_result01


# In[35]:


## resumo do cliente 372301 de janeiro a fevereiro:
result_C_372301 = df_result01.loc[df_result01['customerCode'] == 372301]
result_C_372301 = result_C_372301.sort_values(['end_date'])### organizar por data
##result_C_372301


# In[36]:


## MEDIA GERAL DO CLIENTE 372301 - CONSIDERANDO JANEIRO E FEVEREIRO
result_C_372301['SLA_day'].mean()


# In[37]:


df_jan372301 = result_C_372301


# In[38]:


## mes de janeiro cliente 372301

import datetime
data_inicio = datetime.datetime(2019, 1, 1)
data_final = datetime.datetime(2019, 1, 31)
jan_372301 = df_jan372301[(data_inicio <= df_jan372301["end_date"]) &  (data_final > df_jan372301["end_date"])]
jan_372301 = jan_372301.sort_values(['end_date'])### organizar por data
jan_372301


# In[39]:


## primeira quinzena janeiro cliente 372301

import datetime
data_inicio = datetime.datetime(2019, 1, 1)
data_final = datetime.datetime(2019, 1, 16)
jan_1q_372301 = df_jan372301[(data_inicio <= df_jan372301["end_date"]) &  (data_final > df_jan372301["end_date"])]
jan_1q_372301 = jan_1q_372301.sort_values(['end_date'])### organizar por data
jan_1q_372301


# In[40]:


# MEDIA GERAL DO CLIENTE 372301 - CONSIDERANDO APENAS primeira quinzena de JANEIRO
jan_1q_372301['SLA_day'].mean()


# In[41]:


# MEDIA GERAL DO CLIENTE 372301 - CONSIDERANDO APENAS JANEIRO
jan_372301['SLA_day'].mean()


# In[42]:


## vamos olhar o mes de fevereiro cliente 372301

import datetime
data_inicio = datetime.datetime(2019, 2, 1)
data_final = datetime.datetime(2019, 2, 28)
feb_372301 = result_C_372301[(data_inicio <= result_C_372301["end_date"]) &  (data_final > result_C_372301["end_date"])]
feb_372301 = feb_372301.sort_values(['end_date'])### organizar por data
feb_372301


# In[43]:


# MEDIA GERAL DO CLIENTE 372301 - CONSIDERANDO APENAS fevereiro
feb_372301['SLA_day'].mean()


# In[44]:


### RESULTADO IMPORTANTE:
## MÉDIA DE JANEIRO E FEVEREIRO: 0.6627396267609257
## MÉDIA DE JANEIRO SOMENTE: 0.6664559265782145
## MÉDIA PRIMEIRA QUINZENA DE JANEIRO: 0.6599447881716412
## MÉDIA DE FEVEREIRO SOMENTE: 0.6599763705246426

## NA MEDIA GERAL DO MES, OU QUINZENA, O VALOR É MUITO PROXIMO A 0.66. 
## AVALIANDO COMO UMA SÉRIE TEMPORAL, EM PERIODOS MENORES QUE O MES (QUINZENA) E TENDO RESULTADOS DE MESES
    ## ANTERIORES, É POSSIVEL TER UMA BOA PREVISAO DO TARGET MENSAL. 

### AQUI OS DADOS AINDA NÃO ESTÃO NORMALIZADOS

## Abaixo tentamos desenvoler um modelo de regressão linear, entretanto, o mesmo não se demonstrou viável. 
## também verificamos não haver correlação do SLA com o dia do mes, ou seja, o SLA diário é aleatório e não temos 
  ## no momento uma variável que demonstra que determinado dia do mes tem um aumento / decrescimento.
  ## Isto nos reforça o entendimento de que é possivel prever uma média por período, 
  ## mas nao há como ter uma previsão diária. 


# In[45]:


## Vamos criar um dataframe em formato número para plotar alguns gráficos:
##organizar por data
df_aux1 = result_C_372301
df_aux1 = df_aux1.sort_values(['end_date'])
##df_aux1


# In[46]:


##resetar o indice
df_aux1= df_aux1.reset_index(drop=True)
##df_aux1


# In[47]:


dfgraph= df_aux1[['end_date','SLA_day']]
##dfgraph


# In[48]:


df_aux3 = pd.DataFrame(columns=['nr_dias'])
##df_aux3


# In[49]:


## convertendo a data em numero de dias (float64)

ano= 2019       #formato AAAA
mes=  1         #usar numeros
dia= 1

import datetime
data_inicio = datetime.date(ano, mes, dia)
dfgraph['dias'] = dfgraph['end_date'] - data_inicio
df_aux3['nr_dias'] = dfgraph['dias']/ np.timedelta64(1, 'D')
##df_aux3


# In[50]:


##df_aux3 = dfgraph[['end_date']] 
df_aux3['end_date'] = '0'
df_aux3['end_date'] = dfgraph['end_date']
##df_aux3


# In[51]:


dfgraph1 = pd.merge(dfgraph,df_aux3,on=['end_date'])
##dfgraph1


# In[52]:


dfgraph2 = pd.DataFrame(columns=['SLA_day'])
dfgraph2['SLA_day'] = dfgraph1['SLA_day']
##dfgraph2


# In[53]:


dfgraph3 = pd.DataFrame(columns=['SLA_day','nr_dias'])
dfgraph3['SLA_day'] = dfgraph1['SLA_day']
dfgraph3['nr_dias'] = dfgraph1['nr_dias']
##dfgraph3


# In[54]:


## normalizacao da coluna data:
dfgraph3['nr_dias'] = dfgraph3['nr_dias']/dfgraph3['nr_dias'].max() 


# In[55]:


## normalizacao da coluna SLA_day:
dfgraph3['SLA_day'] = dfgraph3['SLA_day']/dfgraph3['SLA_day'].max() 
##dfgraph3


# ## VISUALIZACAO DOS DADOS EM GRÁFICO - DISTRIBUICAO DIÁRIA

# In[56]:


import matplotlib.pyplot as plt
plt.figure(); dfgraph2.plot(); plt.legend(loc='best')


# In[57]:


dftime = pd.DataFrame(columns=['SLA_day'])
dftime['SLA_day'] = dfgraph3['SLA_day']
dftime['nr_dias'] = '0'
dftime['nr_dias'] = dfgraph3['nr_dias']
##dftime


# In[58]:


# Variáveis para o Bar Chart
y_axis = dftime['SLA_day']
x_axis = range(len(y_axis))
width_n = 0.4
bar_color = 'blue'
plt.bar(x_axis, y_axis, width=width_n, color=bar_color)
plt.show()


# In[59]:


scatter_plot = plt.scatter(dftime['nr_dias'], dftime['SLA_day'], alpha=0.9, c=dftime['SLA_day'])
plt.show()


# In[60]:


## existe correlação entre data e SLA??
dftime[['SLA_day', 'nr_dias']].corr() 

### resultado muito proximo de zero, não existe correlação


# In[61]:


dftime.describe()


# In[62]:


from scipy import stats


# In[63]:


### VERIFICAR SE EXISTE ALGUMA CORRELAÇÃO ENTRE DATA E SLA DIARIO
from scipy import stats


# In[64]:


pearson_coef, p_value = stats.pearsonr(dftime['nr_dias'], dftime['SLA_day'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

## CONCLUSAO:
   ## TEMOS p-value >0.001, ENTÃO A CORRELAÇÃO ENTRE SLA and nr_dias NÃO É SIGNIFICANTE
      ## A CORRELACAO PEARSON TB NOS INFORMA QUE A RELAÇÃO LINEAR É INSIGNIFICANTE (0.007) - ENTRE 0 E 0.3


# In[65]:


## APESAR DOS RESULTADOS VAMOS ANALISAR UM MODELO DE REGRESSAO LINEAR
from sklearn.linear_model import LinearRegression


# In[66]:


lm = LinearRegression()
lm


# In[67]:


## Vamos gerar uma funcao linear com "SLA_days" sendo predictor variável e "nr_days" a variavel resposta.
   ### os dados de dftime sao normalizados

X = dftime[['nr_dias']]
Y = dftime['SLA_day']


# In[68]:


lm.fit(X,Y)


# In[69]:


## saida prediccao
Yhat=lm.predict(X)
Yhat[0:5] 


# In[70]:


## What is the value of the intercept (a)?
lm.intercept_


# In[71]:


## What is the value of the Slope (b)?
lm.coef_


# ## Portanto, nossa equação linear seria:
# ## SLA_day = 0.8024 - 0.00178295 x nr_dias

# In[72]:


dftime.mean()


# In[73]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[74]:


## Regression Plot - Vamos visualizar nosso modelo num gráfico 
 ### (a equação encontrada juntamente com os pontos que temos)
    ### veja que o gráfico confirma a inexistencia de correlação entre SLA e dia

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="nr_dias", y="SLA_day", data=dftime)
plt.ylim(0,)


# In[75]:


## Residual Plot
## AQUI O OBJETIVO É ANALISAR A VARIANCIA DOS DADOS (A DIFERENCA ENTRE Y REAL E Y PREDICTED (YHAT))
### SE A VARIANCIA DOS PONTOS ESPALHADOS É CONSTANTE, ENTAO, UM MODELO LINEAR PODE SER APROPRIADO
    
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(dftime['nr_dias'], dftime['SLA_day'])
plt.show()


# In[76]:


## analisando o grafico acima, não é possivel visualizar um contorno de curvas para um modelo linear ou polinomio.
## Veja nosso modelo linear comparado aos dados reais


# In[77]:


Z = dftime[['nr_dias']]


# In[78]:


lm.fit(Z, dftime['SLA_day'])


# In[79]:


plt.figure(figsize=(width, height))

ax1 = sns.distplot(dftime['SLA_day'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted days for SLA')
plt.xlabel('SLA')
plt.ylabel('')

plt.show()
plt.close()


# In[80]:


### CONCLUSAO: DENTRO DO JÁ EXPOSTO ACIMA, O MODELO DE REGRESSÃO LINEAR NÃO TEM UM DESEMPENHO ADEQUADO. 
### TAMBÉM NAO IDENTIFICAMOS CORRELAÇÃO ENTRE DATA E SLA
### NO ENTANTO, O VALOR DA MÉDIA DO SLA EM PERIODOS MAIS CURTOS QUE UM MES SE MOSTRA VIÁVEL E NO MOMENTO PARECE
    ### SER O MODO MAIS EFICIENTE DE PREVISAO. 

