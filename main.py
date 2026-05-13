import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.header("Previsão de Vendas")

# Dados: [Investimento em Marketing] -> Faturamento
dados_vendas = pd.DataFrame({
    'investimento': [100, 200, 300, 400, 500, 600],
    'faturamento': [1200, 2500, 3200, 4800, 5100, 6300]
})

# objetivo: previsão de FATURAMENTO 


modelofat = LinearRegression() 
modelofat.fit(dados_vendas[['investimento']], dados_vendas['faturamento'])
fat = st.slider('investimento', 0,7000,1200)
fat_final = modelofat.predict([[fat]])
print(fat_final)


st.metric(f'faturamento' ,f'{max(fat_final[0], 10.0):.2f}')