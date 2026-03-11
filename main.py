import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

import warnings
warnings.filterwarnings('ignore')

# Configurações visuais
st.set_page_config(page_title="Análise e Previsão de Alta Pressão (SARIMAX)", layout="wide")

try:
    st.image("logo.png", width=320)
except FileNotFoundError:
    pass

st.title("Análise e Previsão de Alta Pressão (SARIMAX)")


# Upload do dataset
st.header("Upload do Dataset")

uploaded_file = st.file_uploader(
    "Envie o arquivo CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Faça o upload do arquivo para continuar.")
    st.stop()

df = pd.read_csv(uploaded_file, delimiter=";")
st.success("Dataset carregado com sucesso!")


# Preparação dos Dados (Datas)
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df.set_index("Date", inplace=True)

# Tratamento de valores infinitos ou nulos antes da análise
df['High_Pressure'].replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['High_Pressure'], inplace=True)


# Visualização inicial
st.header("Visualização dos dados")

num_linhas = st.slider(
    "Quantidade de linhas para visualizar",
    min_value=5,
    max_value=50,
    value=10
)

st.dataframe(df.head(num_linhas))


# Análise Exploratória
st.header("Análise Exploratória de Dados")

st.subheader("Série Histórica e Médias Móveis")

janela_meses = st.slider("Tamanho da janela (Rolling window)", 2, 12, 4)

df["rolling_avg"] = df["High_Pressure"].rolling(window=janela_meses).mean()
df["rolling_std"] = df["High_Pressure"].rolling(window=janela_meses).std()

fig1, ax1 = plt.subplots(figsize=(15, 6))
ax1.plot(df.index, df["High_Pressure"], color='#379BDB', label='Original')
ax1.plot(df.index, df["rolling_avg"], color='#D22A0D', label='Média móvel')
ax1.plot(df.index, df["rolling_std"], color='#142039', label='Desvio padrão')
ax1.set_xlabel('Data')
ax1.set_ylabel('High Pressure')
ax1.legend(loc='best')
ax1.set_title(f'Série Original vs Média/Desvio (Janela={janela_meses})')
st.pyplot(fig1)
plt.clf()


# Teste de Estacionariedade
st.subheader("Teste Dickey-Fuller Aumentado")

if st.button("Realizar Teste ADF"):
    dftest = adfuller(df['High_Pressure'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Estatística de Teste','Valor p','# Lags Usados','Nº Observações Usadas'])
    for key, value in dftest[4].items():
        dfoutput[f'Valor Crítico ({key})'] = value
    st.dataframe(dfoutput)


# Configuração do modelo
st.header("Configuração do Modelo SARIMAX")
st.write("Ajuste as ordens do modelo Auto Regressivo Integrado de Médias Móveis Sazonais.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ordem ARIMA (p, d, q)")
    p = st.number_input("p (Auto-Regressivo)", 0, 5, 1)
    d = st.number_input("d (Diferenciação)", 0, 3, 1)
    q = st.number_input("q (Média Móvel)", 0, 5, 0)

with col2:
    st.subheader("Ordem Sazonal (P, D, Q, s)")
    P = st.number_input("P (Sazonal AR)", 0, 5, 1)
    D = st.number_input("D (Sazonal Diff)", 0, 3, 1)
    Q = st.number_input("Q (Sazonal MA)", 0, 5, 0)
    s = st.number_input("s (Períodos Sazonais)", 0, 30, 7)


# Treino do Modelo
st.header("Treinamento e Avaliação")

if st.button("Treinar Modelo"):
    with st.spinner("Treinando o modelo SARIMAX..."):
        model = SARIMAX(
            df["High_Pressure"], 
            order=(p, d, q), 
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False, 
            enforce_invertibility=False
        )
        SARIMAX_model = model.fit()
        
        st.session_state['modelo_sarimax'] = SARIMAX_model
        
        st.success("Modelo treinado com sucesso!")

        # Diagnóstico
        st.subheader("Diagnóstico do Modelo")
        fig_diag = SARIMAX_model.plot_diagnostics(figsize=(15, 10))
        st.pyplot(fig_diag)
        plt.clf()

        st.subheader("Resumo Estatístico")
        st.text(SARIMAX_model.summary())


# Previsão
st.header("Previsão de novo cenário (Forecasting)")

forecast_steps = st.slider("Dias no futuro para prever:", min_value=1, max_value=60, value=20)

if st.button("Gerar Previsão"):
    if 'modelo_sarimax' not in st.session_state:
        st.warning("Você precisa treinar o modelo primeiro.")
    else:
        modelo_treinado = st.session_state['modelo_sarimax']
        
        forecast = modelo_treinado.get_forecast(steps=forecast_steps)
        forecast_values = forecast.predicted_mean
        
        ultimo_dia = df.index[-1]
        forecast_index = pd.date_range(start=ultimo_dia + pd.Timedelta(days=1), periods=forecast_steps, freq="D")

        # Gráfico da previsão
        fig_prev, ax_prev = plt.subplots(figsize=(15, 6))
        ax_prev.plot(df.index, df['High_Pressure'], label="Dados Históricos", color='blue')
        ax_prev.plot(forecast_index, forecast_values, label="Previsão SARIMAX", color='red', linestyle="dashed")
        
        ax_prev.set_xlabel("Data")
        ax_prev.set_ylabel("High Pressure")
        ax_prev.legend()
        ax_prev.set_title(f"Previsão de High Pressure para os próximos {forecast_steps} dias")
        ax_prev.grid()

        ax_prev.xaxis.set_major_locator(mdates.DayLocator(interval=1)) 
        ax_prev.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=90)

        plt.tight_layout()
        st.pyplot(fig_prev)
        plt.clf()
        
        st.success("Previsão gerada com sucesso!")