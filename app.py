import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predictor",
    layout="wide"
)

help_texts_pt = {
    "area_worst": "A média da área das três MAIORES células encontradas. É uma forte medida de TAMANHO.",
    "concave points_worst": "A média dos três MAIORES valores de 'pontos côncavos'. Mede a IRREGULARIDADE da forma da célula.",
    "radius_worst": "A média do raio das três MAIORES células encontradas na amostra. Outra medida de TAMANHO.",
    "perimeter_worst": "A média do perímetro das três MAIORES células. Relacionado ao tamanho e à irregularidade.",
    "concave points_mean": "A média geral do número de reentrâncias (partes côncavas) no contorno das células.",
    "perimeter_mean": "O perímetro médio de todas as células analisadas na amostra.",
    "radius_mean": "O raio médio de todas as células analisadas na amostra.",
    "concavity_mean": "A média geral da profundidade das reentrâncias (concavidade) no contorno das células.",
    "area_mean": "A área média de todas as células analisadas na amostra.",
    "concavity_worst": "A média da profundidade de concavidade das três células MAIS irregulares."
}

labels_pt = {
    "area_worst": "Área (Pior)",
    "concave points_worst": "Pontos Côncavos (Pior)",
    "radius_worst": "Raio (Pior)",
    "perimeter_worst": "Perímetro (Pior)",
    "concave points_mean": "Pontos Côncavos (Média)",
    "perimeter_mean": "Perímetro (Média)",
    "radius_mean": "Raio (Média)",
    "concavity_mean": "Concavidade (Média)",
    "area_mean": "Área (Média)",
    "concavity_worst": "Concavidade (Pior)"
}

@st.cache_resource
def carregar_arquivos():
    try:
        modelo = joblib.load('modelo_random_forest.pkl')
        medianas = joblib.load('medianas_features.pkl')
        colunas_originais = list(medianas.index)
        return modelo, medianas, colunas_originais
    except FileNotFoundError:
        st.error("Erro: Arquivos 'modelo_random_forest.pkl' ou 'medianas_features.pkl' não encontrados.")
        st.error("Por favor, rode o script 'treinamento.py' primeiro para gerar os arquivos.")
        return None, None, None

modelo, medianas, colunas_originais = carregar_arquivos()

st.title('Predictor Breast Cancer Wisconsin (Diagnostic)')
st.markdown("**(Projeto de Estudo Acadêmico)**")
st.warning(
    "**AVISO IMPORTANTE:** Esta é uma ferramenta de estudo e não substitui um diagnóstico médico profissional. "
    "Os resultados são baseados em um modelo de Machine Learning e não devem ser usados para decisões clínicas."
)

st.divider()
st.sidebar.header("Parâmetros Principais")
st.sidebar.markdown("Ajuste os *sliders* com os valores conhecidos.")

features_principais = {
    "area_worst": (medianas['area_worst'] * 0.1, medianas['area_worst'] * 5, medianas['area_worst']),
    "concave points_worst": (0.0, 0.3, medianas['concave points_worst']),
    "radius_worst": (medianas['radius_worst'] * 0.2, medianas['radius_worst'] * 3, medianas['radius_worst']),
    "perimeter_worst": (medianas['perimeter_worst'] * 0.2, medianas['perimeter_worst'] * 3,medianas['perimeter_worst']),
    "concave points_mean": (0.0, 0.25, medianas['concave points_mean']),
    "perimeter_mean": (medianas['perimeter_mean'] * 0.2, medianas['perimeter_mean'] * 3, medianas['perimeter_mean']),
    "radius_mean": (medianas['radius_mean'] * 0.2, medianas['radius_mean'] * 3, medianas['radius_mean']),
    "concavity_mean": (0.0, 0.5, medianas['concavity_mean']),
    "area_mean": (medianas['area_mean'] * 0.1, medianas['area_mean'] * 5, medianas['area_mean']),
    "concavity_worst": (0.0, 1.3, medianas['concavity_worst']),
}

inputs_usuario = {}

for feature_key, (min_val, max_val, default_val) in features_principais.items():
    label_amigavel = labels_pt.get(feature_key, feature_key)
    help_text = help_texts_pt.get(feature_key)

    inputs_usuario[feature_key] = st.sidebar.slider(
        label=label_amigavel,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default_val),
        step=float((max_val - min_val) / 1000),
        help = help_text
    )

if st.sidebar.button("Verificar Possível Diagnóstico"):
    if modelo is not None:
        dados_completos = medianas.to_dict()

        for feature, valor in inputs_usuario.items():
            dados_completos[feature] = valor

        df_input = pd.DataFrame([dados_completos])
        df_input = df_input[colunas_originais]

        try:
            predicao = modelo.predict(df_input)
            probabilidade = modelo.predict_proba(df_input)

            resultado_numerico = predicao[0]
            confianca = probabilidade[0][resultado_numerico]

            st.subheader("Resultado da Análise")

            col1, col2 = st.columns(2)

            if resultado_numerico == 1:
                st.error(
                    "**Diagnóstico Previsto: Maligno**",
                )
            else:  # Benigno
                st.success(
                    "**Diagnóstico Previsto: Benigno**",
                )

            st.metric(
                label="Confiança do Modelo",
                value=f"{confianca * 100:.2f}%"
            )

            with st.expander("Ver detalhes dos dados enviados ao modelo"):
                st.dataframe(df_input)

        except Exception as e:
            st.error(f"Erro ao fazer a predição: {e}")
    else:
        st.error("Modelo não carregado. Verifique os arquivos.")