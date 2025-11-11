# 🔬 Classificação de tumores (Projeto de Estudo)

Este é um projeto acadêmico de Machine Learning. O objetivo foi criar uma aplicação web interativa que utiliza um modelo treinado para classificar tumores como benignos ou malignos, com base no dataset "Wisconsin Breast Cancer" (Diagnostic) do Kaggle.

A aplicação foi construída com **Streamlit**, e o modelo de **Random Forest** foi treinado e avaliado usando **Scikit-learn**.

> ⚠️ **Disclaimer Importante**
>
> Esta é uma ferramenta **100% acadêmica**. Os resultados **não** representam um diagnóstico médico real e **não devem**, em hipótese alguma, ser usados para decisões clínicas. Sempre consulte um profissional de saúde.

---

## Demonstração da Aplicação

<img width="1271" height="653" alt="image" src="https://github.com/user-attachments/assets/cd04bd10-1cea-4d6e-b16c-821e9454e53b" />



---

## Funcionalidades Principais

* **Modelo Preditivo (Random Forest):** O projeto compara uma `DecisionTreeClassifier` com uma `RandomForestClassifier`. O Random Forest foi escolhido como modelo final por sua acurácia superior (97%+) e, principalmente, por sua maior capacidade de **minimizar Falsos Negativos** (casos malignos classificados como benignos), a métrica mais crítica para este problema.
* **UX Inteligente (Feature Importance):** Em vez de sobrecarregar o usuário com 30 campos de entrada, a aplicação pede apenas as **10 features mais importantes** que o modelo identificou. Os 20 campos restantes são preenchidos automaticamente com os valores medianos do dataset de treino.
* **Interface Amigável:** Construído com Streamlit, o app apresenta sliders interativos e **tooltips de ajuda (?)** em português para explicar termos técnicos (ex: "Concavidade (Pior)", "Área (Pior)"), tornando a ferramenta mais acessível.

---

## Tecnologias Utilizadas

* **Python**
* **Streamlit:** Para a criação da aplicação web interativa.
* **Scikit-learn:** Para o treinamento, avaliação (Matriz de Confusão, `classification_report`) e pré-processamento dos modelos.
* **Pandas:** Para a manipulação e análise exploratória dos dados.
* **Joblib:** Para salvar e carregar os artefatos do modelo (`.pkl`).

---

## Como Executar Localmente

Siga os passos abaixo para rodar o projeto em sua máquina.

**1. Clone o repositório:**
```bash
git clone https://github.com/danielfercope/breast_cancer_wisconsin_ml.git
cd breast_cancer_wisconsin_ml
