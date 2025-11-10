import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
import joblib

warnings.filterwarnings('ignore', category=FutureWarning)

def carregar_e_processar_dados(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    if 'Unnamed: 32' in df.columns:
        df = df.drop('Unnamed: 32', axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

def treinar_e_avaliar_modelos(df):
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("---Treinando Árvore de Decisão---")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    print("---Treinando Random Forest---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("\n================== RESULTADOS ==================")
    print("\n---Árvore de Decisão (Decision Tree)---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred_dt) * 100:.2f}%")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred_dt))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred_dt, target_names=['Benigno (0)', 'Maligno (1)']))

    print("\n---Random Forest---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred_rf, target_names=['Benigno (0)', 'Maligno (1)']))
    print("================================================")

    return dt_model, rf_model, X.columns

def analisar_resultados(y_test, y_pred_dt, y_pred_rf):
    print("\n---Análise de Falsos Negativos (erro mais crítico)---")
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    fn_dt = cm_dt[1][0]
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    fn_rf = cm_rf[1][0]
    print(f"Árvore de Decisão: {fn_dt} casos Malignos classificados como Benignos (FN).")
    print(f"Random Forest:     {fn_rf} casos Malignos classificados como Benignos (FN).")

    if fn_rf < fn_dt:
        print("\nO Random Forest foi melhor em reduzir o erro mais perigoso!")
    elif fn_dt < fn_rf:
        print("\nA Árvore de Decisão foi melhor em reduzir o erro mais perigoso.")
    else:
        print("\nAmbos os modelos tiveram o mesmo desempenho em Falsos Negativos.")

if __name__ == "__main__":
    arquivo_csv = 'data.csv'

    try:
        df_processado = carregar_e_processar_dados(arquivo_csv)
        print("Dados carregados e processados com sucesso.")
        dt_model, rf_model, colunas = treinar_e_avaliar_modelos(df_processado)
        y_test = train_test_split(df_processado.drop('diagnosis', axis=1),
                                  df_processado['diagnosis'],
                                  test_size=0.2,
                                  random_state=42,
                                  stratify=df_processado['diagnosis'])[3]

        analisar_resultados(y_test, dt_model.predict(train_test_split(df_processado.drop('diagnosis', axis=1),
                                                                      df_processado['diagnosis'],
                                                                      test_size=0.2,
                                                                      random_state=42,
                                                                      stratify=df_processado['diagnosis'])[1]),
                            rf_model.predict(train_test_split(df_processado.drop('diagnosis', axis=1),
                                                              df_processado['diagnosis'],
                                                              test_size=0.2,
                                                              random_state=42,
                                                              stratify=df_processado['diagnosis'])[1]))
    except FileNotFoundError:
        print(f"Erro: Arquivo '{arquivo_csv}' não encontrado.")

    print("\n--- Análise de Importância das Features (Random Forest) ---")

    importancias = rf_model.feature_importances_
    nomes_colunas = colunas
    df_importancias = pd.DataFrame(
        {'Feature': nomes_colunas, 'Importancia': importancias}
    )
    df_importancias = df_importancias.sort_values(by='Importancia', ascending=False)

    print("As 10 features mais importantes para o modelo:")
    print(df_importancias.head(10))
    medianas = df_processado[nomes_colunas].median()
    print("\nValores medianos:")
    print(medianas)