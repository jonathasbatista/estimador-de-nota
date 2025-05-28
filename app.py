import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Carregar dados
df = pd.read_csv('student_habits_performance.csv')

#Limpando dados
df.dropna()

#Traduzindo dados
df.rename(columns={
    'student_id': 'id_aluno',
    'age': 'idade',
    'gender': 'genero',
   'study_hours_per_day': 'horas_de_estudo_por_dia',
    'social_media_hours': 'horas_de_rede_social',
    'netflix_hours':'horas_de_netflix',
    'part_time_job':'trabalho_integral',
    'attendance_percentage': 'porcentagem_de_presenca',
    'sleep_hours': 'horas_de_sono',
    'diet_quality': 'qualidade_da_dieta',
    'exercise_frequency': 'frequencia_de_exercicios',
    'parental_education_level': 'nivel_de_ensino_parental',
    'internet_quality': 'qualidade_da_internet',
    'mental_health_rating':'avaliacao_mental',
    'extracurricular_participation': 'participacao_extracurricular',
    'exam_score': 'nota_do_exame'
}, inplace=True)

traducoes = {
    'genero': {'Male': 'Masculino', 'Female': 'Feminino', 'Other': 'Outro'},
    'trabalho_integral': {'Yes': 'Sim', 'No': 'Não'},
    'qualidade_da_dieta': {'Poor': 'Ruim', 'Fair': 'Regular', 'Good': 'Boa'},
   'nivel_de_ensino_parental': {'Master': 'Mestrado', 'High School': 'Ensino Médio', 'Bachelor': 'Graduação'},
   'qualidade_da_internet': {'Poor': 'Ruim', 'Good': 'Boa', 'Average': 'Média'},
   'participacao_extracurricular': {'Yes': 'Sim', 'No': 'Não'}
}

for coluna, mapa in traducoes.items():
    df[coluna] = df[coluna].map(mapa)

# Definir colunas
colunas_features = [
    'genero',
    'horas_de_estudo_por_dia',
    'horas_de_rede_social',
    'horas_de_netflix',
    'trabalho_integral',
    'porcentagem_de_presenca',
    'horas_de_sono',
    'qualidade_da_dieta',
    'frequencia_de_exercicios',
    'nivel_de_ensino_parental',
    'qualidade_da_internet',
    'avaliacao_mental',
    'participacao_extracurricular'
]

x = df[colunas_features].copy()
y = df['nota_do_exame']

# Label Encoding nas variáveis categóricas
colunas_categoricas = [
    'genero',
    'trabalho_integral',
    'qualidade_da_dieta',
    'nivel_de_ensino_parental',
    'qualidade_da_internet',
    'participacao_extracurricular'
]

# Salvar os LabelEncoders para usar no input do usuário
label_encoders = {}

for coluna in colunas_categoricas:
    le = LabelEncoder()
    x[coluna] = le.fit_transform(x[coluna])
    label_encoders[coluna] = le

# Treinar modelo
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
modelo = RandomForestRegressor(random_state=42)
modelo.fit(x_train, y_train)

# Interface Streamlit
st.set_page_config(page_title="Estimador de Nota do Exame", layout="centered")
st.title("Estimador de Nota do Exame")
st.divider()

# Inputs do usuário
genero = st.selectbox("Gênero", df['genero'].unique())
horas_estudo = st.slider("Horas de Estudo por Dia", 0.0, 10.0, 2.0)
horas_rede_social = st.slider("Horas em Redes Sociais", 0.0, 10.0, 2.0)
horas_netflix = st.slider("Horas de Netflix", 0.0, 10.0, 2.0)
trabalho_integral = st.selectbox("Trabalho Integral", df['trabalho_integral'].unique())
porcentagem_presenca = st.slider("Porcentagem de Presença", 0, 100, 90)
horas_sono = st.slider("Horas de Sono", 0.0, 12.0, 7.0)
qualidade_dieta = st.selectbox("Qualidade da Dieta", df['qualidade_da_dieta'].unique())
frequencia_exercicios = st.slider("Frequência de Exercícios por Semana", 0, 7, 3)
nivel_ensino_parental = st.selectbox("Nível de Ensino Parental", df['nivel_de_ensino_parental'].unique())
qualidade_internet = st.selectbox("Qualidade da Internet", df['qualidade_da_internet'].unique())
avaliacao_mental = st.slider("Avaliação Mental", 0, 10, 6)
participacao_extracurricular = st.selectbox("Participação Extracurricular", df['participacao_extracurricular'].unique())

# Previsão
if st.button("Prever Nota"):
    entrada = pd.DataFrame([[
        genero,
        horas_estudo,
        horas_rede_social,
        horas_netflix,
        trabalho_integral,
        porcentagem_presenca,
        horas_sono,
        qualidade_dieta,
        frequencia_exercicios,
        nivel_ensino_parental,
        qualidade_internet,
        avaliacao_mental,
        participacao_extracurricular
    ]], columns=colunas_features)

    # Aplicar LabelEncoder nas colunas categóricas
    for coluna in colunas_categoricas:
        le = label_encoders[coluna]
        entrada[coluna] = le.transform(entrada[coluna])

    # Fazer a previsão
    nota_prevista = modelo.predict(entrada)[0]
    st.success(f"A nota prevista é: {nota_prevista:.2f}")