import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'

@st.cache_resource
def load_models():
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    selector = joblib.load(MODEL_DIR / 'selector_sfm.pkl')
    model_dt = joblib.load(MODEL_DIR / 'decision_tree.pkl')
    model_nb = joblib.load(MODEL_DIR / 'naive_bayes.pkl')
    model_svm = joblib.load(MODEL_DIR / 'svm.pkl')
    encoders = joblib.load(MODEL_DIR / 'encoders.pkl')
    return scaler, selector, model_dt, model_nb, model_svm, encoders

scaler, selector, model_dt, model_nb, model_svm, encoders = load_models()

st.set_page_config(page_title="Prediksi Income", page_icon="$")
st.title("$ Prediksi Income")
st.markdown("**Kelas C2 - Kelompok 4** | Dataset: Adult Census Income")
st.markdown("---")

st.sidebar.header("Pengaturan")
model_pilihan = st.sidebar.selectbox(
    "Pilih Model",
    ['Decision Tree', 'Naive Bayes', 'SVM']
)

models = {
    'Decision Tree': model_dt,
    'Naive Bayes': model_nb,
    'SVM': model_svm
}
model = models[model_pilihan]

st.sidebar.markdown("---")
st.sidebar.markdown("**Info Model:**")
st.sidebar.markdown(f"- Model: `{model_pilihan}`")
st.sidebar.markdown(f"- Jumlah fitur: `{selector.get_support().sum()}`")

st.header("Input Data")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur", min_value=17, max_value=90, value=30)
    workclass = st.selectbox(
        "Workclass",
        ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov',
         'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked']
    )
    education = st.selectbox(
        "Education",
        ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
         'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
         '5th-6th', '10th', '1st-4th', 'Preschool', '12th']
    )
    education_num = st.number_input("Education Num", min_value=1, max_value=16, value=10)
    marital_status = st.selectbox(
        "Marital Status",
        ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
         'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
    )
    occupation = st.selectbox(
        "Occupation",
        ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
         'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
         'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
         'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
    )
    relationship = st.selectbox(
        "Relationship",
        ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
    )

with col2:
    race = st.selectbox(
        "Race",
        ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    )
    sex = st.selectbox("Sex", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=99999, value=0)
    hours_per_week = st.number_input("Jam Kerja per Minggu", min_value=1, max_value=99, value=40)
    native_country = st.selectbox(
        "Native Country",
        ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
         'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
         'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy',
         'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland',
         'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti',
         'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland',
         'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru',
         'Hong', 'Holand-Netherlands']
    )

st.markdown("---")

if st.button("Prediksi", type="primary"):
    input_df = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'education': education,
        'education.num': education_num,
        'marital.status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital.gain': capital_gain,
        'capital.loss': capital_loss,
        'hours.per.week': hours_per_week,
        'native.country': native_country
    }])

    st.subheader("Data Input")
    st.dataframe(input_df)

    input_encoded = input_df.copy()
    for col, le in encoders.items():
        if col in input_encoded.columns:
            input_encoded[col] = input_encoded[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    input_scaled = scaler.transform(input_encoded)

    input_selected = selector.transform(input_scaled)

    pred = model.predict(input_selected)[0]

    st.subheader("Hasil Prediksi")
    if pred == 1:
        st.success("**Income > $50K**")
    else:
        st.info("**Income <= $50K**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_selected)[0]
        st.markdown(f"- Probabilitas `<=50K`: **{proba[0]:.2%}**")
        st.markdown(f"- Probabilitas `>50K`: **{proba[1]:.2%}**")