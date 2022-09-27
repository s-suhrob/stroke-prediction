import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import joblib
import streamlit as st
import streamlit.components.v1 as components



rf_pipeline = joblib.load("models/rf_pipeline.joblib") #loading model 



def stroke_run():
    
    
    st.set_page_config(layout='wide')

    st.title('Пресказание инсульта с помощью машинного обучения')

    # age = st.text_input('Ваш возраст', value='18')
    age = st.number_input(label='Ваш возраст', min_value=18, max_value=80)
    gender = st.selectbox(label='Ваш пол', options=['Мужчина', 'Женщина'])
    if gender == 'Мужчина':
        gender = 'Male'
    elif gender == 'Женщина':
        gender = 'Female'
    work_type = st.selectbox(label='Тип работы', options=['Частный', 'Самозанятые', 
                            'Дети', 'Государственная работа', 'Никогда не работал'
                            ])

    if work_type == 'Частный':
        work_type = 'Private'
    elif work_type == 'Самозанятые':
        work_type = 'Self-employed'
    elif work_type =='Дети':
        work_type = 'children'
    elif work_type == 'Государственная работа':
        work_type = 'Govt_job'
    elif work_type == 'Никогда не работал':
        work_type = 'Never_worked'

    residence_type = st.selectbox(label='Тип жительства', options=['Город', 'Сельская'])

    if residence_type =='Город':
        residence_type = 'Urban'
    elif residence_type == 'Сельская':
        residence_type = 'Rural'

    ever_married = st.selectbox(label='Когда-либо был(a) женат(замужем)', options=['Да', 'Нет'])
    if ever_married =='Да':
        ever_married = 'Yes'
    elif ever_married == 'Нет':
        ever_married = 'No'

    hypertension = st.selectbox(label='Был(а) ли у вас гипертония', options=['Да', 'Нет'])

    if hypertension == 'Да':
        hypertension = '1'
    elif hypertension == 'Нет':
        hypertension = '0'

    heart_disease = st.selectbox(label='Есть ли у вас сердечная болезнь', options=['Да', 'Нет'])
    
    if heart_disease == 'Да':
        heart_disease = '1'
    elif heart_disease == 'Нет':
        heart_disease = '0'


    avg_glucose_level = st.number_input(label='Cредний уровень глюкозы', min_value=50 , max_value=300)

    bmi = st.number_input(label='Индекс массы тела',min_value=10, max_value=100)

    smoke_status = st.selectbox(label='Статус курения', options=['Никогда не курил(а)', 'Ранее курил(а)', 'Курю'])

    if smoke_status == 'Никогда не курил(а)':
        smoke_status = 'never smoked'
    elif smoke_status =='Ранее курил(а)':
        smoke_status = 'formerly smoked'
    elif smoke_status == 'Курю':
        smoke_status = 'smokes'


    pred_data = pd.DataFrame(data={
        'gender':gender,
        'age':int(age),
        'hypertension':int(hypertension),
        'heart_disease':int(heart_disease),
        'ever_married':ever_married,
        'work_type':work_type,
        'Residence_type':residence_type,
        'avg_glucose_level':float(avg_glucose_level),
        'bmi':int(bmi),
        'smoking_status':smoke_status
    }, index=[0])

    if st.button('Ок'):
        with st.spinner('Загрузка подождите...'):
            time.sleep(2)
            with st.container():
                # st.dataframe(pred_data)
                st.header('Оценка модели')
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Точность предсказание (модели)', value=0.94)
                with col2:
                    st.metric('AUC',value=0.78)
                    
                st.header('Презсказание модели на ваших данных')
                model_output = rf_pipeline.predict_proba(pred_data)
                st.write(model_output)