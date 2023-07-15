# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 00:28:16 2023

@author: Kaushik
"""
import pickle
import streamlit as st
import pandas as pd
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score



# loading the saved models

diabetes_model = pickle.load(open('C:/Users/Kaushik/Desktop/multiple-disease prediction mini project sem 6/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('C:/Users/Kaushik/Desktop/multiple-disease prediction mini project sem 6/heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('C:/Users/Kaushik/Desktop/multiple-disease prediction mini project sem 6/parkinsons_model.sav', 'rb'))


 

# Sidebar for navigation

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          ['Diabetes Prediction','Breast Cancer Prediction', 'Heart Disease Prediction', 
                           'Stroke Prediction','Parkinsons Prediction'],
                          icons=['activity', 'globe','heart','clock','person'],
                          default_index=0) 


    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    age = int(age) if age.isdigit() else 0    
    with col2:
        sex = st.text_input('Sex')
    sex = int(sex) if sex.isdigit() else 0    
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        input_data = [[age, sex, int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)]]

        heart_prediction = heart_disease_model.predict(input_data)

        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    


    
    
    
    
    
    
    
    
    
    
    
# Breast Cancer Prediction System Page
if (selected == "Breast Cancer Prediction"):
       
    # Title of the application
    st.title('Breast Cancer Prediction')
    
    # Data Collection & Processing
    breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
    data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
    data_frame['label'] = breast_cancer_dataset.target
    
    # Separating the features and target
    X = data_frame.drop(columns='label', axis=1)
    Y = data_frame['label']
    
    # Splitting the data into training data & testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    
    # Model Training
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    # User input for prediction
    st.header('Enter Input for Prediction')
    input_data = []
    for feature_name in breast_cancer_dataset.feature_names:
        value = st.number_input(f'Enter value for {feature_name}', key=feature_name)
        input_data.append(value)
    
    # Submit button
    submit_button = st.button('Submit')
    # code for Prediction
    breast_diagnosis = ''
    # Prediction Result
    if submit_button:
        # Predicting the result
        prediction = model.predict([input_data])
    
    
        # Displaying the result
        st.header('Prediction Result')
        if prediction[0] == 0:
            breast_diagnosis = "The Breast cancer is Malignant"
        else:
            breast_diagnosis = "The Breast Cancer is Benign"
          
    st.success(breast_diagnosis)








if (selected == "Stroke Prediction"):

    # Create the Streamlit app
    st.title('Stroke Prediction System')
    
    
    # Load the dataset
    data = pd.read_csv('C:/Users/Kaushik/Desktop/multiple-disease prediction mini project sem 6/healthcare-dataset-stroke-data.csv')
    
    
    # code for Prediction
    stroke_diagnosis = ''
    
    
    # Select features and target variable
    features = data[['age', 'gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
                     'avg_glucose_level', 'bmi', 'smoking_status']]
    target = data['stroke']
    
    # Convert categorical variables to numerical using one-hot encoding
    features_encoded = pd.get_dummies(features, drop_first=True)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)
    
    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
   
    
    # Collect user input
    age = st.slider('Age', 0, 100)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    hypertension = st.checkbox('Hypertension')
    heart_disease = st.checkbox('Heart Disease')
    ever_married = st.selectbox('Marital Status', ['Yes', 'No'])
    work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
    glucose_level = st.number_input('Average Glucose Level')
    bmi = st.number_input('BMI')
    smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
    
    # Create a button for prediction
    predict_button = st.button('Predict')
    
    # Make predictions based on user input when the button is clicked
    if predict_button:
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [residence_type],
            'avg_glucose_level': [glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status]
        })
    
        # Convert categorical variables to numerical using one-hot encoding
        input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    
        # Align input data with training data columns
        input_data_aligned = input_data_encoded.reindex(columns=X_train.columns, fill_value=0)
    
        # Predict the result
        prediction = model.predict(input_data_aligned)
        
        
    
        # Display the prediction result
        st.header('Prediction Result')
        if prediction[0] == 1:
            stroke_diagnosis = "The individual is predicted to have a stroke."
        else:
            stroke_diagnosis = "The individual is predicted to not have a stroke."
      
    st.success(stroke_diagnosis)
    
    
    
    
    
    
    
    
    
# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)
        
