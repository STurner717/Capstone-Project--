# Samantha Turner

# Libraries
import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np


# Call from other .py files
from main1 import data_clean
from models1 import training_models
from graphs import final_results

st.title('Length of Stay Hospital Outcome Predictor')

st.write('This project will predict the length of hospital stay and mortality risk using clinical data from the MIMIC-III Clinical Database. This includes patient demographics and lab testing results.')
st.caption('Uses Linear Regression (OLS) Model for Length of Stay & Logistic Regression model for Mortality Risk')

# Load Data
final_df, pred_cols, mean_cols, count_cols = data_clean()
results = training_models(final_df, pred_cols, mean_cols, count_cols)
rmse_results, mortality_results, figures= final_results(final_df, results)

# User Input
AgeAdmitted= st.slider('Select Age:', 0, 100)

gender_choice= {'Male', 'Female'}
gender= st.radio('Select Gender:', gender_choice)
gender_value= 1 if gender == 'Male' else 0

user_data= pd.DataFrame({
    'AgeAdmitted': [AgeAdmitted],
    'gender': [gender_value]
})

st.write('Randomized Lab Values Used in Prediction')
st.caption('Real patient lab testing data is randomly selected to maintain realistic combinations.')
lab_cols = mean_cols + count_cols
random_lab = final_df[lab_cols].sample(1).iloc[0]
for col in lab_cols:
    user_data[col] = random_lab[col]
st.dataframe(pd.DataFrame(random_lab).reset_index().rename(columns={'index': 'Lab Feature', 0: 'Value'}))
ols_feat= results['ols_features']
los_input= user_data[ols_feat]
los_input= sm.add_constant(los_input, has_constant='add')
LoS_pred= results['model'].predict(los_input)[0]
LoS_pred= (LoS_pred* results['y_std']) + results['y_mean']
LoS_pred=np.clip(LoS_pred, 0, 60)
log_feat= pred_cols
log_input= sm.add_constant(user_data[log_feat], has_constant='add')
mort_prob= results['marklog'].predict(log_input)[0]
threshold_mor= results['thres']
mort_pred= 1 if mort_prob > threshold_mor else 0

st.subheader('User Input Predicted Outcomes')
st.write(f'Predicted Length of Hospital Stay:{LoS_pred: .2f} days')
st.write(f'Mortality Probability:{mort_prob: .3f}')
if mort_pred ==1:
    st.write('--> High Risk of Mortality')
else:
    st.write('--> Low Risk of Mortality')

# Sample Data
st.subheader('Example of Sample Patient Data')
st.caption('Gender: male=1, female=0')
st.caption('LengthStay: Actual # of Days. |  Length_Stay_Prediction: Predicted # of Days.')
st.caption('hospital_expire_flag: Actual Death (0= Survive, 1= Died)')
st.caption('Mortality_Probability: Predicted probability of death from logistic regression')
st.caption('Mortality_Prediction: Predicted Death using optimized threshold selected using Youdens Index (0= Survive, 1= Died)')

st.dataframe(results['sample_df'])

# Calculated Results
st.header('Results: ')
st.caption('These metrics show how well the model performs after the full dataset is split into testing and training data.')

st.subheader('Length of Hospital Stay - RMSE Comparision')
st.dataframe(rmse_results)
st.caption('Model used Linear Regression (OLS) for Length of Stay.')
st.caption('A baseline and Neural Network were also computed for comparision.')

st.subheader('Mortality Risk')
st.dataframe(mortality_results)
st.caption('Model used Logistic Regression (BFGS optimization algorithm) for mortality classification.')

st.subheader('Visuals')

st.pyplot(figures[0])
st.pyplot(figures[1])
st.pyplot(figures[2])
st.pyplot(figures[3])
