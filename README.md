## Final Capstone Project: Hospital Outcome Predictor
 Samantha Turner
### Synopsis
This project creates a Hospital Outcome Predictor using the MIMIC-III Clinical Database Demo which contains health related data for 100 ICU patients. The project predicts:
- Length of Stay: The number of days the patient is in the hospital
- Mortality Risk: The probability that the patient will died during their hospital stay
### Dataset
#### 1.) Patients.csv
- Columns used: subject_id, gender, dob
- date of birth- converted to proper date time format, filtered to remove unrealistic values, and used to calculate the patients age that they were admitted to the hosiptal
- Gender value- encoded: male= 1, female= 0.
#### 2.) Admissions.csv
- Columns used: subject_id, admittime, dischtime, hospital_expire_flag
- Admittime and dischtime, hosiptal admission and discharge time, converted to proper date time format, filtered to remove unrealistic values, and used to calculate length of hospital stay and age admitted to the hosiptal
- Hospital_expire_flag, mortality count,- encoded: died= 1, survived=0
#### 3.) LabEvents.csv
- Columns used: subject_id, itemid, value
- Valuenum, value for specific lab test, ensured to only had a numerical value
- Itemid, lab test type, was used to select the top ten tests performed and then aggregated by patient for their mean and count of each test. Missing value for mean filled using median and count filled in by zero
#### Data Preparation
Dataset were merged using subject_id. A final dataset was then created with age, gender, mean, count, length of stay, and mortality. Outliers were clipped (1st-99th percentile) and missing values were handled. The final dataset was split into a training and testing subset for modeling and was normalized.
### Models
1. Linear Regression (OLS): used to predict Length of Stay, acted as baseline model
2. Neural Network (PyTorch 100 -> 32 -> 1): comparsion model to predict Length of Stay
3. Logistic Regression (BFGS): used to predict Mortality Risk, optimized threshold selected using Youden's Index 
### Evaluation Metrics
#### Length of Stay
Root Mean Squared Error (RMSE) was calulated for baseline mean model, OLS model, and Neural Network to compare values.
#### Mortality:
Accuracy, Area Under Curve(AUC), and Confusion Matrix was calulated to judge results.
### Running the Code
- python3 --version
- python3 -m venv venv
- source venv/bin/activate
- pip install pandas numpy matplotlib seaborn statsmodels scikit-learn torch streamlit
- run main.py then models.py then graphs.py then streamapp.py
- streamlit run streamapp.py
### Streamlit App
The app provides user to view model predictions when inputing age and gender, a sample dataset, evalutions of metric, and views of multiple visualize of the split training and testing final dataset.
### Video Presentation

### Reference
https://physionet.org/content/mimiciii-demo/1.4/
