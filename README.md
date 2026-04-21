## Final Capstone Project: Hospital Outcome Predictor
 Samantha Turner
### Synopsis
This project creates a Hospital Outcome Predictor using the MIMIC-III Clinical Database Demo which contains health related data for 100 ICU patients. The project predicts:
- Length of Stay: The number of days the patient is in the hospital
- Mortality Risk: The probability that the patient will died during their hospital stay
### Dataset Used
#### 1.) Patients.csv
- Columns used: subject_id, gender, dob
- dob(date of birth): converted to datetime format, filtered to remove unrealistic values, and used to calculate the age admitted to the hosiptal
- Gender encoded: male= 1, female= 0.
#### 2.) Admissions.csv
- Columns used: subject_id, admittime, dischtime, hospital_expire_flag
- Admittime and dischtime(admission & discharge time): converted to datetime format, filtered to remove unrealistic values, and used to calculate length of hospital stay & age admitted to the hosiptal
- Hospital_expire_flag(mortality) encoded: died= 1, survived=0
#### 3.) LabEvents.csv
- Columns used: subject_id, itemid, value
- Valuenum(lab test value): removed non-numerical values
- Itemid(lab test type): top ten tests performed selected, aggregated per patient for mean and count of each test. Missing value filled (median for mean, 0 for count)
#### Data Preparation
- Dataset were merged by subject_id
- Final dataset: age, gender, mean, count, length of stay, mortality
- Outliers clipped (1st-99th percentile) & missing values filled by median
- Split into a training and testing subset & normalized for modeling
### Models
1. Linear Regression (OLS): baseline model to predict Length of Stay
2. Neural Network (PyTorch 100 -> 32 -> 1): comparsion model to predict Length of Stay
3. Logistic Regression (BFGS): used to predict Mortality Risk, optimized threshold selected using Youden's Index 
### Evaluation Metrics
1. Length of Stay: Root Mean Squared Error (RMSE) calulated for baseline mean model, OLS regression, & Neural network
2. Mortality: Accuracy, Area Under Curve(AUC), and Confusion Matrix
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
