## Final Capstone Project: Hospital Outcome Predictor
 **Samantha Turner**
### Synopsis
This project creates a Hospital Outcome Predictor using the MIMIC-III Clinical Database Demo, which contains clinical data for 100 ICU patients. The project predicts:
<br>**Length of Stay:** number of days the patient is in the hospital
<br>**Mortality Risk:** probability of patient death during their hospital stay
### Dataset Used
**Patients.csv** : subject_id, gender, dob
- DOB converted to datetime, filtered unrealistic values, used to calculate age at admission
- Gender encoded male= 1, female= 0

**Admissions.csv** : subject_id, admittime, dischtime, hospital_expire_flag
- Admittime and dischtime converted to datetime, filtered unrealistic values, used to calculate length of stay & age
- Hospital_expire_flag(mortality) encode: died= 1, survived=0

**LabEvents.csv** : subject_id, itemid, value
- Valuenum(lab test value): removed non-numerical values
- Itemid(lab test type): top ten tests performed selected, aggregated per patient for mean and count of each test. Missing value filled (mean -> median, count -> 0)

**Data Preparation:**
- Dataset merged by subject_id
- Final dataset: age, gender, mean, count, length of stay, mortality
- Outliers clipped (1st-99th percentile) & missing values filled by median
- Data split into training/testing subset & normalized for modeling
### Models
**Linear Regression (OLS):** baseline model for length of stay
<br> **Neural Network (100 -> 32 -> 1)**: comparsion model for length of stay
<br>**Logistic Regression (BFGS):** mortality prediction, optimized threshold from  Youden's Index 
### Evaluation Metrics
**Length of Stay:** Root Mean Squared Error (RMSE) calulated for baseline mean model, OLS regression, & Neural network
<br>**Mortality:** Accuracy, Area Under Curve(AUC), and Confusion Matrix
### Running the Code
python3 --version
<br>python3 -m venv venv
<br>source venv/bin/activate
<br>pip install pandas numpy matplotlib seaborn statsmodels scikit-learn torch streamlit
<br>run files: main.py -> models.py -> graphs.py -> streamapp.py
<br>streamlit run streamapp.py
### Streamlit App
The app allows users to input age and gender to generate predictions for length of stay and mortality risk. It provides sample patient predictions, evalution metric, and visuals of model perform on training and testing data.
### Video Presentation

### Reference
https://physionet.org/content/mimiciii-demo/1.4/
