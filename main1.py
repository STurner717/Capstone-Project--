# Samantha Turner

## Data Libraries
import pandas as pd
import numpy as np

def data_clean():
  ## Read in datasets
  patients_df=pd.read_csv("https://raw.githubusercontent.com/STurner717/Capstone-Project--/refs/heads/main/PATIENTS.csv")
  admission_df=pd.read_csv("https://raw.githubusercontent.com/STurner717/Capstone-Project--/refs/heads/main/ADMISSIONS.csv")
  labevents_df=pd.read_csv("https://raw.githubusercontent.com/STurner717/Capstone-Project--/refs/heads/main/LABEVENTS.csv")

  ## Choosing to drop specific columns that are unnessecary
  patients_df = patients_df.drop(['row_id', 'dod', 'dod_hosp','dod_ssn', 'expire_flag'], axis = 1)
  admission_df = admission_df.drop(['row_id', 'admission_location','discharge_location', 'insurance', 'religion', 'language', 'marital_status', 'ethnicity'], axis = 1)
  labevents_df = labevents_df.drop(['row_id', 'hadm_id','charttime', 'valueuom', 'flag'], axis = 1)

  ### Cleaning and Calculating Values
  ## Patients Dataframe
  # Age when admitted
  patients_df['dob']= pd.to_datetime(patients_df['dob'], errors='coerce')
  patients_df = patients_df.dropna(subset=['dob'])

  # Filter out unrealistic dob values to prevent OverflowError
  patients_df = patients_df[(patients_df['dob'].dt.year < 2200)]
  patients_df = patients_df[(patients_df['dob'].dt.year > 1900)]

  # Assigning Male gender to 1 & Female gender to 0
  patients_df['gender']= (patients_df['gender']=='M').astype(int)

  ## Admission Dataframe
  # Length of Stay
  admission_df['admittime']= pd.to_datetime(admission_df['admittime'], errors='coerce')
  admission_df = admission_df.dropna(subset=['admittime'])
  admission_df['dischtime']= pd.to_datetime(admission_df['dischtime'], errors='coerce')
  admission_df = admission_df.dropna(subset=['dischtime'])

  # Filter out unrealistic admittime & dischtime values to prevent OverflowError
  admission_df = admission_df[(admission_df['admittime'].dt.year < 2200) & (admission_df['dischtime'].dt.year < 2200)]

  # Calculate Length of Hosiptal Stay
  admission_df['LengthStay'] = (admission_df['dischtime']- admission_df['admittime']).dt.total_seconds() / (60 * 60 * 24)

  ## Merging Patients df & Admission df
  pat_ad_df= pd.merge(patients_df,admission_df, how='left', on= 'subject_id')

  # Calculate Age Admitted to hospital
  pat_ad_df['AgeAdmitted']= (pat_ad_df['admittime']- pat_ad_df['dob']).dt.total_seconds() / (60 * 60 * 24 * 365.25)

  # Remove any outliers in AgeAdmitted
  pat_ad_df = pat_ad_df[(pat_ad_df['AgeAdmitted'] > 0) & (pat_ad_df['AgeAdmitted'] < 120)]

  ## Lab Events Dataframe
  # Check all values are numerical
  labevents_df['valuenum'] = pd.to_numeric(labevents_df['valuenum'], errors='coerce')
  labevents_df = labevents_df.dropna(subset=['valuenum'])

  # Select the top 10 testing types done in df
  topLabs = labevents_df['itemid'].value_counts().head(10).index
  labevents_df = labevents_df[labevents_df['itemid'].isin(topLabs)]

  # Groups subject_id & itemid & calculations for mean and count of the valuenum column
  labevents_df= labevents_df.groupby(['subject_id', 'itemid']).agg({'valuenum':['mean', 'count']}).unstack()

  # Renames columns simplfying labevent itemid name
  labevents_df.columns= [
      f"{stat}_{itemid}"
      for(_, stat, itemid) in labevents_df.columns
  ]

  # Fills in missing values in mean & count
  for col in labevents_df.columns:
    if col.startswith('mean'):
      labevents_df[col]= labevents_df[col].fillna(labevents_df[col].median())
    else:
      labevents_df[col]= labevents_df[col].fillna(0)

  labevents_df= labevents_df.reset_index()

  ## Merge pat_ad_df & labevents_df
  pat_ad_df= pd.merge(pat_ad_df,labevents_df, how='left', on= 'subject_id')

  # Select wanted columns
  initial_col = [
    'subject_id',
    'AgeAdmitted',
    'diagnosis',
    'LengthStay',
    'gender',
    'hospital_expire_flag'
  ]

  # Identify lab columns from pat_ad_df
  prefix = ['mean', 'count']
  lab_col = [col for col in pat_ad_df.columns if any(col.startswith(p) for p in prefix)]

  # Combine all column names
  all_cols = initial_col + lab_col

  # Create final_df selecting prev columns from pat_ad_df
  final_df = pat_ad_df[all_cols].copy()

  # Separate Mean columns & count columns
  mean_cols = [col for col in final_df.columns if col.startswith('mean')]
  count_cols = [col for col in final_df.columns if col.startswith('count')]

  # Fills na values with their median in select columns
  pred_cols= ['AgeAdmitted', 'gender'] + mean_cols + count_cols
  final_df.loc[:,pred_cols]= final_df[pred_cols]. fillna(final_df[pred_cols].median())

  # Convert to float to prevent clipping error
  for col in final_df.columns:
    if col not in ['subject_id', 'hospital_expire_flag', 'diagnosis']:
      final_df[col] = final_df[col].astype(float)

  # Clipping outliers using
  for col in final_df.columns:
    if col not in ['subject_id', 'hospital_expire_flag', 'diagnosis', 'LengthStay']:
      final_df.loc[:,col] = final_df[col].clip(final_df[col].quantile(0.01), final_df[col].quantile(0.99))

  return final_df, pred_cols, mean_cols, count_cols