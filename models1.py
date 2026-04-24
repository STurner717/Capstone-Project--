# Samantha Turner

## Data Libraries
import pandas as pd
import numpy as np

## Stats Libraries
import statsmodels.api as sm

## ML Library
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, mean_squared_error

## Deep Learning Libraries
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Building NN
class Net(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.fc1 = nn.Linear(dim, 100)
    self.fc2 = nn.Linear(100,32)
    self.fc3 = nn.Linear(32,1)
  def forward(self,x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

def training_models(final_df, pred_cols, mean_cols, count_cols):
    ##### Length of Stay Regression
    # Divide df for training and testing
    trainFraction= .75
    n= len(final_df)
    sample = np.random.uniform(size=n) < trainFraction

    x_nn= final_df[['AgeAdmitted', 'gender'] + mean_cols + count_cols]
    y = final_df['LengthStay']

    x_train = x_nn[sample]
    x_test = x_nn[~sample]

    y_train = y[sample]
    y_test = y[~sample]

    # Calculating baseline value
    baseline_pred= np.mean(y_train)
    baseline_rmse= np.sqrt(mean_squared_error(y_test, [baseline_pred] * len(y_test)))

    # Normalizing Data
    x_mean = x_train.mean()
    x_std= x_train.std().replace(0,1)

    x_train_scale= (x_train - x_mean)/ x_std
    x_test_scale= (x_test - x_mean)/ x_std

    y_mean = y_train.mean()
    y_std= y_train.std()
    if np.isclose(y_std,0):
       y_std = 1

    y_train_scale= (y_train - y_mean)/ y_std
    y_test_scale= (y_test - y_mean)/ y_std

    ## Length of Stay: Linear Regression using OLS
    
    ols_features= ['AgeAdmitted', 'gender'] + mean_cols + count_cols

    train_ols= x_train[ols_features].copy()
    test_ols= x_test[ols_features].copy()

    train_ols= sm.add_constant(train_ols)
    test_ols= sm.add_constant(test_ols)

    ols_mean=train_ols.mean()
    ols_std= train_ols.std().replace(0,1)

    # OLS modeling & Calculations
    model = sm.OLS(y_train_scale, train_ols).fit()
    y_pred= model.predict(test_ols)

    # Undo scale
    y_pred= (y_pred * y_std) +y_mean

    error= y_test - y_pred
    mse = np.mean(error **2)
    rmse = np.sqrt(mse)

    ## Length of Stay: Using NN
    # Convert formats & Automate batch creation
    trainDataset= TensorDataset(torch.Tensor(x_train_scale.values), torch.Tensor(y_train_scale.values).view(-1,1))
    trainloader= torch.utils.data.DataLoader(trainDataset, batch_size=100, shuffle=True)
    net = Net(x_train_scale.shape[1])
    criterion= nn.MSELoss()
    optimizer= optim.Adam(net.parameters(), lr=0.001)

    # Train model
    for epoch in range(50):
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs= net(inputs)
            loss= criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Run test data thru NN
    with torch.no_grad():
        train_pred= net(torch.tensor(x_train_scale.values, dtype=torch.float32)).numpy()
        test_pred= net(torch.tensor(x_test_scale.values, dtype=torch.float32)).numpy()

    # Multi-dim array to 1-D array
    train_pred= train_pred.flatten()
    test_pred= test_pred.flatten()

    # Unscale
    train_pred= train_pred * y_std + y_mean
    test_pred= test_pred * y_std + y_mean

    train_pred= np.clip(train_pred, 0,60)
    test_pred= np.clip(test_pred, 0,60)

    # Calculations
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse= np.sqrt(mean_squared_error(y_test, test_pred))

    #### Mortality - Logistic Regression

    data= final_df[pred_cols + ['hospital_expire_flag']].dropna()
    X_features = data[pred_cols]
    Y_target = data['hospital_expire_flag']

    # Divide train & testing
    trainFraction1 = .75
    sample1 = np.random.rand(len(X_features)) < trainFraction1

    x_train_temp = X_features[sample1].copy()
    x_test_temp = X_features[~sample1].copy()

    y_train1 = Y_target[sample1]
    y_test1 = Y_target[~sample1]

    # Identify columns having more than one unique value
    non_constant_train_cols = x_train_temp.columns[x_train_temp.nunique() > 1].tolist()

    # Filter train & test only include non-constant columns
    x_train1 = x_train_temp[non_constant_train_cols]
    x_test1 = x_test_temp[non_constant_train_cols]

    # Add constant to training and testing sets
    x_train1 = sm.add_constant(x_train1, has_constant='add')
    x_test1 = sm.add_constant(x_test1, has_constant='add')

    # Logit modeling
    marklog= sm.Logit(y_train1, x_train1).fit(method= 'bfgs', maxiter= 300)

    pred1= marklog.predict(x_test1)
    fpr,tpr,thresholds= roc_curve(y_test1,pred1)
    index= np.argmax(tpr-fpr)
    thres= thresholds[index]
    predmor= (pred1 > thres). astype(int)

    results = {
       'model': model,
       'marklog': marklog,
       'y_test': y_test,
       'y_pred': y_pred,
       'train_pred': train_pred,
       'test_pred': test_pred,
       'error': error,
       'baseline_rmse': baseline_rmse,
       'rmse': rmse,
       'test_rmse': test_rmse,
       'y_test1': y_test1,
       'pred1': pred1,
       'thres':thres,
       'predmor': predmor,
       'y_mean': y_mean,
       'y_std': y_std,
       'x_mean': x_mean,
       'x_std': x_std
    }
    sample_df= sample_pred(final_df, model, marklog, thres, mean_cols,count_cols, y_mean, y_std,net,x_mean, x_std)
    results['sample_df']= sample_df
    results['ols_mean']= ols_mean
    results['ols_std']= ols_std
    results['ols_features']= ols_features
    results['nn_model']= net

    return results

def sample_pred(final_df, model, marklog, thres, mean_cols, count_cols, y_mean, y_std, nn_model,x_mean, x_std):
   # sample data for 10 random patients
   sample= final_df.sample(10)
   input_ols= sm.add_constant(sample[['AgeAdmitted', 'gender'] + mean_cols + count_cols], has_constant= 'add')
   pred_ols= model.predict(input_ols)
   ols_values= (pred_ols*y_std) +y_mean
   sample['Length_Stay_OLS']= np.clip(ols_values,0,60)
   x_nn = sample[['AgeAdmitted','gender'] +mean_cols + count_cols]
   x_nn= (x_nn-x_mean)/x_std
   x_nn_tensor= torch.tensor(x_nn.values, dtype=torch.float32)
   with torch.no_grad():
        nn_pred= nn_model(x_nn_tensor).numpy().flatten()
   nn_pred= (nn_pred*y_std) +y_mean
   nn_pred= np.clip(nn_pred,0,60)
   sample['Length_Stay_NN']= nn_pred
   sample['LengthStay_Days']= sample['LengthStay']
   input_log= sample[['AgeAdmitted', 'gender'] + mean_cols +count_cols]
   inputlog= sm.add_constant(input_log, has_constant= 'add')
   sample['Mortality_Probability']= marklog.predict(inputlog)
   sample['Mortality_Prediction']= (sample['Mortality_Probability'] > thres).astype(int)
   sample =sample[[
      'subject_id',
      'AgeAdmitted',
      'gender',
      'diagnosis',
      'LengthStay_Days',
      'Length_Stay_OLS',
      'Length_Stay_NN',
      'hospital_expire_flag',
      'Mortality_Prediction',
      'Mortality_Probability'
   ]]
   
   return sample