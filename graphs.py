# Samantha Turner

## Data Libraries
import pandas as pd

## Ploting/ Visual Libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

## ML Library
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, mean_squared_error

def final_results(final_df, results):
    # Results in dataframe format
    rmse_results= pd.DataFrame({
        'RMSE Type Comparison Results:':[
            'Baseline',
            'OLS',
            'NN'
        ],
        'RMSE Values':[
            results['baseline_rmse'],
            results['rmse'],
            results['test_rmse']
        ]
    })

    mortality_results= pd.DataFrame({
        'Mortality Model Results:':[
            'Accuracy',
            'AUC',
            'Confusion Matrix: [TN FP]/[FN TP]'
        ],
        'Calculation Values':[
            accuracy_score(results['y_test1'], results['predmor']),
            roc_auc_score(results['y_test1'], results['pred1']),
            confusion_matrix(results['y_test1'], results['predmor'])
        ]
    })

    # OUTPUTS:
    figures= []

    fig1,ax= plt.subplots()
    ax.scatter(results['y_test'], results['y_pred'], label= 'OLS')
    ax.scatter(results['y_test'], results['test_pred'], label='NN')
    ax.plot([results['y_test'].min(), results['y_test'].max()], [results['y_test'].min(), results['y_test'].max()])
    ax.legend()
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Actual vs Predicted Values Comparision')
    figures.append(fig1)

    fig2, ax= plt.subplots()
    sns.histplot(final_df['LengthStay'], ax=ax)
    ax.set_xlabel('Days in Hospital')
    ax.set_ylabel('Count of Patients')
    ax.set_title('Length of Hospital Stay')
    ax.set_xlim(0,50)
    figures.append(fig2)

    fig3, ax= plt.subplots()
    sns.histplot(results['error'], kde=True)
    ax.set_xlabel('Prediction Errors')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distribution')
    figures.append(fig3)

    fig4, ax= plt.subplots()
    fpr, tpr, _ = roc_curve(results['y_test1'], results['pred1'])
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1], '--')
    ax.set_xlabel('fpr')
    ax.set_ylabel('tpr')
    ax.set_title('ROC Curve - Mortality Prediction')
    figures.append(fig4)

    return rmse_results, mortality_results, figures
