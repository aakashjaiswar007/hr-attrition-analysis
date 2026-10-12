import pandas as pd

def load_and_clean():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    # Drop useless columns
    df.drop(['EmployeeCount', 'Over18', 'StandardHours'], 
             axis=1, inplace=True)
    
    # Convert target to binary
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Convert categorical columns
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    return df