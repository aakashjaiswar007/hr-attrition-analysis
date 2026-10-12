import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from data_cleaning import load_and_clean

def train_model():
    df = load_and_clean()
    
    # Encode remaining categoricals
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    # Split features and target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2%}")
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("model.pkl saved successfully!")
    return model, accuracy

if __name__ == "__main__":
    train_model()