import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def apply_ml_methods():
    print("Loading the enriched dataset...")
    df = pd.read_csv('data/nba_games_enriched.csv')
    
    df['Win'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    
    features = ['Distance_Traveled_km', 'Days_of_Rest', 'Altitude_Difference']
    
    df_ml = df[features + ['Win']].dropna()
    
    X = df_ml[features]
    y = df_ml['Win']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]} games")
    print(f"Test set size: {X_test.shape[0]} games")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- MODEL 1: LOGISTIC REGRESSION ---")
    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_train_scaled, y_train)
    log_preds = log_model.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, log_preds):.4f}")
    print("Detailed Classification Report:")
    print(classification_report(y_test, log_preds))
    
    print("\n--- MODEL 2: RANDOM FOREST CLASSIFIER ---")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train) 
    rf_preds = rf_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print("Detailed Classification Report:")
    print(classification_report(y_test, rf_preds))
    
    print("\nRandom Forest - Feature Importance:")
    for feature, imp in zip(features, rf_model.feature_importances_):
        print(f"- {feature}: {imp:.4f}")

if __name__ == "__main__":
    apply_ml_methods()