import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def train_models():
    data_dir = Path("data")
    artifacts_dir = Path("artifacts")
    
    # Load processed data
    train_fe = pd.read_csv(data_dir / "train_fe.csv")
    test_fe = pd.read_csv(data_dir / "test_fe.csv")
    
    # Prepare data
    X_train = train_fe.drop('GradeClass', axis=1)
    y_train = train_fe['GradeClass']
    X_test = test_fe.drop('GradeClass', axis=1)
    y_test = test_fe['GradeClass']
    
    # Model 1: Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    
    # Model 2: Random Forest (with feature engineered data)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate models
    def evaluate_model(model, X, y, model_name):
        y_pred = model.predict(X)
        print(f"\n{model_name} Performance:")
        print(classification_report(y, y_pred))
        return y_pred
    
    evaluate_model(dt, X_test, y_test, "Decision Tree")
    evaluate_model(rf, X_test, y_test, "Random Forest")
    
    # Save models
    artifacts_dir.mkdir(exist_ok=True)
    with open(artifacts_dir / "model_1.pkl", 'wb') as f:
        pickle.dump(dt, f)
    with open(artifacts_dir / "model_2.pkl", 'wb') as f:
        pickle.dump(rf, f)
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv(artifacts_dir / "feature_importance.csv", index=False)
    
    print("\nModels trained and saved to artifacts directory")

if __name__ == "__main__":
    train_models()