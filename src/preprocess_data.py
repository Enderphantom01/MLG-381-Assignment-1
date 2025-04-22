import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    data_dir = Path("data")
    
    # Load raw splits
    train_df = pd.read_csv(data_dir / "train_no_fe.csv")
    test_df = pd.read_csv(data_dir / "test_no_fe.csv")
    
    # Feature engineering
    def add_features(df):
        # Example feature engineering
        df['StudyAttendanceRatio'] = df['StudyTimeWeekly'] / (df['AttendanceRate'] + 1)
        df['SupportInteraction'] = df['Tutoring'] * df['ParentalSupport']
        return df
    
    train_fe = add_features(train_df)
    test_fe = add_features(test_df)
    
    # Scale numerical features
    numeric_cols = ['StudyTimeWeekly', 'AttendanceRate', 'Absences', 'StudyAttendanceRatio']
    scaler = StandardScaler()
    train_fe[numeric_cols] = scaler.fit_transform(train_fe[numeric_cols])
    test_fe[numeric_cols] = scaler.transform(test_fe[numeric_cols])
    
    # Save processed data
    train_fe.to_csv(data_dir / "train_fe.csv", index=False)
    test_fe.to_csv(data_dir / "test_fe.csv", index=False)
    
    print("Data preprocessing and feature engineering complete")

if __name__ == "__main__":
    preprocess_data()