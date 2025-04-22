import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_data():
    # Load raw data
    data_dir = Path("data")
    df = pd.read_csv(data_dir / "Student_performance_data.csv")
    
    # Basic cleaning
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save raw splits
    train_df.to_csv(data_dir / "train_no_fe.csv", index=False)
    test_df.to_csv(data_dir / "test_no_fe.csv", index=False)
    
    print("Data prepared and split into train/test sets")

if __name__ == "__main__":
    prepare_data()