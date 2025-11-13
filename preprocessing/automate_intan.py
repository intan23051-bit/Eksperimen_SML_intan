import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')


class HeartDiseasePreprocessor:
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def load_data(self, filepath):
        print(f"Loading data from {filepath}...")
        df = pd.read_csv('C:/Users/DELL/Downloads/Eksperimen_SML_intan/heart.csv')
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def handle_missing_values(self, df):
        print("\nHandling missing values...")
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            numerical_features = df.select_dtypes(include=[np.number]).columns
            df[numerical_features] = self.imputer.fit_transform(df[numerical_features])
            print(f"Imputed {missing_count} missing values")
        else:
            print("No missing values found")
        
        return df
    
    def remove_duplicates(self, df):
        print("\nRemoving duplicates...")
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        print(f"Removed {removed} duplicate rows")
        return df
    
    def handle_outliers(self, df, numerical_cols):
        print("\nHandling outliers...")
        outlier_count = 0
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers before capping
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count += len(outliers)
            
            # Cap outliers
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"Capped {outlier_count} outliers using IQR method")
        return df
    
    def feature_engineering(self, df):
        print("\nEngineering features...")
        # umur
        df['age_group'] = pd.cut(df['age'], 
                                  bins=[0, 40, 50, 60, 100], 
                                  labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        # kolestrol
        df['chol_category'] = pd.cut(df['chol'], 
                                      bins=[0, 200, 240, 500], 
                                      labels=['Normal', 'Borderline', 'High'])
        
        print("Created 2 new features: age_group, chol_category")
        return df
    
    def encode_categorical(self, df):
        print("\nEncoding categorical variables...")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            print(f"Encoded {len(categorical_cols)} categorical columns")
        else:
            print("No categorical columns to encode")
        
        return df
    
    def split_data(self, df, target_col='target'):
        print("\nSplitting data...")
        
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        print("\nScaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, 
                                       columns=X_train.columns, 
                                       index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, 
                                      columns=X_test.columns, 
                                      index=X_test.index)
        
        print("Features scaled using StandardScaler")
        
        return X_train_scaled, X_test_scaled
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, output_dir='dataset_preprocessing'):
        print(f"\nSaving preprocessed data to {output_dir}/...")
        
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save files
        X_train.to_csv(f'{output_dir}/X_train_preprocessed aoutomate.csv', index=False)
        X_test.to_csv(f'{output_dir}/X_test_preprocessed aoutomate.csv', index=False)
        y_train.to_csv(f'{output_dir}/y_train_preprocessed aoutomate.csv', index=False)
        y_test.to_csv(f'{output_dir}/y_test_preprocessed aoutomate.csv', index=False)
        
        print("Data saved successfully!")
        print(f"{output_dir}/X_train_preprocessed aoutomate.csv")
        print(f"{output_dir}/X_test_preprocessed aoutomate.csv")
        print(f"{output_dir}/y_train_preprocessed aoutomate.csv")
        print(f"{output_dir}/y_test_preprocessed aoutomate.csv")
    
    def preprocess(self, filepath, output_dir='dataset_preprocessing'):
        print("="*80)
        print("AUTOMATED PREPROCESSING PIPELINE")
        print("="*80)
        
        # Step 1: Load data
        df = self.load_data(filepath)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Get numerical columns (before feature engineering)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numerical_cols:
            numerical_cols.remove('target')
        
        # Step 4: Handle outliers
        df = self.handle_outliers(df, numerical_cols)
        
        # Step 5: Feature engineering
        df = self.feature_engineering(df)
        
        # Step 6: Encode categorical variables
        df = self.encode_categorical(df)
        
        # Step 7: Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Step 8: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Step 9: Save preprocessed data
        self.save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, output_dir)
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nFinal dataset shape:")
        print(f"  Training: {X_train_scaled.shape}")
        print(f"  Testing: {X_test_scaled.shape}")
        print(f"\nTarget distribution:")
        print(f"  Train: {y_train.value_counts().to_dict()}")
        print(f"  Test: {y_test.value_counts().to_dict()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    """Main function to run preprocessing"""
    
    RAW_DATA_PATH = 'C:/Users/DELL/Downloads/Eksperimen_SML_intan/heart.csv'  
    OUTPUT_DIR = 'dataset_preprocessing'
    
    # Initialize preprocessor
    preprocessor = HeartDiseasePreprocessor(test_size=0.2, random_state=42)
    
    # Run preprocessing pipeline
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess(
            filepath=RAW_DATA_PATH,
            output_dir=OUTPUT_DIR
        )
        
        print("\nPreprocessing pipeline completed successfully!")
        
    except FileNotFoundError:
        print(f"\nError: File '{RAW_DATA_PATH}' not found!")
        print("Please make sure the raw dataset is in the correct location.")
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()