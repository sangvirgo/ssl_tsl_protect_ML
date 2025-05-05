import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import sys
import traceback

def create_directories():
    try:
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise


def preprocess(file_path):
    try:
        #Check exist
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cant find {file_path}")
        
        df = pd.read_csv(file_path)

         # Loại bỏ cột 'Timestamp'
        df = df.drop('Timestamp', axis=1)

        df.columns = df.columns.str.strip()
        if 'Label' not in df.columns:
            label_cand = [col for col in df.columns if 'label' in col.lower()]
            if label_cand:
                df.rename(columns={label_cand[0]: 'Label'}, inplace=True)
            else:
                raise ValueError("Cant find")
        
        if df['Label'].dtype == object:
            df['Label'] = df['Label'].apply(
                lambda x: 0 if str(x).lower() in ['benign'] else 1
                )
        
        X = df.drop('Label', axis=1)
        Y = df['Label']
        
        
        #infi variables
        inf_count = np.isinf(X.values).sum()
        if inf_count > 0:
            X = X.replace([np.inf, -np.inf], np.nan)
        
        #nan variables
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            X = X.fillna(X.mean(skipna=True))
        
        
        feature_names = X.columns.to_list()
        feature_path = os.path.join('data', 'feature_names.npy')
        np.save(feature_path, np.array(feature_names))
        
        X_array = X.values.astype(np.float64)
        
        
        #Big num 
        max_val = np.max(np.abs(X_array))
        if max_val > 1e10:
            X_array = np.clip(X_array, -1e10, 1e10)
        
        
        try:
            scaler = StandardScaler()
            X_scale = scaler.fit_transform(X_array)
        except Exception as e:
            print(f"Error: {e}")
        
        
        scaler_path = os.path.join('models', 'scaler.joblib')
        joblib.dump(scaler, scaler_path)

        
        
        x_train, x_test, y_train, y_test = train_test_split(
            X_scale, Y, test_size=0.2, random_state=12, stratify=Y
        )       #stratify: balance between attack and normal
        
        np.save(os.path.join('data', 'x_train.npy'), x_train)
        np.save(os.path.join('data', 'x_test.npy'), x_test)
        np.save(os.path.join('data', 'y_train.npy'), y_train)
        np.save(os.path.join('data', 'y_test.npy'), y_test)
        
        return x_train, x_test, y_train, y_test, feature_names
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        create_directories()
        file_path = "./input.csv"
        if not os.path.exists(file_path):
            file_path = "input.csv"
        x_train, x_test, y_train, y_test, feature_names = preprocess(file_path)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1) 