import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os

def train_xgb():
    # Load dữ liệu
    x_train = np.load('data/x_train.npy')
    x_test = np.load('data/x_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    feature_names = np.load('data/feature_names.npy')
    
    # Kết hợp train và test để cross-validation
    X = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    
    # Định nghĩa các giá trị max_depth và learning_rate
    max_depth_values = [5, 10, 15, 20]
    learning_rates = [0.01, 0.05, 0.1]
    
    cv_scores = []
    test_scores = []
    best_score = 0
    best_params = {}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
    
    for lr in learning_rates:
        for depth in max_depth_values:
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=depth,
                learning_rate=lr,
                random_state=12,
            )
            
            # Cross-validation
            cv_results = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            cv_mean = np.mean(cv_results)
            cv_scores.append((lr, depth, cv_mean))
            
            # Huấn luyện trên tập train
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            test_acc = accuracy_score(y_test, y_pred)
            test_scores.append((lr, depth, test_acc))
            
            print(f"learning_rate={lr}, max_depth={depth}:")
            print(f"  CV Accuracy: {cv_mean:.4f} (±{np.std(cv_results):.4f})")
            print(f"  Test Accuracy: {test_acc:.4f}")
            print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
            print(f"  Recall: {recall_score(y_test, y_pred):.4f}")
            
            # Lưu tham số tốt nhất
            if cv_mean > best_score:
                best_score = cv_mean
                best_params = {'learning_rate': lr, 'max_depth': depth}
            
            # Vẽ confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix (lr={lr}, depth={depth})")
            plt.ylabel('Reality')
            plt.xlabel('Predict')
            plt.gca().invert_yaxis()
            plt.savefig(f'xgb_cfs_mtrx_lr{lr}_depth{depth}.png')
            plt.close()
    
    # Chọn tham số tốt nhất
    print(f"\nBest parameters: learning_rate={best_params['learning_rate']}, max_depth={best_params['max_depth']}")
    
    # Huấn luyện mô hình cuối cùng với tham số tốt nhất
    final_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        random_state=12,
    )
    final_model.fit(x_train, y_train)
    final_pred = final_model.predict(x_test)
    final_proba = final_model.predict_proba(x_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, final_pred):.4f}")
    print(f"Precision: {precision_score(y_test, final_pred):.4f}")
    print(f"Recall: {recall_score(y_test, final_pred):.4f}")
    
    # Lưu mô hình
    xgb_path = os.path.join('models', 'xgb_best.joblib')
    joblib.dump(final_model, xgb_path)
    
    # Vẽ tầm quan trọng của đặc trưng
    if hasattr(final_model, 'feature_importances_'):
        important = sorted(zip(final_model.feature_importances_, feature_names), reverse=True)
        plt.figure(figsize=(10, 8))
        top = important[:10]
        plt.barh([x[1] for x in top], [x[0] for x in top])
        plt.title(f'Top 10 Features (lr={best_params["learning_rate"]}, depth={best_params["max_depth"]})')
        plt.tight_layout()
        plt.savefig('xgb_feature_importance_best.png')
    
    return final_model, final_pred, final_proba

if __name__ == "__main__":
    model, pred, proba = train_xgb()