import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib
import os

def train_rdf():
    # Load dữ liệu
    x_train = np.load('data/x_train.npy')
    x_test = np.load('data/x_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    feature_names = np.load('data/feature_names.npy')
    
    # Kết hợp train và test để cross-validation
    X = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    
    # Mở rộng phạm vi max_depth
    max_depth_values = [5, 10, 15, 20]
    cv_scores = []
    test_scores = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
    
    for depth in max_depth_values:
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=depth,
            min_samples_split=2,
            random_state=12
        )
        
        # Cross-validation
        cv_results = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        cv_scores.append(np.mean(cv_results))
        
        # Huấn luyện trên tập train
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_scores.append(test_acc)
        
        print(f"max_depth={depth}:")
        print(f"  CV Accuracy: {np.mean(cv_results):.4f} (±{np.std(cv_results):.4f})")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"  Recall: {recall_score(y_test, y_pred):.4f}")
        
        # Vẽ confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix (max_depth={depth})")
        plt.ylabel('Reality')
        plt.xlabel('Predict')
        plt.gca().invert_yaxis()
        plt.savefig(f'rdf_depth{depth}.png')
        plt.close()
    
    # Vẽ biểu đồ so sánh accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(max_depth_values, cv_scores, marker='o', label='Cross_Val')
    plt.plot(max_depth_values, test_scores, marker='s', label='Test')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.title('RandomForest Accuracy vs max_depth')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('rdf_max_depth_comparison.png')
    
    # Chọn max_depth tốt nhất
    best_depth_index = np.argmax(cv_scores)
    best_depth = max_depth_values[best_depth_index]
    print(f"\nBest max_depth: {best_depth}")
    
    # Huấn luyện mô hình cuối cùng với max_depth tốt nhất
    final_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=best_depth,
        min_samples_split=2,
        random_state=12
    )
    final_model.fit(x_train, y_train)
    
    final_pred = final_model.predict(x_test)
    final_proba = final_model.predict_proba(x_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, final_pred):.4f}")
    print(f"Precision: {precision_score(y_test, final_pred):.4f}")
    print(f"Recall: {recall_score(y_test, final_pred):.4f}")
    
    # Lưu mô hình
    rdf_path = os.path.join('models', 'rdf_best.joblib')
    joblib.dump(final_model, rdf_path)
    
    # Vẽ tầm quan trọng của đặc trưng
    if hasattr(final_model, 'feature_importances_'):
        important = sorted(zip(final_model.feature_importances_, feature_names), reverse=True)
        plt.figure(figsize=(10, 8))
        top = important[:10]
        plt.barh([x[1] for x in top], [x[0] for x in top])
        plt.title(f'Top 10 Features (max_depth={best_depth})')
        plt.tight_layout()
        plt.savefig('feature_importance_rdf.png')
        
    return final_model, final_pred, final_proba

if __name__ == "__main__":
    model, pred, proba = train_rdf()