# modules/modeling.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "Random Forest": RandomForestClassifier(class_weight='balanced'),
        "SVM": SVC(class_weight='balanced', probability=True),
        "XGBoost": XGBClassifier(scale_pos_weight=6.7, eval_metric='logloss', random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models

def evaluate_models(models, X_test, y_test):
    results = {}

    print("\nModel Evaluation Summary:")
    print("-" * 80)
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Print one-line summary
        print(f"{name:<20} | Acc: {acc:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f} | F1: {f1:.2f} | AUC: {auc:.2f}")

        results[name] = {
            "confusion_matrix": cm,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc
        }
    
    return results

def plot_roc(models, X_test, y_test):
    plt.figure(figsize=(10, 6))
    
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            continue
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name}")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/roc_curve.png")
    plt.close()

def select_best_model_weighted(results, models, weights=None):
    if weights is None:
        weights = {
            'recall': 0.5,
            'roc_auc': 0.25,
            'f1_score': 0.15,
            'accuracy': 0.05,
            'precision': 0.05
        }

    scores = {}
    for name, metrics in results.items():
        total = sum(metrics[k] * w for k, w in weights.items())
        scores[name] = total

    best_model_name, best_score = max(scores.items(), key=lambda x: x[1])
    print(f"\n Best Overall Model: {best_model_name} (Score = {best_score:.2f})")

    return models[best_model_name]