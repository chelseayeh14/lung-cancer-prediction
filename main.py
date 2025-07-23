from modules.preprocessing import preprocessing
from modules.modeling import train_models, evaluate_models, plot_roc, select_best_model_weighted

# Only run this block when the script is executed directly
if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = preprocessing("data/survey_lung_cancer.csv")

    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    plot_roc(models, X_test, y_test)
    select_best_model_weighted(results)