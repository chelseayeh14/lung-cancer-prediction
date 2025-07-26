from modules.preprocessing import preprocessing
from modules.modeling import train_models, evaluate_models, plot_roc, select_best_model_weighted
from modules.application import run_insurance_application

# Only run this block when the script is executed directly
if __name__ == "__main__":
    
    # Preprocessing
    X_train, X_test, y_train, y_test, scaler = preprocessing("data/survey_lung_cancer.csv")

    # Modeling
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    plot_roc(models, X_test, y_test)
    selected_model = select_best_model_weighted(results, models)

    # Application
    run_insurance_application(selected_model, scaler=scaler)

    print("\nAll done!")
