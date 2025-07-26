from modules.preprocessing import clean_data, one_hot_encoder, select_features_for_prediction
import pandas as pd

def get_user_input():
    """Collect and preprocess user inputs for lung cancer prediction."""

    print("\nPlease answer the following questions to assess lung cancer risk and the recommended insurance plan.")
    print("👉 For yes/no questions, enter Y or N. For gender, enter M or F.\n")

    questions = [
        ("Gender (M/F): ", "gender"),
        ("Age: ", "age"),
        ("Smoking (Y/N): ", "smoking"),
        ("Yellow Fingers (Y/N): ", "yellow_fingers"),
        ("Anxiety (Y/N): ", "anxiety"),
        ("Peer Pressure (Y/N): ", "peer_pressure"),
        ("Chronic Disease (Y/N): ", "chronic_disease"),
        ("Fatigue (Y/N): ", "fatigue"),
        ("Allergy (Y/N): ", "allergy"),
        ("Wheezing (Y/N): ", "wheezing"),
        ("Alcohol Consuming (Y/N): ", "alcohol"),
        ("Coughing (Y/N): ", "coughing"),
        ("Shortness of Breath (Y/N): ", "short_breath"),
        ("Swallowing Difficulty (Y/N): ", "swallowing"),
        ("Chest Pain (Y/N): ", "chest_pain")
    ]
    
    inputs = []
    for question, var in questions:
        ans = input(question).strip().upper()
        if var == "gender":
            if ans in ['M', 'F']:
                inputs.append(ans)
            else:
                print("❌ Invalid gender. Enter M or F.")
                return get_user_input()
        elif ans in ['Y', 'N']:
            inputs.append(1 if ans == 'Y' else 0)
        else:
            try:
                inputs.append(int(ans))
            except:
                print("❌ Invalid input. Please enter again.")
                return get_user_input()
    
    return [inputs]

def predict_probability(X_input, scaler=None, model=None):
    """
    Apply preprocessing and scaler, then predict lung cancer probability.
    
    Args:
        X_input (list): User input values as a list of features.
        scaler (StandardScaler): Trained scaler.
        model (classifier): Trained classification model.
    
    Returns:
        float: Predicted probability of lung cancer.
    """

    feature_names = ['GENDER', 'AGE', 'SMOKING', 'YELLOW FINGERS', 'ANXIETY',
                    'PEER PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY',
                    'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING',
                    'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']

    df = pd.DataFrame(X_input, columns=feature_names)

    # Preprocess
    df = clean_data(df)
    df = one_hot_encoder(df)
    df = select_features_for_prediction(df)

    # Scale
    X_scaled = scaler.transform(df)

    return model.predict_proba(X_scaled)[0][1]

def insurance_recommendation(prob):
    """ Recommend insurance plan based on predicted probability. """
    
    prob_percent = prob * 100
    print("\n" + "-" * 30)
    print(f"📊 Predicted lung cancer risk: {prob:.2%}")

    if prob_percent <= 5:
        print("🟢 Low risk: No additional insurance plan needed.")
        return None
    elif prob_percent <= 20:
        level = 'Basic'
    elif prob_percent <= 40:
        level = 'Standard'
    else:
        level = 'Premium'

    plans = {
        'Basic': {
            'Hospitalized Treatment': '100-200K',
            'Hospitalized Surgery': '50-100K',
            'Non-Hospitalized Surgery': '50-100K',
            'Daily Ward Cost': '2K',
        },
        'Standard': {
            'Hospitalized Treatment': '200-300K',
            'Hospitalized Surgery': '100-200K',
            'Non-Hospitalized Surgery': '100-200K',
            'Daily Ward Cost': '3K',
            'Cancer Surgery': '50-100K',
            'Cancer Ward Cost': '2K',
        },
        'Premium': {
            'Hospitalized Treatment': '300-400K',
            'Hospitalized Surgery': '200-250K',
            'Non-Hospitalized Surgery': '200-250K',
            'Daily Ward Cost': '4K',
            'Cancer Surgery': '100-200K',
            'Cancer Ward Cost': '3K',
            'Severe Cancer Payment': '2M',
            'Critical Illness Payment': '1.5M'
        }
    }

    print(f"📌 Recommended insurance plan: {level}")
    for item, amount in plans[level].items():
        print(f"  - {item}: {amount}")

def run_insurance_application(model, scaler=None):

    user_input = get_user_input()
    prob = predict_probability(user_input, scaler, model)
    insurance_recommendation(prob)
