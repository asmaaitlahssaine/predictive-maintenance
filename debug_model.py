import joblib

model = joblib.load("models/best_model.joblib")

print("MODEL FEATURE NAMES:")
try:
    print(model.feature_names_in_)
except:
    print("Model has no feature_names_in_ attribute.")
