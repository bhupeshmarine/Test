import mlflow
import pandas as pd

mlflow.set_registry_uri("databricks-uc")

model_uri = "models:/mrd_ered_dev.ered_gold.gedr_training_session/1"

model = mlflow.pyfunc.load_model(model_uri)

print("Model loaded successfully")


test_input = pd.DataFrame([
    [0.95, 0.88, 0.91, 0.76, 0.84]
])

prediction = model.predict(test_input)

print("Prediction:", prediction)




test_input = (
    split["X_test_scaled"][:5]
    if best["best_needs_scale"]
    else split["X_test"][:5]
)

prediction = model.predict(test_input)

print("Input shape:", test_input.shape)
print("Predictions:", prediction)
