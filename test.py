import mlflow
import pandas as pd

# Replace with your actual registered model name and version
model_uri = "models:/catalog.schema.ml_inference_model/1"

# Load registered model
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Prepare input
settings_path = f"{PROJECT_ROOT}/configs/settings.yaml"

test_input = pd.DataFrame([{
    "settings_path": settings_path,
    "env": "test_agent_v1"
}])

# Run prediction
result = loaded_model.predict(test_input)

display(result)
