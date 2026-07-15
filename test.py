import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from mlflow.models import infer_signature


# Select experiment
mlflow.set_experiment("/Shared/simple_endpoint_test")

# Disable automatic MLflow logging
mlflow.autolog(disable=True)


# Create test data
X = pd.DataFrame({
    "x1": [0, 1, 2, 3, 4, 5],
    "x2": [0, 0, 1, 1, 2, 2]
})

y = [0, 0, 0, 1, 1, 1]


# Train model
model = LogisticRegression()
model.fit(X, y)


# Create signature
predictions = model.predict(X)
signature = infer_signature(X, predictions)


# Create only ONE MLflow run
with mlflow.start_run() as run:

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X.iloc[:2]
    )

    print("Run ID:", run.info.run_id)
