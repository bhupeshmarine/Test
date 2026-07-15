import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from mlflow.models import infer_signature


# 1. Create / select experiment
mlflow.set_experiment("/Shared/simple_endpoint_test")


# 2. Create simple test data
X = pd.DataFrame({
    "x1": [0, 1, 2, 3, 4, 5],
    "x2": [0, 0, 1, 1, 2, 2]
})

y = [0, 0, 0, 1, 1, 1]


# 3. Train model
model = LogisticRegression()
model.fit(X, y)


# 4. Create signature
predictions = model.predict(X)

signature = infer_signature(
    X,
    predictions
)


# 5. Log model
with mlflow.start_run() as run:

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X.iloc[:2]
    )

    print("Run ID:", run.info.run_id)
