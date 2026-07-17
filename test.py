# ============================================================
# CELL 1 - Imports and project setup
# ============================================================

import os
import sys
import yaml
import mlflow
import pandas as pd

from mlflow.models.signature import infer_signature


PROJECT_ROOT = "/Workspace/Shared/Global_Entity_Resolution/final Workflow/SAMPLE_LangGraph_Agent_ml_flow_V1"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


for pkg in [
    "agents",
    "agents/search_expansion_agent",
    "tools"
]:
    init_path = os.path.join(PROJECT_ROOT, pkg, "__init__.py")

    if not os.path.exists(init_path):
        open(init_path, "w").close()


from agents.search_expansion_agent.agent import search_expansion_node



# ============================================================
# CELL 2 - Load config
# ============================================================

def load_config(settings_path, env):

    with open(settings_path, "r") as file:
        settings = yaml.safe_load(file)

    if env in settings:
        return settings[env]

    if "environments" in settings and env in settings["environments"]:
        return settings["environments"][env]

    if "env" in settings and env in settings["env"]:
        return settings["env"][env]

    if "database_table" in settings:
        return settings

    raise ValueError(
        f"Could not find environment '{env}' "
        f"in settings file: {settings_path}"
    )


# ============================================================
# CELL 3 - MLflow wrapper
# ============================================================

class SearchExpansionModel(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input, params=None):

        results = []

        for _, row in model_input.iterrows():

            config = load_config(
                settings_path=row["settings_path"],
                env=row["env"]
            )

            state = {
                "config": config
            }

            result = search_expansion_node(state)

            results.append(result)

        return pd.DataFrame(results)


# ============================================================
# CELL 4 - Input example, sample output, signature
# ============================================================

settings_path = f"{PROJECT_ROOT}/configs/settings.yaml"


input_example = pd.DataFrame([{
    "settings_path": settings_path,
    "env": "test_agent_v1"
}])


sample_config = load_config(
    settings_path=settings_path,
    env="test_agent_v1"
)


sample_output = pd.DataFrame([
    search_expansion_node({
        "config": sample_config
    })
])


signature = infer_signature(
    input_example,
    sample_output
)


# ============================================================
# CELL 5 - Log model
# ============================================================

with mlflow.start_run(
    run_name="search_expansion_node"
) as run:

    logged_model_info = mlflow.pyfunc.log_model(

        artifact_path="model",

        python_model=SearchExpansionModel(),

        input_example=input_example,

        signature=signature,

        code_path=[
            f"{PROJECT_ROOT}/agents",
            f"{PROJECT_ROOT}/tools"
        ],

        pip_requirements=[
            "pandas",
            "pyyaml"
        ]
    )


print(logged_model_info.model_uri)


# ============================================================
# CELL 6 - Register model
# ============================================================

model_name = "mrd_red_dev.ered_gold.search_expansion_agent"

registered_model = mlflow.register_model(
    model_uri=logged_model_info.model_uri,
    name=model_name
)

print(
    "Registered model version:",
    registered_model.version
)
