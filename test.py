# ============================================================
# CELL 2 - Helper function to load config from settings.yaml
# ============================================================

def load_config(settings_path, env):

    with open(settings_path, "r") as file:
        settings = yaml.safe_load(file)

    # If environment names are directly at the top level
    if env in settings:
        return settings[env]

    # If environments are stored under "environments"
    if "environments" in settings and env in settings["environments"]:
        return settings["environments"][env]

    # If environments are stored under "env"
    if "env" in settings and env in settings["env"]:
        return settings["env"][env]

    # If settings.yaml itself is already the full config
    if (
        "database_table" in settings
        and "model_objects" in settings
    ):
        return settings

    raise ValueError(
        f"Could not find environment '{env}' "
        f"in settings file: {settings_path}"
    )


####
# ============================================================
# CELL 3 - MLflow wrapper
# ============================================================

class MLInferenceModel(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input, params=None):

        results = []

        for _, row in model_input.iterrows():

            settings_path = row["settings_path"]
            env = row["env"]

            # Load the real configuration from settings.yaml
            config = load_config(
                settings_path=settings_path,
                env=env
            )

            # ml_inference_node expects state["config"]
            state = {
                "config": config
            }

            result = ml_inference_node(state)

            results.append(result)

        return pd.DataFrame(results)

#####
# ============================================================
# CELL 4 - Input example, sample output, signature
# ============================================================

settings_path = f"{PROJECT_ROOT}/configs/settings.yaml"


input_example = pd.DataFrame([{
    "settings_path": settings_path,
    "env": "test_agent_v1"
}])


# Load the SAME real config that the wrapper will use
sample_config = load_config(
    settings_path=settings_path,
    env="test_agent_v1"
)


# Run the actual ML inference node to determine output structure
sample_output = pd.DataFrame([
    ml_inference_node({
        "config": sample_config
    })
])


signature = infer_signature(
    input_example,
    sample_output
)



#####
# ============================================================
# CELL 5 - Log model to MLflow
# ============================================================

with mlflow.start_run(
    run_name="ml_inference_node"
) as run:

    logged_model_info = mlflow.pyfunc.log_model(

        artifact_path="model",

        python_model=MLInferenceModel(),

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
