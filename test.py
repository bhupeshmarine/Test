class MLInferenceModel(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input, params=None):

        results = []

        for _, row in model_input.iterrows():

            config_state = {
                "settings_path": row["settings_path"],
                "env": row["env"]
            }

            config_result = config_context_node(config_state)

            state = {
                "config": config_result["config"]
            }

            result = ml_inference_node(state)

            results.append(result)

        return pd.DataFrame(results)






settings_path = f"{PROJECT_ROOT}/configs/settings.yaml"

input_example = pd.DataFrame([{
    "settings_path": settings_path,
    "env": "test_agent_v1"
}])

sample_output = pd.DataFrame([
    ml_inference_node({
        "config": config_context_node({
            "settings_path": settings_path,
            "env": "test_agent_v1"
        })["config"]
    })
])

signature = infer_signature(
    input_example,
    sample_output
)






























































