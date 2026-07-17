class MLInferenceModel(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input, params=None):

        results = []

        for _, row in model_input.iterrows():

            config = json.loads(row["config"])

            state = {
                "config": config
            }

            result = ml_inference_node(state)

            results.append(result)

        return pd.DataFrame(results)

#####
config = {
    "input_type": "raw",

    "model_objects": {
        "pickle_file_path": "PUT_YOUR_REAL_PICKLE_FILE_PATH_HERE"
    },

    "database_table": {
        "validation_input_table":
            "PUT_YOUR_REAL_VALIDATION_INPUT_TABLE_HERE",

        "step3_features_table":
            "PUT_YOUR_REAL_STEP3_FEATURES_TABLE_HERE",

        "validation_ML_output_table":
            "PUT_YOUR_REAL_VALIDATION_ML_OUTPUT_TABLE_HERE",

        "step3_ML_output_table":
            "PUT_YOUR_REAL_STEP3_ML_OUTPUT_TABLE_HERE"
    }
}

###
config_json = json.dumps(config)

input_example = pd.DataFrame([{
    "config": config_json
}])

sample_output = pd.DataFrame([
    ml_inference_node({
        "config": config
    })
])

signature = infer_signature(
    input_example,
    sample_output
)

###
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
            "pandas"
        ]
    )

print(logged_model_info.model_uri)
