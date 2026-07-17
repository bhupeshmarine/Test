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
