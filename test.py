state = {
    "config": {
        "input_type": row["input_type"],
        "model_objects": {
            "pickle_file_path": row["model_file"]
        },
        "database_table": {
            "validation_input_table": row["validation_input_table"],
            "step3_features_table": row["step3_features_table"],
            "validation_ML_output_table": row["validation_ML_output_table"],
            "step3_ML_output_table": row["step3_ML_output_table"]
        }
    }
}




result = ml_inference_node(state)



class MLInferenceModel(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input, params=None):

        results = []

        for _, row in model_input.iterrows():

            state = {
                "config": {
                    "input_type": row["input_type"],

                    "model_objects": {
                        "pickle_file_path": row["model_file"]
                    },

                    "database_table": {
                        "validation_input_table":
                            row["validation_input_table"],

                        "step3_features_table":
                            row["step3_features_table"],

                        "validation_ML_output_table":
                            row["validation_ML_output_table"],

                        "step3_ML_output_table":
                            row["step3_ML_output_table"]
                    }
                }
            }

            result = ml_inference_node(state)

            results.append(result)

        return pd.DataFrame(results)



input_example = pd.DataFrame([{

    "input_type": "raw",

    "model_file":
        "/your/model/path/model.pkl",

    "validation_input_table":
        "catalog.schema.validation_input",

    "step3_features_table":
        "catalog.schema.step3_features",

    "validation_ML_output_table":
        "catalog.schema.validation_ml_output",

    "step3_ML_output_table":
        "catalog.schema.step3_ml_output"
}])



sample_state = {
    "config": {
        "input_type": "raw",

        "model_objects": {
            "pickle_file_path":
                "/your/model/path/model.pkl"
        },

        "database_table": {
            "validation_input_table":
                "catalog.schema.validation_input",

            "step3_features_table":
                "catalog.schema.step3_features",

            "validation_ML_output_table":
                "catalog.schema.validation_ml_output",

            "step3_ML_output_table":
                "catalog.schema.step3_ml_output"
        }
    }
}

sample_output = pd.DataFrame([
    ml_inference_node(sample_state)
])



signature = infer_signature(
    input_example,
    sample_output
)


with mlflow.start_run(
    run_name="ml_inference_node"
) as run:

    logged_model_info = mlflow.pyfunc.log_model(

        artifact_path="model",

        python_model=MLInferenceModel(),

        input_example=input_example,

        signature=signature,

        code_path=[
            "agents",
            "tools"
        ],

        pip_requirements=[
            "pandas"
        ]
    )

print(logged_model_info.model_uri)
