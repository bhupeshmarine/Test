input_example = pd.DataFrame([{
    "settings_path": settings_path,
    "env": "test_agent_v1"
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
