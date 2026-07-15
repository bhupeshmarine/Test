from mlflow.genai.agent_server import invoke


@invoke()
def data_acquisition_agent(request):
    return {
        "agent_id": 2,
        "country": "US",
        "current_stage": "data_acquisition",
        "status": "test_success",
        "history": [
            {
                "agent_id": 2,
                "stage": "data_acquisition",
                "status": "completed"
            }
        ],
        "received_request": request
    }
