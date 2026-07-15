from mlflow.genai.agent_server import invoke
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)


@invoke()
def data_acquisition_agent(
    request: ResponsesAgentRequest
) -> ResponsesAgentResponse:

    return ResponsesAgentResponse(
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Data Acquisition Agent is running",
                        "annotations": []
                    }
                ]
            }
        ],
        custom_outputs={
            "agent_id": 2,
            "current_stage": "data_acquisition",
            "status": "test_success"
        }
    )
