from mlflow.genai.agent_server import invoke


@invoke()
def invoke_agent(request):
    return {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Data Acquisition Agent is running"
                    }
                ]
            }
        ]
    }
