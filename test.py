from mlflow.agent_server import invoke


@invoke()
def data_acquisition_agent(request):
    return {
        "agent_id": 2,
        "current_stage": "data_acquisition",
        "status": "test_success",
        "message": "Data Acquisition Agent is running"
    }


import agent

from mlflow.agent_server import AgentServer

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app
