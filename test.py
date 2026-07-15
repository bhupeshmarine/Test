import agent

from mlflow.genai.agent_server import AgentServer

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app



command:
  - gunicorn
  - start_server:app
  - --bind
  - 0.0.0.0:8000
