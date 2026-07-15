from databricks.sdk import WorkspaceClient
import requests

# 1. Get the OAuth client ID of your Databricks App
w = WorkspaceClient()

app_name = "data-acquisition-agent"

app_client_id = w.apps.get(
    app_name
).oauth2_app_client_id

print("App OAuth client ID retrieved successfully")
