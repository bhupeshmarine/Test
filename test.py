from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

print("Workspace host:", w.config.host)
print("Authentication configured:", w.config.auth_type)
