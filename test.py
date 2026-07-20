import subprocess

result = subprocess.run(
    ["pip", "install", "databricks-langchain==0.2.0", "-v"],
    capture_output=True,
    text=True
)

print(result.stderr[-5000:])
