import subprocess

result = subprocess.run(
    ["pip", "install", "databricks-langchain", "-v"],
    capture_output=True,
    text=True
)

print(result.stdout)
print(result.stderr)
