import requests

app_url = "PASTE_YOUR_APP_URL_HERE"

response = requests.post(
    f"{app_url}/invoke",
    json={
        "country": "US",
        "prdm_naics_code": "541511",
        "bbg_naics_code": "541511"
    }
)

print("Status code:", response.status_code)
print("Response:", response.json())
