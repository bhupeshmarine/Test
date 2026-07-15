print("Status code:", response.status_code)
print("Content-Type:", response.headers.get("content-type"))
print("Response URL:", response.url)
print("Response text:", response.text[:1000])
