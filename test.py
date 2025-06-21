import requests

url = "https://childface-recoginition-api.onrender.com/recognize/"

files = {'file': open('Caucasian_200_16.png', 'rb')}

response = requests.post(url, files=files)

print(response.json())
