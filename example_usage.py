import base64
import requests
image1 = "examples/Darkened Photo.jpg"
image2 = "examples/Darkened Photo.jpg"
with open(image1, "rb") as img:
    string1 = base64.b64encode(img.read()).decode('utf-8')
with open(image2, "rb") as img:
    string2 = base64.b64encode(img.read()).decode('utf-8')


api_url = "http://127.0.0.1:5000/handle_photo_upload_color"
response = requests.post(url= api_url, json={'user_photo_0':string1, 'user_photo_1':string2})
print(response.text)
api_url = "http://127.0.0.1:5000/handle_analysis_request_color"
response = requests.post(url = api_url, json={'user_photo_0':string1, 'user_photo_1':string2})
print(response.text)
api_url = "http://127.0.0.1:5000/provide_recommendations_color"
response = requests.get(url = api_url, json={'analysis_result':response.text})

print(response.text)

api_url = "http://127.0.0.1:5000/provide_recommendations_bite"
response = requests.get(url = api_url, json={'analysis_result':1})
# print(response.text)
