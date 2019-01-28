import requests
import api_keys

subscription_key = api_keys.azure_free

face_url = "https://westeurope.api.cognitive.microsoft.com/face/v1.0/detect?returnFaceId=true&" + \
           "returnFaceLandmarks=true&returnFaceAttributes=emotion"

headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-Type': 'application/octet-stream'
}


def query_picture(image_path):
    print("Querying image: {}".format(image_path))
    image_data = open(image_path, "rb").read()
    response = requests.post(
        face_url,
        headers=headers,
        data=image_data
    )
    raw_json = response.json()
    print(raw_json)
    return raw_json
