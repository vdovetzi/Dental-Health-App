import base64

import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
from model import MouthSegmentationModel
import cv2 as cv
import sqlite3

app = Flask(__name__)
api = Api(app)

BITE = 3
COLOR = 2
IMAGE_SIZE = 320


class RecommendationService:
    def __init__(self):
        self.conn = sqlite3.connect('bite.db', check_same_thread=False)
        self.cursor = self.conn.cursor()

    def fetch_recommendations_color(self, color_result):
        self.cursor.execute('SELECT description, products from colour_recommendations WHERE name == ?', (color_result,))
        color_rec = self.cursor.fetchone()
        if color_rec is None:
            return "ERROR"
        return str([color_rec, color_result])

    def fetch_recommendations_bite(self, bite_result):
        self.cursor.execute('SELECT description, products from recommendations WHERE id == ?', (bite_result,))
        bite_rec = self.cursor.fetchone()
        if bite_rec is None:
            return "ERROR"
        return str([bite_rec, bite_result])


class ColorAnalysisModel:
    def __init__(self):
        self.teeth_classification = [
            ["M1-0", [81.03947688180386, 0.8859899004057814, 2.9768595816322296]],
            ["M1-0.5", [77.71775668915578, 0.8373204651945798, 3.8867341844379233]],
            ["M1-1", [78.4543669284185, 0.8305754190757186, 5.1450645807798345]],
            ["M1.5-1", [80.13395947214302, 0.8529191496197686, 6.417520252699371]],
            ["M2-1", [62.144995801931316, 0.08521748679662933, 5.749631169589642]],
            ["M2-1.5", [77.16293115225228, 0.7041237476859141, 0.7041237476859141]],
            ["M2-2", [73.05630494966022, 0.7363584694365377, 8.185205672723871]],
            ["M2-2.5", [71.94195429081635, 1.1236007517185786, 9.187014519886771]],
            ["M2-3", [76.01491626199146, 1.5793407574538065, 10.898127169999983]],
            ["M2-3.5", [66.17690036379452, 2.1228072379733676, 10.263142874411301]],
            ["M2-4", [68.16364612274106, 2.430132116825523, 12.16512674694512]],
            ["M2-4.5", [64.61949239354921, 2.8924542596560077, 12.945466588666243]],
            ["M2-5", [62.51902578744688, 3.116126266393293, 15.255148037868604]],
            ["M2.5-5", [64.43170498609842, 2.8019454501201912, 16.995660599124495]],
            ["M3-5", [65.19020672917452, 2.882759242881061, 21.240407217182767]]
        ]

    @staticmethod
    def resize_image(images):
        resized_images = []
        for image in images:
            resized_image = cv.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            resized_images.append(resized_image)
        return resized_images

    def analyze_photo(self, photo):
        photo = self.improve_brightness(photo)
        resized_photos = self.resize_image(photo)
        model = MouthSegmentationModel()
        teeth = model.predict(resized_photos)
        answers = {}
        for i in range(len(teeth[0])):
            first_ans = self.get_color(teeth[0][i])
            second_ans = self.get_color(teeth[1][i])
            if first_ans == second_ans:
                if first_ans in answers:
                    answers[first_ans] += 1
                else:
                    answers[first_ans] = 1
        try:
            return sorted(answers.items(), key=lambda x: -x[1])[0][0]
        except IndexError:
            return ("Probably, you are trying to analyze two photos of different people or some of your photos were "
                    "taken in bad conditions -> choose another photos or take new ones")

    def get_color(self, photo):
        def calculate_delta(color1, color2):
            return np.sqrt((color1[0] - color2[0]) ** 2 + (color1[1] - color2[1]) ** 2 + (color1[2] - color2[2]) ** 2)

        lab_photo = cv.cvtColor(photo, cv.COLOR_BGR2LAB)
        mean = cv.mean(lab_photo)
        color = [mean[0] * 100 / 255, mean[1] - 128, mean[2] - 128]
        answer_color = None
        min_delta = float('inf')
        for classification in self.teeth_classification:
            delta = calculate_delta(color, classification[1])
            if delta < min_delta:
                min_delta = delta
                answer_color = classification[0]
        return answer_color

    def improve_brightness(self, photo):
        improved = []
        for picture in photo:
            photo_dec = cv.imdecode(picture, cv.IMREAD_COLOR)
            photo_gray = cv.cvtColor(photo_dec, cv.COLOR_BGR2GRAY)
            brightness_now = np.mean(photo_gray)
            brightness_target = 128.6
            scale = brightness_target / brightness_now
            adjusted_photo = np.clip(photo_dec * scale, 0, 255).astype(np.uint8)
            improved.append(adjusted_photo)
        return improved


class BiteAnalysisModel:
    def analyze_photo(self, photo):
        return 1


class PhotoAnalysisService:
    def analyze_color(self, photo):
        model = ColorAnalysisModel()
        return model.analyze_photo(photo)

    def analyze_bite(self, photo):
        model = BiteAnalysisModel()

        return model.analyze_photo(photo)


class APIController:
    def __init__(self):
        self.analysis_service = PhotoAnalysisService()

    def handle_photo_upload(self, number):
        try:
            for i in range(number):
                base64.b64decode(request.get_json().get(f'user_photo_{i}'))
        except:
            return f"Upload {number} photos again", 400

        return request.get_json(), 201

    def handle_analysis_request(self, number):
        photo = []
        try:
            for i in range(number):
                photo_now = base64.b64decode(request.get_json().get(f'user_photo_{i}'))
                photo.append(np.frombuffer(photo_now, np.uint8))
        except:
            return "Upload different pictures, please", 400
        if number == COLOR:
            color_result = self.analysis_service.analyze_color(photo)
            return str(color_result), 201
        else:
            bite_result = self.analysis_service.analyze_bite(photo)
            return str(bite_result), 201

    def provide_recommendations(self, number):
        results = request.get_json()
        ans = results.get('analysis_result')
        service = RecommendationService()
        if number == COLOR:
            recommendations = service.fetch_recommendations_color(ans)
        else:
            recommendations = service.fetch_recommendations_bite(ans)
        if recommendations == "ERROR":
            return "Can't procces the recommendations", 400
        return recommendations, 200


controller = APIController()


@app.route('/handle_photo_upload_color', methods=['POST'])
def upload_color():
    return controller.handle_photo_upload(COLOR)


@app.route('/handle_photo_upload_bite', methods=['POST'])
def upload_bite():
    return controller.handle_photo_upload(BITE)


@app.route('/handle_analysis_request_bite', methods=['POST'])
def request_analysis_bite():
    return controller.handle_analysis_request(BITE)


@app.route('/handle_analysis_request_color', methods=['POST'])
def request_analysis_color():
    return controller.handle_analysis_request(COLOR)


@app.route('/provide_recommendations_color', methods=['GET'])
def provide_color():
    return controller.provide_recommendations(COLOR)


@app.route('/provide_recommendations_bite', methods=['GET'])
def provide_bite():
    return controller.provide_recommendations(BITE)


if __name__ == '__main__':
    app.run(debug=True)
