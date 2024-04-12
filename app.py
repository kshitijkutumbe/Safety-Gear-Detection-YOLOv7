
import sys,os
from safetygear.pipeline.training_pipeline import TrainPipeline
from safetygear.exception import isdException
from safetygear.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from safetygear.configuration.s3_operations import S3Operation
from safetygear.entity.config_entity import ModelPusherConfig


app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"



@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Done"
    


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        if not os.path.exists("best.pt"):
            S3Operation().download_object(key=ModelPusherConfig.S3_MODEL_KEY_PATH,bucket_name=ModelPusherConfig.MODEL_BUCKET_NAME,filename='best.pt')

        os.system("cd yolov7/ && python detect.py --weights best.pt  --source ../data/inputImage.jpg")
        opencodedbase64 = encodeImageIntoBase64("yolov7/runs/detect/exp/inputImage.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}
        os.system("rm -rf yolov7/runs")

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080)
    
