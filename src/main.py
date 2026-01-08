import json
import pandas as pd
<<<<<<< HEAD
from .eval import model_eval
from .models_creator import CnnCardiacDetectorModel, ResNetCardiacDetectorModel
=======
from eval import model_eval
from models_creator import CnnCardiacDetectorModel, ResNetCardiacDetectorModel
>>>>>>> 810d20ac4cc7e683ef29d23edf1cf8a7656c3819

if __name__ == "__main__":

    with open("src/config.json", "r") as f:
        configs = json.load(f)

    result = {}
    result["CNN"] = model_eval(CnnCardiacDetectorModel(), configs["eval"]["checkpoint_path"][0])
    result["Resnet"] = model_eval(ResNetCardiacDetectorModel(), configs["eval"]["checkpoint_path"][1])

    print(pd.DataFrame(result))