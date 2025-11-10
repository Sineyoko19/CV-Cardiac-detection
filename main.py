import json
import pandas as pd
from models.eval import model_eval
from models.models_creator import CnnCardiacDetectorModel, ResNetCardiacDetectorModel

if __name__ == "__main__":

    with open("config.json", "r") as f:
        configs = json.load(f)

    result = {}
    result["CNN"] = model_eval(CnnCardiacDetectorModel(), configs["eval"]["checkpoint_path"][0])
    result["Resnet"] = model_eval(ResNetCardiacDetectorModel(), configs["eval"]["checkpoint_path"][1])

    print(pd.DataFrame(result))