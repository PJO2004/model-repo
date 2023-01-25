import os

from model.jw_climate_prediction import runLSTM

DEFAULT_DIR: str = f"{os.getcwd()}{os.sep}"

def Prediction(csv_path: str = None):
    lstm = runLSTM()
    lstm.data_prediction(csv_path=f"{DEFAULT_DIR}indir/{csv_path}")