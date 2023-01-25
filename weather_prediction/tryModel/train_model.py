import os

from model.jw_climate_prediction import runLSTM

DEFAULT_DIR: str = f"{os.getcwd()}{os.sep}"

def Train():
    lstm = runLSTM()
    lstm.data_load(f"{DEFAULT_DIR}indir/jena_climate_2009_2016.csv")
    lstm.data_setting()
    lstm.train()