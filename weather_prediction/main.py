"""_summary_
기온 예측 모델 
model : lstm

model 설명
- in_dir의 값에는 p (mbar) ,T (degC), rho (m/g**3), Date Time 컬럼은 필수 입니다.
- 테스트 값 들은 기본으로 120개의 low값 (최소 2일치 값)을 넣어 주어야 합니다.
- 너무 추운경우에는 예측시가 정확하게 나오지 못하고 있습니다.
"""
import os

from tryModel.train_model import Train
from tryModel.test_model import Prediction

def main(in_dir: str = None, trainOX: bool = False):
    if trainOX and os.path.isfile("jw_climate_prediction_model.h5"):
        Prediction(csv_path=in_dir)

    else:
        Train()
        Prediction(csv_path=in_dir)

if __name__ == "__main__":
    main(in_dir='jena_climate_2009_2016.csv', trainOX=True)