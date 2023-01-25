"""_summary_
모델 => LSTM
"""
import tensorflow as tf
import pandas as pd
import numpy as np

class runLSTM:
    """_summary_
    LSTM Model
    """
    def __init__(self) -> None:
        """_summary_
        features_considered : 대기압, 대기온도, 공기밀도
        model_train_percent : 80%
        model_validation_percent : 20%
        
        future_considered : outdir의 csv데이터 컬럼
        
        normalization
        - dataset_mean
        - dataset_standard_deviation
        
        model
        - buffer_size : shuffle시키기 위해서 필요한 파람
        - batch_size : 모델 학습 배치 사이즈 지정
        
        history_size : 5일 전의 데이터를 기준으로 예측 훈련 (24 * 6 * 5)
        future_size : 하룻 동안의 대기 온도 값을 예측 (24 * 6)
        step : 10분간의 예측을 1시간 마다의 예측 값으로
        epoch : 모델 학습 량
        steps_per_epoch : 학습 량
        validation_steps : 테스트 량
        """
        self.features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
        self.features_index = 'Date Time'
        self.model_train_percent = 0.8
        self.model_name = 'jw_climate_prediction_model.h5'
        
        self.future_considered = ['Date Time', 'T (degC)']
        
        self.df = None
        self.features = None
        self.train_split = None
        self.dataset_mean = None
        self.dataset_standard_deviation = None
        self.dataset = None
        
        self.history_size = 720
        self.future_size = 144
        self.step = 6
        self.epoch = 10
        self.steps_per_epoch = 200
        self.validation_steps = 50
        
        self.buffer_size = 10000
        self.batch_size = 256
        
        self.x_train = []
        self.y_train = []
        self.x_validation = []
        self.y_validation = []

    def train(self):
        """_summary_
            model train
            
            Dense : 뉴련 입/출력 -> predict 에서도 사용
        """
        
        train_data = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        train_data = train_data.cache().shuffle(buffer_size=self.buffer_size).batch(batch_size=self.batch_size).repeat()

        validation_data = tf.data.Dataset.from_tensor_slices((self.x_validation, self.y_validation))
        validation_data = validation_data.batch(batch_size=self.batch_size).repeat()
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=self.x_train.shape[-2:]))
        model.add(tf.keras.layers.LSTM(16, activation='relu'))
        model.add(tf.keras.layers.Dense(self.future_size))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
        
        history = model.fit(train_data, epochs=self.epoch, steps_per_epoch=self.steps_per_epoch, 
                            validation_data=validation_data, validation_steps=self.validation_steps)
        
        model.save(self.model_name)
        print(f"""
                                                Train Successful
        ==============================================================================================
        model_name = {self.model_name}
        ==============================================================================================
        """)

    def data_prediction(self, csv_path: str = None):
        """_summary_
            이전 데이터들을 통해서 예측
        """
        self.data_load(csv_path=csv_path, prediction=True)
        
        model = tf.keras.models.load_model(self.model_name)
        dataset = np.array(self.dataset).reshape(1, 120, 3)
        
        predict_T_degC = model.predict(dataset)
        future_T_degC = (predict_T_degC * self.dataset_standard_deviation[1]) + self.dataset_mean[1]
        
        print(f"""
                                              predict Successful
        ==============================================================================================
        - future_T_degC
        
        {future_T_degC}
        ==============================================================================================
        """)
        self.data_upload(future_T_degC=future_T_degC)

    def data_load(self, csv_path: str = None, prediction: bool = False):
        """_summary_
            csv file load
        """
        self.df = pd.read_csv(csv_path)
        
        self.data_preprocessing(prediction=prediction)
        print(f"""
                                                    Train Set
        ==============================================================================================
        train_file = {csv_path}
        
        dataset_length = {len(self.dataset)}
        train_data_length = {self.train_split}
        
        dataset_mean = {self.dataset_mean}
        dataset_std = {self.dataset_standard_deviation}
        ==============================================================================================
        """)

    def data_preprocessing(self, prediction: bool = False):
        """_summary_
            데이터 전처리
        """
        self.features = self.df[self.features_considered]
        self.features.index = self.df[self.features_index]
        self.train_split = int(self.df.shape[0] * self.model_train_percent)
        
        if prediction:
            self.dataset = self.features.tail(120).values
        else:
            self.dataset = self.features.values

        self.dataset_mean = self.dataset[:self.train_split].mean(axis=0)
        self.dataset -= self.dataset_mean

        self.dataset_standard_deviation = self.dataset[:self.train_split].std(axis=0)
        self.dataset /= self.dataset_standard_deviation

    def data_setting(self):
        """_summary_
        x_train : 모델 학습용 -> 대기압, 대기온도, 공기밀도
        y_train : 모델 학습 정답 -> 대기 온도
        
        x_validation : 모델 테스트용 -> ''
        y_validation : 모델 테스트용 정답 -> ''
        """
        for index in range(self.history_size, self.train_split):
            indices = range(index - self.history_size, index, self.step)
            self.x_train.append(self.dataset[indices])
            self.y_train.append(self.dataset[:, 1][index : index + self.future_size])

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        
        for index in range(self.train_split + self.history_size, len(self.dataset) - self.future_size):
            indices = range(index - self.history_size, index, self.step)
            self.x_validation.append(self.dataset[indices])
            self.y_validation.append(self.dataset[:, 1][index : index + self.future_size])
            
        self.x_validation = np.array(self.x_validation)
        self.y_validation = np.array(self.y_validation)
        
        print(f"""
                                            Train Setting successful
        ==============================================================================================
        x_train_shape = {self.x_train.shape}
        y_train_shape = {self.y_train.shape}
        
        x_validation_shape = {self.x_validation.shape}
        y_validation_shape = {self.y_validation.shape}
        ==============================================================================================
        """)

    def data_upload(self, future_T_degC = None):
        """_summary_
            csv file upload
            
            date :  Date Time -> 인덱스가 될 예정
                    Date Time 형식 : 01.01.01 01:10:00
        """
        date = ''
        year_month_day = self.features.tail(120).index[-1].split(' ')[0].split('.')
        start = year_month_day
        hour_minute_second = self.features.tail(120).index[-1].split(' ')[1].split(':')
        future = []
        
        for _ in range(120):
            h, m, s = hour_minute_second
            h, m, s = int(h), int(m), s
            
            d, mo, y = year_month_day
            d, mo, y = int(d), int(mo), int(y)
            
            if m == 50:
                hour_minute_second = [h+1, '00', s]
                
                if h == 23:
                    hour_minute_second = ['00', '00', s]
                    year_month_day = [d+1, mo, y]
                    
            else:
                hour_minute_second = [h, m + 10, s]
                
            date = f'{year_month_day[0]}.{year_month_day[1]}.{year_month_day[2]} {hour_minute_second[0]}:{hour_minute_second[1]}:{hour_minute_second[2]}'
            future.append([date, future_T_degC[0][_]])
        
        future = pd.DataFrame(future, columns=self.future_considered)
        future = future.set_index(keys=self.features_index, inplace=False, drop=True)
        future.to_csv(f"outdir/{start}_{year_month_day}.csv")
        
        print(f"""
                                              upload successful
        ==============================================================================================
        csv_file_name : {start}_{year_month_day}.csv
        ==============================================================================================
        """)